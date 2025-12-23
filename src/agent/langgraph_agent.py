"""
LangGraph disruption alert agent.

State machine:

    monitor_news -> match_node -> retrieve_context -> reason_over_models -> generate_alert -> END

  1. monitor_news       — fetch live news via NewsClient (falls back to
                           synthetic disruption headlines if NEWSAPI_KEY is
                           unset or the call fails).
  2. match_node          — heuristically match the news item to a graph node
                           (by country/region keyword overlap), falling back
                           to the highest structural-risk node if no match.
  3. retrieve_context    — RAG retrieval over the supplier-profile + news
                           vector store for context relevant to the matched
                           node/news.
  4. reason_over_models  — pull GNN critical-node/impact predictions (falling
                           back to graph centrality heuristics if the GNN
                           hasn't been trained yet) and the cached causal ATE
                           estimate; compute a composite severity score.
  5. generate_alert      — call the LLMClient (Claude / Ollama / rule-based,
                           in that cost-aware priority order) to draft the
                           narrative and recommended actions, then assemble
                           the full structured `DisruptionAlert`.

Run:
    python -m src.agent.langgraph_agent
"""
from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.alert_schema import DisruptionAlert, Evidence, Severity
from src.agent.alert_store import append_alert
from src.agent.llm_client import LLMClient
from src.agent.news_client import NewsClient
from src.causal.causal_model import get_or_compute_causal_ate
from src.graph.build_graph import build_and_save_default_graph, load_graph, GRAPH_PATH
from src.graph.graph_utils import identify_critical_nodes, simulate_disruption_propagation
from src.logging_config import get_logger
from src.retrieval.ingest import generate_fallback_news_docs
from src.retrieval.retriever import Retriever

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    news_query: str
    news_articles: list[dict]
    news_trigger: str | None
    origin_node: str
    rag_context: list[dict]
    critical_probability: float
    predicted_impact: float
    affected_nodes: list[str]
    causal_ate: float | None
    severity: str
    alert: dict


def _load_graph_safe():
    try:
        return load_graph(GRAPH_PATH)
    except FileNotFoundError:
        logger.info("No persisted graph found; building synthetic graph for agent run")
        return build_and_save_default_graph()


def monitor_news_node(state: AgentState) -> AgentState:
    client = NewsClient()
    query = state.get("news_query", "supply chain disruption")
    articles = client.fetch_disruption_news(query=query)
    if not articles:
        logger.info("No live news available; using synthetic fallback headlines")
        fallback = generate_fallback_news_docs()
        articles = [{"title": d["text"][:80], "description": d["text"], "url": "", "published_at": "", "source_name": "synthetic"} for d in fallback]
    return {**state, "news_articles": articles, "news_trigger": articles[0]["title"] if articles else None}


def match_node_node(state: AgentState) -> AgentState:
    g = _load_graph_safe()
    articles = state.get("news_articles", [])
    text_blob = " ".join(f"{a.get('title','')} {a.get('description','')}" for a in articles).lower()

    best_match, best_score = None, 0
    for node_id, data in g.nodes(data=True):
        country = str(data.get("country", "")).lower()
        region = str(data.get("region", "")).lower()
        score = int(country and country in text_blob) + int(region and region in text_blob)
        if score > best_score:
            best_match, best_score = node_id, score

    if best_match is None:
        ranked = identify_critical_nodes(g, top_k=1)
        best_match = ranked[0][0] if ranked else next(iter(g.nodes()))
        logger.info("No news/node keyword match; defaulting to top structural-risk node %s", best_match)
    else:
        logger.info("Matched news to node %s (keyword overlap score=%d)", best_match, best_score)

    return {**state, "origin_node": best_match}


def retrieve_context_node(state: AgentState) -> AgentState:
    origin_node = state["origin_node"]
    query = f"disruption risk near {origin_node} {state.get('news_trigger') or ''}"
    try:
        retriever = Retriever()
        context = retriever.retrieve(query, k=5)
    except FileNotFoundError:
        logger.warning("Vector store not found; run `python -m src.retrieval.ingest` first. Proceeding with no RAG context.")
        context = []
    return {**state, "rag_context": context}


def reason_over_models_node(state: AgentState) -> AgentState:
    g = _load_graph_safe()
    origin_node = state["origin_node"]

    critical_probability = None
    predicted_impact = None
    try:
        from src.gnn.inference import GNNPredictor

        predictor = GNNPredictor(g)
        pred = predictor.predict_node(origin_node)
        critical_probability = pred["critical_probability"]
        predicted_impact = pred["predicted_impact"]
        logger.info("Using trained GNN predictions for %s", origin_node)
    except Exception as exc:
        logger.info("GNN unavailable/untrained (%s); falling back to graph-heuristic scoring", exc)

    if critical_probability is None:
        ranked = dict(identify_critical_nodes(g, top_k=g.number_of_nodes()))
        max_score = max(ranked.values()) if ranked else 1.0
        critical_probability = float(ranked.get(origin_node, 0.0) / (max_score or 1.0))
        sim = simulate_disruption_propagation(g, origin_node)
        total_capacity = sum(float(d.get("capacity", 0.0)) for _, d in g.nodes(data=True)) or 1.0
        predicted_impact = float(sim.total_capacity_lost / total_capacity)

    sim = simulate_disruption_propagation(g, origin_node)

    try:
        causal_result = get_or_compute_causal_ate(g)
        causal_ate = causal_result.ate
    except Exception as exc:
        logger.warning("Causal ATE unavailable (%s); proceeding without it", exc)
        causal_ate = None

    return {
        **state,
        "critical_probability": critical_probability,
        "predicted_impact": predicted_impact,
        "affected_nodes": sim.affected_nodes,
        "causal_ate": causal_ate,
    }


def _compute_severity(critical_probability: float, predicted_impact: float, causal_ate: float | None) -> Severity:
    causal_component = min(max(causal_ate or 0.0, 0.0), 1.0)
    composite = 0.4 * critical_probability + 0.4 * predicted_impact + 0.2 * causal_component
    if composite >= 0.7:
        return Severity.CRITICAL
    if composite >= 0.45:
        return Severity.HIGH
    if composite >= 0.2:
        return Severity.MEDIUM
    return Severity.LOW


def _rule_based_actions(severity: Severity, node_type: str) -> list[str]:
    actions = {
        Severity.CRITICAL: [
            "Immediately activate backup/alternate sourcing for this node and all direct downstream dependents.",
            "Notify affected manufacturers' supply chain leads within 24 hours.",
            "Expedite in-transit shipments where possible and reprioritize warehouse allocations.",
        ],
        Severity.HIGH: [
            "Increase monitoring frequency for this node and its immediate downstream neighbors.",
            "Pre-qualify a secondary supplier/route as a contingency.",
        ],
        Severity.MEDIUM: [
            "Flag for weekly review in the risk dashboard.",
            "Confirm current safety stock levels are adequate for a short-term disruption.",
        ],
        Severity.LOW: [
            "No immediate action required; continue routine monitoring.",
        ],
    }
    base = actions[severity]
    if node_type == "port":
        base = base + ["Evaluate rerouting through an alternate port to reduce dwell-time exposure."]
    return base


def generate_alert_node(state: AgentState) -> AgentState:
    g = _load_graph_safe()
    origin_node = state["origin_node"]
    node_data = g.nodes[origin_node]
    severity = _compute_severity(state["critical_probability"], state["predicted_impact"], state.get("causal_ate"))

    llm = LLMClient()
    context_snippets = "\n".join(f"- ({c['source']}) {c['text']}" for c in state.get("rag_context", []))
    system_prompt = (
        "You are a supply chain risk analyst. Given quantitative model outputs and retrieved "
        "context, write a concise (3-4 sentence) disruption alert narrative, followed by 2-3 "
        "specific recommended actions as a bulleted list. Be specific and actionable."
    )
    user_prompt = (
        f"Node: {origin_node} ({node_data.get('name')}, type={node_data.get('node_type')}, "
        f"country={node_data.get('country')})\n"
        f"Severity: {severity.value}\n"
        f"GNN critical-node probability: {state['critical_probability']:.2f}\n"
        f"Predicted downstream capacity-loss fraction: {state['predicted_impact']:.2f}\n"
        f"Causal ATE of disruption on downstream impact: {state.get('causal_ate')}\n"
        f"Affected downstream nodes: {', '.join(state.get('affected_nodes', [])[:10])}\n"
        f"News trigger: {state.get('news_trigger')}\n"
        f"Retrieved context:\n{context_snippets}\n"
    )
    narrative = llm.generate(system_prompt, user_prompt)

    alert = DisruptionAlert(
        alert_id=str(uuid.uuid4()),
        severity=severity,
        origin_node=origin_node,
        affected_nodes=state.get("affected_nodes", []),
        predicted_capacity_loss_fraction=state["predicted_impact"],
        critical_node_probability=state["critical_probability"],
        causal_ate_estimate=state.get("causal_ate"),
        news_trigger=state.get("news_trigger"),
        evidence=[Evidence(text=c["text"], source=c["source"], score=c["score"]) for c in state.get("rag_context", [])],
        summary=narrative,
        recommended_actions=_rule_based_actions(severity, str(node_data.get("node_type", ""))),
        llm_backend=llm.backend,
    )
    logger.info("Generated alert %s for node %s (severity=%s)", alert.alert_id, origin_node, severity.value)
    return {**state, "alert": alert.model_dump(mode="json")}


def build_agent_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("monitor_news", monitor_news_node)
    graph.add_node("match_node", match_node_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("reason_over_models", reason_over_models_node)
    graph.add_node("generate_alert", generate_alert_node)

    graph.set_entry_point("monitor_news")
    graph.add_edge("monitor_news", "match_node")
    graph.add_edge("match_node", "retrieve_context")
    graph.add_edge("retrieve_context", "reason_over_models")
    graph.add_edge("reason_over_models", "generate_alert")
    graph.add_edge("generate_alert", END)

    return graph.compile()


def run_agent(news_query: str = "supply chain disruption") -> DisruptionAlert:
    app = build_agent_graph()
    final_state = app.invoke({"news_query": news_query})
    alert = DisruptionAlert.model_validate(final_state["alert"])
    append_alert(alert)
    return alert


if __name__ == "__main__":
    alert = run_agent()
    logger.info("Final structured alert:\n%s", alert.model_dump_json(indent=2))
