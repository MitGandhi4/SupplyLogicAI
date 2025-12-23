"""
Ingest supplier documents and news articles into the FAISS vector store.

Two document sources:
  1. Supplier/node profile documents — generated deterministically from the
     graph (background, certifications, historical risk notes). Always
     available, no external API needed.
  2. News articles — pulled live from NewsAPI when `NEWSAPI_KEY` is set
     (see src.agent.news_client); otherwise a small set of representative
     synthetic disruption-news snippets is used so retrieval/agent demos
     still work fully offline and at zero cost.

Run:
    python -m src.retrieval.ingest
"""
from __future__ import annotations

import networkx as nx
import zlib

from src.agent.news_client import NewsClient
from src.config import settings
from src.graph.build_graph import build_and_save_default_graph, load_graph, GRAPH_PATH
from src.logging_config import get_logger
from src.retrieval.chunking import Chunk, chunk_document
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore

logger = get_logger(__name__)

_CERTIFICATIONS = ["ISO 9001", "ISO 14001", "IATF 16949", "C-TPAT", "AEO"]


def generate_supplier_profile_docs(g: nx.DiGraph) -> list[dict]:
    """Deterministically generate a profile document per node from graph attributes."""
    docs = []
    for node_id, data in g.nodes(data=True):
        name = data.get("name", node_id)
        node_type = data.get("node_type", "entity")
        country = data.get("country", "unknown")
        risk = float(data.get("risk_score", 0.0))
        reliability = float(data.get("reliability", 0.0))
        capacity = float(data.get("capacity", 0.0))
        tier = data.get("tier", "n/a")

        risk_band = "low" if risk < 0.33 else "moderate" if risk < 0.66 else "elevated"
        cert_index = zlib.crc32(node_id.encode("utf-8")) % len(_CERTIFICATIONS)
        cert = _CERTIFICATIONS[cert_index]

        text = (
            f"Synthetic profile: {name} is a {node_type} located in {country}, operating at supply chain tier {tier}. "
            f"Historical reliability (on-time fulfillment rate) is {reliability:.0%}. "
            f"Current baseline disruption risk is assessed as {risk_band} ({risk:.2f} on a 0-1 scale). "
            f"Rated throughput capacity is approximately {capacity:,.0f} units. "
            f"{name} holds {cert} certification. Recent audits found no material compliance issues. "
            f"Primary exposure factors include regional geopolitical risk, weather-related logistics "
            f"disruption, and single-sourcing dependency for critical components."
        )
        docs.append({"doc_id": node_id, "source": "supplier_profile", "text": text, "metadata": {"node_id": node_id, "node_type": node_type}})
    return docs


def generate_fallback_news_docs() -> list[dict]:
    """Generate a varied synthetic news corpus when NewsAPI is unavailable."""
    templates = [
        "Port congestion at a major {region} shipping hub is delaying container vessels and disrupting downstream deliveries.",
        "Flooding in {region} has temporarily halted production at component suppliers and extended expected lead times.",
        "Unresolved labor negotiations at {region} logistics facilities could disrupt scheduled freight movements.",
        "An electronics component shortage affecting {region} is forcing manufacturers to prioritize high-volume customers.",
        "New trade restrictions affecting shipments through {region} are prompting buyers to reassess sourcing plans.",
        "Severe weather in {region} has reduced freight capacity and delayed shipments of temperature-sensitive materials.",
        "A cyber incident affecting a logistics operator in {region} has interrupted booking and cargo-tracking systems.",
        "Rail service interruptions in {region} are increasing transit times for industrial inputs and finished goods.",
        "A supplier quality recall in {region} has paused shipments while affected batches undergo inspection.",
        "Rising fuel prices in {region} are increasing carrier surcharges and pressure on contracted shipping budgets.",
    ]
    regions = [
        "East Asia", "Southern Europe", "the Gulf Coast", "Southeast Asia", "Eastern Europe",
        "North America", "the Pacific Northwest", "Central Europe", "the Indian Ocean", "Latin America",
    ]
    docs = []
    for template_index, template in enumerate(templates):
        for region_index, region in enumerate(regions):
            index = template_index * len(regions) + region_index
            text = template.format(region=region)
            docs.append({
                "doc_id": f"news_fallback_{index:03d}",
                "source": "news_fallback",
                "text": f"Synthetic scenario: {text}",
                "metadata": {"region": region, "scenario_type": template_index},
            })
    return docs


def build_and_save_vector_store(g: nx.DiGraph | None = None, news_query: str = "supply chain disruption") -> FAISSVectorStore:
    if g is None:
        try:
            g = load_graph(GRAPH_PATH)
        except FileNotFoundError:
            g = build_and_save_default_graph()

    profile_docs = generate_supplier_profile_docs(g)

    news_client = NewsClient()
    news_articles = news_client.fetch_disruption_news(query=news_query)
    if news_articles:
        news_docs = [
            {"doc_id": a["url"] or f"news_{i}", "source": "newsapi", "text": f"{a['title']}. {a['description'] or ''}", "metadata": {"published_at": a.get("published_at"), "source_name": a.get("source_name")}}
            for i, a in enumerate(news_articles)
        ]
    else:
        news_docs = generate_fallback_news_docs()

    all_docs = profile_docs + news_docs

    all_chunks: list[Chunk] = []
    for doc in all_docs:
        all_chunks.extend(
            chunk_document(doc_id=doc["doc_id"], source=doc["source"], text=doc["text"], metadata=doc.get("metadata", {}))
        )

    embedder = EmbeddingModel()
    embedder.fit([c.text for c in all_chunks])
    embeddings = embedder.encode([c.text for c in all_chunks])

    store = FAISSVectorStore(
        dim=embedder.dim,
        embedding_backend=embedder.backend,
        embedding_model=embedder.model_name if embedder.backend == "sentence-transformers" else None,
    )
    store.add(embeddings, all_chunks)
    store.save()

    logger.info(
        "Ingestion complete: %d source docs (%d supplier profiles, %d news) -> %d chunks, backend=%s",
        len(all_docs), len(profile_docs), len(news_docs), len(all_chunks), embedder.backend,
    )
    return store


if __name__ == "__main__":
    build_and_save_vector_store()
