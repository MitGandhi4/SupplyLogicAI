"""
SupplyMind Streamlit dashboard.

Talks exclusively to the FastAPI backend (never touches model/graph files
directly), so it can run in its own container against a remote API.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import logging
import os

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("supplymind.dashboard")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30

st.set_page_config(page_title="SupplyMind", layout="wide", page_icon="🌐")

NODE_COLORS = {
    "supplier": "#4C78A8",
    "warehouse": "#F58518",
    "port": "#54A24B",
    "manufacturer": "#E45756",
    "retailer": "#B279A2",
}


def api_get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        logger.error("GET %s failed: %s", path, exc)
        st.error(f"API request failed ({path}): {exc}")
        return None


def api_post(path: str, json_body: dict | None = None) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body or {}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        logger.error("POST %s failed: %s", path, exc)
        st.error(f"API request failed ({path}): {exc}")
        return None


st.sidebar.title("🌐 SupplyMind")
st.sidebar.caption("Supply chain intelligence & resilience platform")
st.sidebar.text_input("API base URL", value=API_BASE_URL, key="api_base_url_display", disabled=True)
if st.sidebar.button("Check API health"):
    health = api_get("/health")
    if health:
        st.sidebar.success(f"API healthy: {health}")

page = st.sidebar.radio("View", ["Network Graph", "Demand Forecast", "Disruption Alerts", "Agent Reasoning Log"])

# --------------------------------------------------------------------------
if page == "Network Graph":
    st.title("Supply Chain Network")
    st.caption(
        "Explore suppliers, warehouses, ports, manufacturers, and retailers. "
        "Node outlines show entity type, fill color shows structural criticality, and size shows baseline risk."
    )

    graph_data = api_get("/graph")
    if graph_data:
        g = nx.DiGraph()
        for n in graph_data["nodes"]:
            g.add_node(n["node_id"], **n)
        for e in graph_data["edges"]:
            g.add_edge(e["source"], e["target"], **e)

        col1, col2 = st.columns([3, 1])

        critical = api_get("/graph/critical-nodes", params={"top_k": g.number_of_nodes()})
        risk_scores = {entry["node_id"]: entry["composite_score"] for entry in (critical or [])}
        node_legend = [
            {"label": label.title(), "color": color}
            for label, color in NODE_COLORS.items()
        ]

        with col2:
            st.metric("Nodes", g.number_of_nodes())
            st.metric("Edges", g.number_of_edges())
            top_k = st.slider("Top critical nodes", min_value=3, max_value=25, value=10)
            if critical:
                st.subheader("Most Critical Nodes")
                st.dataframe(pd.DataFrame(critical[:top_k]), hide_index=True, use_container_width=True)
            st.caption("Node types")
            st.markdown("  \n".join(
                f"<span style='color:{item['color']}'>●</span> {item['label']}"
                for item in node_legend
            ), unsafe_allow_html=True)

        with col1:
            pos = nx.spring_layout(g, seed=42, k=0.6)
            edge_x, edge_y = [], []
            for u, v in g.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.65, color="rgba(148, 163, 184, 0.32)"),
                hoverinfo="none",
                mode="lines",
            )

            node_x, node_y, node_color, node_text, node_size, node_line = [], [], [], [], [], []
            for n, d in g.nodes(data=True):
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                node_color.append(risk_scores.get(n, 0.0))
                node_size.append(10 + 20 * float(d.get("risk_score", 0.0)))
                node_line.append(NODE_COLORS.get(d.get("node_type"), "#888"))
                node_text.append(
                    f"{d.get('name', n)}<br>type={d.get('node_type')}<br>country={d.get('country')}"
                    f"<br>baseline risk={float(d.get('risk_score', 0.0)):.2f}"
                    f"<br>criticality={risk_scores.get(n, 0.0):.2f}"
                    f"<br>reliability={float(d.get('reliability', 0.0)):.2f}"
                )

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
                marker=dict(
                    color=node_color,
                    colorscale="YlOrRd",
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Risk"),
                    size=node_size,
                    line=dict(width=1.5, color=node_line),
                ),
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=0, r=20, t=0, b=0),
                height=650,
                               xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Disruption Propagation Simulation")
        node_options = sorted(g.nodes())
        selected_node = st.selectbox("Simulate disruption at node:", node_options)
        if st.button("Run simulation"):
            sim = api_get(f"/graph/disruption-simulation/{selected_node}")
            if sim:
                st.write(f"**{sim['origin_node']}** disruption propagates to **{len(sim['affected_nodes'])}** nodes "
                         f"over **{sim['n_hops']}** hops, losing an estimated **{sim['total_capacity_lost']:,.0f}** units of capacity.")
                st.write(sim["affected_nodes"])

# --------------------------------------------------------------------------
elif page == "Demand Forecast":
    st.title("Demand Forecasting")
    st.caption("Prophet forecast fit on synthetic historical order data (LSTM training via `python -m src.forecasting.train`, tracked in MLflow).")

    entity_id = st.text_input("Manufacturer entity ID", value="MFG_000")
    periods = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30)

    if st.button("Generate forecast"):
        with st.spinner("Fitting Prophet model..."):
            forecast = api_get(f"/forecast/{entity_id}", params={"periods": periods})
        if forecast:
            hist_df = pd.DataFrame(forecast["history"])
            fcst_df = pd.DataFrame(forecast["forecast"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df["ds"], y=hist_df["y"], mode="lines", name="Historical demand"))
            fig.add_trace(go.Scatter(x=fcst_df["ds"], y=fcst_df["yhat"], mode="lines", name="Forecast", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(
                x=list(fcst_df["ds"]) + list(fcst_df["ds"][::-1]),
                y=list(fcst_df["yhat_upper"]) + list(fcst_df["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(color="rgba(255,255,255,0)"),
                name="Confidence interval", showlegend=True,
            ))
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
elif page == "Disruption Alerts":
    st.title("Live Disruption Alerts")
    st.caption("Triggers the LangGraph agent: monitors news, retrieves RAG context, reasons over GNN + causal outputs, issues a structured alert.")

    news_query = st.text_input("News monitoring query", value="supply chain disruption")
    if st.button("Run agent now", type="primary"):
        with st.spinner("Agent reasoning..."):
            alert = api_post("/alerts/generate", {"news_query": news_query})
        if alert:
            severity_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(alert["severity"], "⚪")
            st.subheader(f"{severity_color} {alert['severity'].upper()} — {alert['origin_node']}")
            st.write(alert["summary"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Critical node probability", f"{alert['critical_node_probability']:.2f}")
            col2.metric("Predicted capacity loss", f"{alert['predicted_capacity_loss_fraction']:.2%}")
            col3.metric("Causal ATE", f"{alert['causal_ate_estimate']:.3f}" if alert.get("causal_ate_estimate") is not None else "n/a")

            st.write("**Affected downstream nodes:**", ", ".join(alert["affected_nodes"]) or "none")
            st.write("**Recommended actions:**")
            for action in alert["recommended_actions"]:
                st.markdown(f"- {action}")

            with st.expander("Retrieved evidence (RAG context)"):
                for ev in alert["evidence"]:
                    st.markdown(f"**[{ev['source']}]** (score={ev['score']:.2f}) — {ev['text']}")

            st.caption(f"LLM backend used: `{alert['llm_backend']}` · Alert ID: `{alert['alert_id']}`")

# --------------------------------------------------------------------------
elif page == "Agent Reasoning Log":
    st.title("Agent Reasoning Log")
    st.caption("History of all previously generated disruption alerts.")

    limit = st.slider("Number of alerts to show", min_value=5, max_value=100, value=20)
    history = api_get("/alerts/history", params={"limit": limit})
    if history:
        for alert in history:
            severity_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(alert["severity"], "⚪")
            with st.expander(f"{severity_color} {alert['generated_at']} — {alert['origin_node']} ({alert['severity']})"):
                st.write(alert["summary"])
                st.json(alert)
    else:
        st.info("No alerts generated yet. Go to 'Disruption Alerts' and click 'Run agent now'.")
