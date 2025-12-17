"""
Causal ML: estimate the causal effect of node-level disruption events on
downstream supply chain impact, using DoWhy — going beyond correlation by
explicitly modeling confounders (node risk profile, centrality, tier) that
would otherwise bias a naive correlational estimate.

Pipeline (DoWhy's four steps):
  1. Model  — declare a causal graph: confounders -> treatment, confounders
              -> outcome, treatment -> outcome.
  2. Identify — derive the backdoor adjustment formula from the graph.
  3. Estimate — fit a linear-regression backdoor estimator for the Average
              Treatment Effect (ATE) of "node disrupted" on "downstream
              capacity loss".
  4. Refute  — sanity-check the estimate with a placebo-treatment test and a
              random-common-cause test; an estimate that collapses under
              either is not trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import CausalModel

from src.graph.graph_utils import compute_centrality_scores, simulate_disruption_propagation
from src.logging_config import get_logger

logger = get_logger(__name__)

TREATMENT = "disrupted"
OUTCOME = "downstream_impact"
CONFOUNDERS = ["risk_score", "reliability", "capacity_log", "tier", "betweenness"]

CAUSAL_GRAPH_GML = """
graph [
  directed 1
  node [ id 0 label "disrupted" ]
  node [ id 1 label "downstream_impact" ]
  node [ id 2 label "risk_score" ]
  node [ id 3 label "reliability" ]
  node [ id 4 label "capacity_log" ]
  node [ id 5 label "tier" ]
  node [ id 6 label "betweenness" ]
  edge [ source 0 target 1 ]
  edge [ source 2 target 0 ]
  edge [ source 2 target 1 ]
  edge [ source 3 target 0 ]
  edge [ source 3 target 1 ]
  edge [ source 4 target 0 ]
  edge [ source 4 target 1 ]
  edge [ source 5 target 0 ]
  edge [ source 5 target 1 ]
  edge [ source 6 target 0 ]
  edge [ source 6 target 1 ]
]
"""


def build_causal_dataset(g: nx.DiGraph, seed: int = 42, n_trials_per_node: int = 8) -> pd.DataFrame:
    """Build an observational dataset for causal estimation.

    For each node, simulate `n_trials_per_node` independent "trials": a
    Bernoulli disruption event whose probability depends on the node's risk
    score (confounding by design, mirroring how real disruptions are more
    likely at already-risky nodes), then measure downstream impact as the
    BFS-cascade capacity loss (with noise) when disrupted, or a small
    baseline noise level when not disrupted.
    """
    rng = np.random.default_rng(seed)
    centrality = compute_centrality_scores(g)
    total_capacity = sum(float(d.get("capacity", 0.0)) for _, d in g.nodes(data=True)) or 1.0

    rows = []
    for node, data in g.nodes(data=True):
        risk = float(data.get("risk_score", 0.0))
        reliability = float(data.get("reliability", 0.0))
        capacity_log = float(np.log1p(float(data.get("capacity", 0.0))))
        tier = float(data.get("tier", 0))
        betweenness = centrality[node]["betweenness"]

        # Precompute the "true" cascade impact once per node (expensive BFS).
        sim = simulate_disruption_propagation(g, node)
        true_impact = sim.total_capacity_lost / total_capacity

        for _ in range(n_trials_per_node):
            # Confounded treatment assignment: riskier / less reliable nodes
            # are disrupted more often (this is exactly the confounding a
            # naive correlation would fail to adjust for).
            p_disrupt = float(np.clip(0.15 + 0.5 * risk - 0.2 * reliability, 0.02, 0.9))
            disrupted = int(rng.random() < p_disrupt)

            if disrupted:
                outcome = true_impact + rng.normal(0, 0.03)
            else:
                # Small baseline "background noise" impact even without a
                # disruption at this node (other parts of the network).
                outcome = 0.01 * risk + rng.normal(0, 0.01)

            rows.append({
                "node_id": node,
                TREATMENT: disrupted,
                OUTCOME: max(0.0, outcome),
                "risk_score": risk,
                "reliability": reliability,
                "capacity_log": capacity_log,
                "tier": tier,
                "betweenness": betweenness,
            })

    df = pd.DataFrame(rows)
    logger.info("Built causal dataset: %d rows (%d nodes x %d trials)", len(df), g.number_of_nodes(), n_trials_per_node)
    return df


@dataclass
class CausalEstimationResult:
    ate: float
    confidence_interval: tuple[float, float] | None
    placebo_refutation_passed: bool
    random_common_cause_refutation_passed: bool
    summary: str


def estimate_disruption_causal_effect(df: pd.DataFrame) -> CausalEstimationResult:
    """Run the full DoWhy model -> identify -> estimate -> refute pipeline."""
    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        common_causes=CONFOUNDERS,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    logger.info("Identified estimand:\n%s", identified_estimand)

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        confidence_intervals=True,
    )
    ate = float(estimate.value)
    ci = None
    try:
        ci_arr = estimate.get_confidence_intervals()
        ci = (float(ci_arr[0][0]), float(ci_arr[0][1]))
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not compute confidence interval: %s", exc)

    logger.info("Estimated ATE of disruption on downstream_impact: %.4f (95%% CI: %s)", ate, ci)

    # --- Refutation 1: placebo treatment (effect should collapse to ~0) -----
    placebo_passed = False
    try:
        placebo = model.refute_estimate(
            identified_estimand, estimate, method_name="placebo_treatment_refuter",
            placebo_type="permute", num_simulations=20,
        )
        placebo_passed = abs(placebo.new_effect) < abs(ate) * 0.5 + 1e-3
        logger.info("Placebo refutation: new_effect=%.4f (passed=%s)", placebo.new_effect, placebo_passed)
    except Exception as exc:
        logger.warning("Placebo refutation failed to run: %s", exc)

    # --- Refutation 2: add a random common cause (effect should be stable) ---
    random_cc_passed = False
    try:
        random_cc = model.refute_estimate(
            identified_estimand, estimate, method_name="random_common_cause",
        )
        random_cc_passed = abs(random_cc.new_effect - ate) < abs(ate) * 0.3 + 1e-3
        logger.info("Random common cause refutation: new_effect=%.4f (passed=%s)", random_cc.new_effect, random_cc_passed)
    except Exception as exc:
        logger.warning("Random common cause refutation failed to run: %s", exc)

    summary = (
        f"Disrupting a node causes an estimated +{ate:.3f} fraction-of-network "
        f"capacity loss downstream (on average), after adjusting for risk score, "
        f"reliability, capacity, tier, and betweenness centrality as confounders."
    )

    return CausalEstimationResult(
        ate=ate,
        confidence_interval=ci,
        placebo_refutation_passed=placebo_passed,
        random_common_cause_refutation_passed=random_cc_passed,
        summary=summary,
    )


def run_causal_analysis(g: nx.DiGraph) -> CausalEstimationResult:
    df = build_causal_dataset(g)
    return estimate_disruption_causal_effect(df)


def get_or_compute_causal_ate(g: nx.DiGraph, cache_path=None, force_recompute: bool = False) -> CausalEstimationResult:
    """Cache the (relatively expensive) DoWhy pipeline to disk so callers like
    the alert agent don't re-run identify/estimate/refute on every alert."""
    import json as _json

    from src.config import settings as _settings

    cache_path = cache_path or (_settings.model_dir / "causal_ate_cache.json")

    if cache_path.exists() and not force_recompute:
        try:
            with open(cache_path) as f:
                cached = _json.load(f)
            return CausalEstimationResult(
                ate=cached["ate"],
                confidence_interval=tuple(cached["confidence_interval"]) if cached.get("confidence_interval") else None,
                placebo_refutation_passed=cached["placebo_refutation_passed"],
                random_common_cause_refutation_passed=cached["random_common_cause_refutation_passed"],
                summary=cached["summary"],
            )
        except Exception as exc:
            logger.warning("Failed to load cached causal result (%s); recomputing", exc)

    result = run_causal_analysis(g)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump({
            "ate": float(result.ate),
            "confidence_interval": [float(x) for x in result.confidence_interval] if result.confidence_interval else None,
            "placebo_refutation_passed": bool(result.placebo_refutation_passed),
            "random_common_cause_refutation_passed": bool(result.random_common_cause_refutation_passed),
            "summary": result.summary,
        }, f)
    return result


if __name__ == "__main__":
    from src.graph.build_graph import build_and_save_default_graph, load_graph, GRAPH_PATH

    try:
        graph = load_graph(GRAPH_PATH)
    except FileNotFoundError:
        graph = build_and_save_default_graph()

    result = run_causal_analysis(graph)
    logger.info(result.summary)
