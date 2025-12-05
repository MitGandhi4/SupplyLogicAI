"""
Synthetic supply chain network generator.

Real-world supplier network data with ground-truth failure-propagation
labels is not publicly available, so SupplyMind generates a structurally
realistic synthetic network (tiered suppliers -> ports/warehouses ->
manufacturers) with plausible attribute distributions. This mirrors the
approach recommended for demonstrating the GNN when authoritative Kaggle /
FRED data does not provide a labeled graph.

The generator is deterministic given `settings.random_seed`, so re-running
it reproduces the same network.
"""
from __future__ import annotations

import random

import numpy as np

from src.config import settings
from src.graph.schema import Edge, EdgeType, Node, NodeType, SupplyChainGraphData
from src.logging_config import get_logger

logger = get_logger(__name__)

_COUNTRIES = [
    ("China", "APAC"), ("Vietnam", "APAC"), ("India", "APAC"), ("Taiwan", "APAC"),
    ("South Korea", "APAC"), ("Germany", "EU"), ("Poland", "EU"), ("Netherlands", "EU"),
    ("Mexico", "AMER"), ("USA", "AMER"), ("Brazil", "AMER"), ("Turkey", "EMEA"),
]

_PORT_NAMES = ["Shanghai", "Rotterdam", "Los Angeles", "Singapore", "Busan", "Hamburg", "Santos", "Jebel Ali"]


def _rand_coords(rng: random.Random) -> tuple[float, float]:
    return round(rng.uniform(-60, 70), 4), round(rng.uniform(-170, 170), 4)


def generate_synthetic_graph_data(
    n_suppliers: int | None = None,
    n_warehouses: int | None = None,
    n_ports: int | None = None,
    n_manufacturers: int | None = None,
    n_retailers: int | None = None,
    seed: int | None = None,
) -> SupplyChainGraphData:
    """Generate a validated, tiered synthetic supply chain network.

    Structure: tier-2/tier-3 suppliers -> tier-1 suppliers -> ports/warehouses
    -> manufacturers -> retailers. Edge weights/volumes are drawn from distributions
    calibrated so that a handful of nodes end up as structural bottlenecks
    (useful for demonstrating critical-node detection).
    """
    n_suppliers = n_suppliers or settings.n_suppliers
    n_warehouses = n_warehouses or settings.n_warehouses
    n_ports = n_ports or settings.n_ports
    n_manufacturers = n_manufacturers or settings.n_manufacturers
    n_retailers = n_retailers or settings.n_retailers
    seed = seed if seed is not None else settings.random_seed
    if n_retailers and not n_manufacturers:
        raise ValueError("At least one manufacturer is required when generating retailers")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    nodes: list[Node] = []
    edges: list[Edge] = []

    # --- Suppliers (mix of tiers 1-3) -----------------------------------
    supplier_ids = []
    for i in range(n_suppliers):
        country, region = rng.choice(_COUNTRIES)
        lat, lon = _rand_coords(rng)
        tier = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        node_id = f"SUP_{i:03d}"
        supplier_ids.append((node_id, tier))
        nodes.append(
            Node(
                node_id=node_id,
                node_type=NodeType.SUPPLIER,
                name=f"Supplier {i} ({country})",
                country=country,
                region=region,
                latitude=lat,
                longitude=lon,
                risk_score=float(np.clip(np_rng.beta(2, 5), 0, 1)),
                capacity=float(np_rng.lognormal(mean=8, sigma=0.7)),
                reliability=float(np.clip(np_rng.beta(8, 2), 0, 1)),
                tier=tier,
            )
        )

    # --- Ports -------------------------------------------------------------
    port_ids = []
    for i in range(n_ports):
        name = _PORT_NAMES[i % len(_PORT_NAMES)]
        country, region = rng.choice(_COUNTRIES)
        lat, lon = _rand_coords(rng)
        node_id = f"PORT_{i:03d}"
        port_ids.append(node_id)
        nodes.append(
            Node(
                node_id=node_id,
                node_type=NodeType.PORT,
                name=f"{name} Port",
                country=country,
                region=region,
                latitude=lat,
                longitude=lon,
                risk_score=float(np.clip(np_rng.beta(2, 6), 0, 1)),
                capacity=float(np_rng.lognormal(mean=10, sigma=0.5)),
                reliability=float(np.clip(np_rng.beta(9, 2), 0, 1)),
                tier=1,
            )
        )

    # --- Warehouses -----------------------------------------------------------
    warehouse_ids = []
    for i in range(n_warehouses):
        country, region = rng.choice(_COUNTRIES)
        lat, lon = _rand_coords(rng)
        node_id = f"WH_{i:03d}"
        warehouse_ids.append(node_id)
        nodes.append(
            Node(
                node_id=node_id,
                node_type=NodeType.WAREHOUSE,
                name=f"Warehouse {i} ({country})",
                country=country,
                region=region,
                latitude=lat,
                longitude=lon,
                risk_score=float(np.clip(np_rng.beta(2, 6), 0, 1)),
                capacity=float(np_rng.lognormal(mean=9, sigma=0.5)),
                reliability=float(np.clip(np_rng.beta(8, 2), 0, 1)),
                tier=1,
            )
        )

    # --- Manufacturers (sinks) ---------------------------------------------------
    manufacturer_ids = []
    for i in range(n_manufacturers):
        country, region = rng.choice(_COUNTRIES)
        lat, lon = _rand_coords(rng)
        node_id = f"MFG_{i:03d}"
        manufacturer_ids.append(node_id)
        nodes.append(
            Node(
                node_id=node_id,
                node_type=NodeType.MANUFACTURER,
                name=f"Manufacturer {i} ({country})",
                country=country,
                region=region,
                latitude=lat,
                longitude=lon,
                risk_score=float(np.clip(np_rng.beta(2, 8), 0, 1)),
                capacity=float(np_rng.lognormal(mean=10, sigma=0.4)),
                reliability=float(np.clip(np_rng.beta(9, 2), 0, 1)),
                tier=0,
            )
        )

    retailer_ids = []
    for i in range(n_retailers):
        country, region = rng.choice(_COUNTRIES)
        lat, lon = _rand_coords(rng)
        node_id = f"RTL_{i:03d}"
        retailer_ids.append(node_id)
        nodes.append(
            Node(
                node_id=node_id,
                node_type=NodeType.RETAILER,
                name=f"Retailer {i} ({country})",
                country=country,
                region=region,
                latitude=lat,
                longitude=lon,
                risk_score=float(np.clip(np_rng.beta(2, 8), 0, 1)),
                capacity=float(np_rng.lognormal(mean=9, sigma=0.6)),
                reliability=float(np.clip(np_rng.beta(8, 2), 0, 1)),
                tier=0,
            )
        )

    # --- Edges: lower-tier suppliers depend on higher-tier suppliers --------------
    tier1 = [nid for nid, t in supplier_ids if t == 1]
    tier2 = [nid for nid, t in supplier_ids if t == 2]
    tier3 = [nid for nid, t in supplier_ids if t == 3]

    def _connect(sources: list[str], targets: list[str], edge_type: EdgeType, k_range=(1, 3)):
        for s in sources:
            k = rng.randint(*k_range)
            chosen = rng.sample(targets, min(k, len(targets))) if targets else []
            for t in chosen:
                strength = float(np.clip(np_rng.beta(2, 2), 0.05, 1.0))
                transit_days = float(np_rng.gamma(shape=3, scale=4))
                edges.append(
                    Edge(
                        source=s,
                        target=t,
                        edge_type=edge_type,
                        weight=strength,
                        lead_time_days=transit_days,
                        volume=float(np_rng.lognormal(mean=6, sigma=0.8)),
                        shipping_cost=float(np_rng.lognormal(mean=5, sigma=0.7)),
                        transit_days=transit_days,
                        dependency_strength=strength,
                    )
                )

    if tier1:
        _connect(tier3, tier1, EdgeType.DEPENDS_ON)
        _connect(tier2, tier1, EdgeType.DEPENDS_ON)
        # Tier-1 suppliers ship to ports/warehouses.
        _connect(tier1, port_ids + warehouse_ids, EdgeType.SHIPS_TO, k_range=(1, 2))
    else:
        _connect(tier2 + tier3, port_ids + warehouse_ids, EdgeType.SHIPS_TO, k_range=(1, 2))

    # Ports feed warehouses, warehouses feed manufacturers, ports can feed
    # manufacturers directly too.
    _connect(port_ids, warehouse_ids, EdgeType.SHIPS_TO, k_range=(1, 2))
    _connect(warehouse_ids, manufacturer_ids, EdgeType.SUPPLIES, k_range=(1, 3))
    _connect(port_ids, manufacturer_ids, EdgeType.SUPPLIES, k_range=(0, 1))
    _connect(manufacturer_ids, retailer_ids, EdgeType.SUPPLIES, k_range=(1, 3))
    supplied_retailers = {edge.target for edge in edges if edge.target in retailer_ids}
    for retailer_id in retailer_ids:
        if retailer_id not in supplied_retailers:
            manufacturer_id = rng.choice(manufacturer_ids)
            _connect([manufacturer_id], [retailer_id], EdgeType.SUPPLIES, k_range=(1, 1))

    graph_data = SupplyChainGraphData(nodes=nodes, edges=edges)
    logger.info(
        "Generated synthetic graph: %d nodes (%d suppliers, %d ports, %d warehouses, %d manufacturers, %d retailers), %d edges",
        len(nodes), len(supplier_ids), len(port_ids), len(warehouse_ids), len(manufacturer_ids), len(retailer_ids), len(edges),
    )
    return graph_data


if __name__ == "__main__":
    data = generate_synthetic_graph_data()
    logger.info("Sample node: %s", data.nodes[0].model_dump())
    logger.info("Sample edge: %s", data.edges[0].model_dump())
