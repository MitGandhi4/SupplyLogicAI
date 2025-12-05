"""
Pydantic schemas for supply chain graph ingestion.

Every node and edge that enters the graph is validated against these models
before it is allowed into the NetworkX graph. This prevents malformed rows
(missing IDs, out-of-range risk scores, negative capacities, dangling edge
references, etc.) from silently corrupting downstream GNN / causal / RAG
pipelines.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class NodeType(str, Enum):
    SUPPLIER = "supplier"
    WAREHOUSE = "warehouse"
    PORT = "port"
    MANUFACTURER = "manufacturer"
    RETAILER = "retailer"


class EdgeType(str, Enum):
    SHIPS_TO = "ships_to"
    DEPENDS_ON = "depends_on"
    SUPPLIES = "supplies"


class Node(BaseModel):
    """A single entity in the supply chain network."""

    node_id: str = Field(..., min_length=1, description="Unique node identifier")
    node_type: NodeType
    name: str = Field(..., min_length=1)
    country: str = Field(..., min_length=2)
    region: str = Field(default="unknown")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

    # Risk / capacity attributes used by GNN + causal model.
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Baseline disruption risk, 0=safe, 1=critical")
    capacity: float = Field(..., gt=0.0, description="Throughput capacity, arbitrary units")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Historical on-time / fulfillment rate")
    tier: int = Field(default=1, ge=0, le=5, description="Supply chain tier depth from manufacturer (0=manufacturer)")

    model_config = {"use_enum_values": True}


class Edge(BaseModel):
    """A directed relationship between two nodes (shipping lane or dependency)."""

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    edge_type: EdgeType
    weight: float = Field(..., ge=0.0, le=1.0, description="Dependency strength / criticality, 0-1")
    lead_time_days: float = Field(..., ge=0.0)
    volume: float = Field(..., ge=0.0, description="Shipped volume, arbitrary units")
    shipping_cost: float = Field(default=0.0, ge=0.0)
    transit_days: float | None = Field(default=None, ge=0.0)
    dependency_strength: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def no_self_loops(self) -> "Edge":
        if self.source == self.target:
            raise ValueError(f"Self-loop edge not allowed: {self.source} -> {self.target}")
        if self.transit_days is None:
            self.transit_days = self.lead_time_days
        if self.dependency_strength is None:
            self.dependency_strength = self.weight
        return self


class SupplyChainGraphData(BaseModel):
    """A fully validated collection of nodes and edges ready for graph construction."""

    nodes: list[Node]
    edges: list[Edge]

    @field_validator("nodes")
    @classmethod
    def unique_node_ids(cls, nodes: list[Node]) -> list[Node]:
        ids = [n.node_id for n in nodes]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"Duplicate node_id(s) found: {dupes}")
        return nodes

    @model_validator(mode="after")
    def edges_reference_existing_nodes(self) -> "SupplyChainGraphData":
        node_ids = {n.node_id for n in self.nodes}
        for e in self.edges:
            if e.source not in node_ids:
                raise ValueError(f"Edge source '{e.source}' does not reference a known node")
            if e.target not in node_ids:
                raise ValueError(f"Edge target '{e.target}' does not reference a known node")
        return self
