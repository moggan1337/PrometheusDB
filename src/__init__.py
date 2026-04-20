"""
PrometheusDB - High-Performance Time-Series Database with Vector Operations

A next-generation time-series database combining:
- Time-series storage with configurable retention policies
- Vector similarity search using HNSW and IVF-PQ algorithms
- Hybrid queries (time-range + semantic search)
- PromQL-compatible query language
- Gorilla compression for time-series data
- High-cardinality metric handling
- Distributed clustering with consistent hashing
"""

__version__ = "0.1.0"
__author__ = "Moggan"
__license__ = "MIT"

from .storage.database import PrometheusDB
from .storage.schema import Metric, TimeSeries, Vector
from .query.engine import QueryEngine, QueryResult
from .cluster.node import ClusterNode

__all__ = [
    "PrometheusDB",
    "Metric",
    "TimeSeries",
    "Vector",
    "QueryEngine",
    "QueryResult",
    "ClusterNode",
]
