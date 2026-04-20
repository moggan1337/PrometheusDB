"""
Hybrid Index combining Time-Series and Vector Search.

This module provides a hybrid index that combines:
1. Time-series storage with efficient range queries
2. Vector similarity search for semantic queries
3. Combined hybrid queries (time-range + semantic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from .hnsw import HNSWIndex, SearchResult
from .ivf_pq import IVFPQIndex


@dataclass
class HybridSearchResult:
    """Result from a hybrid search combining time-series and vector data."""
    id: str
    distance: float = 0.0
    score: float = 0.0
    timestamp: int | None = None
    value: float | None = None
    metric_name: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: np.ndarray | None = None


class HybridIndex:
    """
    Hybrid index combining time-series storage with vector similarity search.
    
    This index supports:
    - Time-series data with efficient range queries
    - Vector embeddings for semantic similarity
    - Hybrid queries combining time and semantic search
    - Automatic synchronization between time-series and vector indices
    
    Example:
        >>> hybrid = HybridIndex(dimension=128)
        >>> hybrid.add_ts("cpu_usage", {"host": "server1"}, timestamp=1000, value=0.75)
        >>> hybrid.add_vector("cpu_usage", {"host": "server1"}, embedding)
        >>> results = hybrid.hybrid_search(
        ...     query_vector=embedding,
        ...     time_range=(0, 2000),
        ...     k=10
        ... )
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "hnsw",
        hnsw_m: int = 16,
        hnsw_ef: int = 50,
        ivf_nlist: int = 256,
        ivf_nprobe: int = 8,
        distance: str = "cosine",
    ):
        """
        Initialize hybrid index.
        
        Args:
            dimension: Vector dimension
            index_type: Vector index type ('hnsw' or 'ivf_pq')
            hnsw_m: HNSW M parameter (if using HNSW)
            hnsw_ef: HNSW ef parameter (if using HNSW)
            ivf_nlist: IVF number of clusters (if using IVF-PQ)
            ivf_nprobe: IVF nprobe parameter (if using IVF-PQ)
            distance: Distance metric
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Create vector index
        if index_type == "hnsw":
            self.vector_index = HNSWIndex(
                dimension=dimension,
                M=hnsw_m,
                ef_search=hnsw_ef,
                distance=distance,
            )
        elif index_type == "ivf_pq":
            self.vector_index = IVFPQIndex(
                dimension=dimension,
                nlist=ivf_nlist,
                nprobe=ivf_nprobe,
                distance=distance,
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Time-series storage
        self.time_series: dict[str, list[tuple[int, float]]] = {}
        
        # Mapping from vector ID to time-series metadata
        self.id_mapping: dict[str, dict[str, Any]] = {}
        
        # Metrics registry
        self.metrics: dict[str, dict[str, Any]] = {}
    
    def add_ts(
        self,
        metric_name: str,
        labels: dict[str, str],
        timestamp: int,
        value: float,
    ) -> str:
        """
        Add a time-series data point.
        
        Args:
            metric_name: Name of the metric
            labels: Label key-value pairs
            timestamp: Timestamp in milliseconds
            value: Metric value
        
        Returns:
            Generated time-series ID
        """
        # Generate unique ID
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        ts_id = f"{metric_name}{{{label_str}}}"
        
        if ts_id not in self.time_series:
            self.time_series[ts_id] = []
        
        self.time_series[ts_id].append((timestamp, value))
        self.time_series[ts_id].sort(key=lambda x: x[0])  # Sort by timestamp
        
        # Update metrics registry
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                "name": metric_name,
                "type": "gauge",
                "labels": set(),
                "series_count": 0,
            }
        
        self.metrics[metric_name]["labels"].add(frozenset(labels.items()))
        self.metrics[metric_name]["series_count"] = len(self.time_series)
        
        return ts_id
    
    def add_vector(
        self,
        metric_name: str,
        labels: dict[str, str],
        vector: np.ndarray,
        timestamp: int | None = None,
        value: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a vector with associated time-series metadata.
        
        Args:
            metric_name: Name of the metric
            labels: Label key-value pairs
            vector: Embedding vector
            timestamp: Optional associated timestamp
            value: Optional associated value
            metadata: Additional metadata
        
        Returns:
            Generated vector ID
        """
        # Generate unique ID
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        vector_id = f"{metric_name}{{{label_str}}}"
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "metric_name": metric_name,
            "labels": labels,
            "timestamp": timestamp,
            "value": value,
        })
        
        # Add to vector index
        self.vector_index.add(vector_id, vector, metadata)
        
        # Store mapping
        self.id_mapping[vector_id] = {
            "metric_name": metric_name,
            "labels": labels,
            "timestamp": timestamp,
            "value": value,
        }
        
        return vector_id
    
    def time_range_query(
        self,
        metric_name: str | None = None,
        labels: dict[str, str] | None = None,
        start: int | None = None,
        end: int | None = None,
        limit: int = 1000,
    ) -> list[tuple[str, int, float]]:
        """
        Query time-series data within a time range.
        
        Args:
            metric_name: Optional metric name filter
            labels: Optional label filters
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            limit: Maximum results
        
        Returns:
            List of (series_id, timestamp, value) tuples
        """
        results = []
        
        for ts_id, points in self.time_series.items():
            # Filter by metric name
            if metric_name:
                if not ts_id.startswith(metric_name):
                    continue
            
            # Filter by labels
            if labels:
                # Extract labels from series ID
                label_part = ts_id.split("{")[1].rstrip("}") if "{" in ts_id else ""
                series_labels = {}
                for pair in label_part.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        series_labels[k.strip()] = v.strip()
                
                if not all(series_labels.get(k) == v for k, v in labels.items()):
                    continue
            
            # Filter by time range
            for ts, val in points:
                if start is not None and ts < start:
                    continue
                if end is not None and ts > end:
                    continue
                results.append((ts_id, ts, val))
                
                if len(results) >= limit:
                    break
        
        return results
    
    def vector_search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        metric_name: str | None = None,
        filter_func: Callable[[str], bool] | None = None,
    ) -> list[HybridSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            metric_name: Optional metric name filter
            filter_func: Optional filter function
        
        Returns:
            List of search results
        """
        # Create filter that includes metric name
        def combined_filter(vid: str) -> bool:
            if metric_name:
                if not vid.startswith(metric_name):
                    return False
            if filter_func:
                return filter_func(vid)
            return True
        
        # Search vector index
        vector_results = self.vector_index.search(
            query_vector,
            k=k,
            filter_func=combined_filter,
        )
        
        # Convert to hybrid results
        results = []
        for res in vector_results:
            mapping = self.id_mapping.get(res.id, {})
            
            results.append(HybridSearchResult(
                id=res.id,
                distance=res.distance,
                score=1.0 / (1.0 + res.distance),  # Convert distance to score
                timestamp=mapping.get("timestamp"),
                value=mapping.get("value"),
                metric_name=mapping.get("metric_name", ""),
                labels=mapping.get("labels", {}),
                metadata=res.metadata,
                vector=res.vector,
            ))
        
        return results
    
    def hybrid_search(
        self,
        query_vector: np.ndarray | None = None,
        metric_name: str | None = None,
        labels: dict[str, str] | None = None,
        time_range: tuple[int, int] | None = None,
        k: int = 10,
        time_weight: float = 0.5,
        vector_weight: float = 0.5,
        limit: int = 100,
    ) -> list[HybridSearchResult]:
        """
        Perform a hybrid search combining time-range and vector similarity.
        
        Args:
            query_vector: Query embedding for semantic search
            metric_name: Filter by metric name
            labels: Filter by labels
            time_range: (start, end) tuple for time filtering
            k: Number of results
            time_weight: Weight for time relevance (0-1)
            vector_weight: Weight for vector similarity (0-1)
            limit: Maximum results
        
        Returns:
            Combined and ranked results
        """
        candidates: dict[str, HybridSearchResult] = {}
        
        # Step 1: Get candidates from time range
        if time_range:
            start, end = time_range
            ts_results = self.time_range_query(
                metric_name=metric_name,
                labels=labels,
                start=start,
                end=end,
                limit=limit * 2,
            )
            
            for ts_id, ts, val in ts_results:
                # Time relevance score (closer to end = more relevant)
                time_score = 1.0
                if time_range:
                    start, end = time_range
                    range_duration = end - start if end > start else 1
                    position = (ts - start) / range_duration
                    time_score = 1.0 - abs(0.5 - position)  # Peak in middle
                
                if ts_id not in candidates:
                    candidates[ts_id] = HybridSearchResult(
                        id=ts_id,
                        timestamp=ts,
                        value=val,
                        score=time_score * time_weight,
                        metric_name=metric_name or ts_id.split("{")[0],
                    )
                else:
                    # Update with latest value if newer
                    if ts > (candidates[ts_id].timestamp or 0):
                        candidates[ts_id].timestamp = ts
                        candidates[ts_id].value = val
        
        # Step 2: Get candidates from vector search
        if query_vector is not None:
            vector_results = self.vector_search(
                query_vector,
                k=limit * 2,
                metric_name=metric_name,
            )
            
            for res in vector_results:
                if res.id not in candidates:
                    candidates[res.id] = res
                    candidates[res.id].score = res.score * vector_weight
                else:
                    # Combine scores
                    candidates[res.id].score += res.score * vector_weight
                    candidates[res.id].distance = res.distance
        
        # Step 3: Sort by combined score
        sorted_results = sorted(
            candidates.values(),
            key=lambda x: x.score,
            reverse=True,
        )
        
        return sorted_results[:limit]
    
    def aggregate(
        self,
        metric_name: str,
        labels: dict[str, str] | None = None,
        start: int | None = None,
        end: int | None = None,
        agg_func: str = "avg",
    ) -> float | None:
        """
        Aggregate values over a time range.
        
        Args:
            metric_name: Metric to aggregate
            labels: Optional label filters
            start: Start timestamp
            end: End timestamp
            agg_func: Aggregation function ('avg', 'sum', 'min', 'max', 'count')
        
        Returns:
            Aggregated value or None
        """
        results = self.time_range_query(
            metric_name=metric_name,
            labels=labels,
            start=start,
            end=end,
            limit=100000,
        )
        
        if not results:
            return None
        
        values = [val for _, _, val in results]
        
        if agg_func == "avg":
            return sum(values) / len(values)
        elif agg_func == "sum":
            return sum(values)
        elif agg_func == "min":
            return min(values)
        elif agg_func == "max":
            return max(values)
        elif agg_func == "count":
            return len(values)
        else:
            return values[-1] if values else None
    
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        stats = {
            "num_series": len(self.time_series),
            "num_vectors": len(self.id_mapping),
            "dimension": self.dimension,
            "index_type": self.index_type,
        }
        
        if hasattr(self.vector_index, "get_stats"):
            stats["vector_index"] = self.vector_index.get_stats()
        
        return stats
    
    def save(self, filepath: str) -> None:
        """Save index to file."""
        import pickle
        
        data = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "time_series": self.time_series,
            "id_mapping": self.id_mapping,
            "metrics": self.metrics,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, vector_index: HNSWIndex | IVFPQIndex | None = None) -> HybridIndex:
        """Load index from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct vector index if provided
        if vector_index is not None:
            index = cls(
                dimension=data["dimension"],
                index_type=data["index_type"],
            )
            index.vector_index = vector_index
        else:
            index = cls(dimension=data["dimension"], index_type=data["index_type"])
        
        index.time_series = data["time_series"]
        index.id_mapping = data["id_mapping"]
        index.metrics = data["metrics"]
        
        return index
