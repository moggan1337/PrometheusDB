"""
PrometheusDB - Main Database Implementation.

This module provides the main PrometheusDB class that coordinates
all storage, indexing, and query components.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np

from .schema import (
    Metric, TimeSeries, DataPoint, Label, Vector,
    ValueType, RetentionPolicy, DEFAULT_RETENTION
)
from .wal import WriteAheadLog
from .retention import RetentionManager
from ..compression import GorillaCompressor, DeltaOfDeltaCompressor
from ..indexing import HybridIndex, ANNRegistry
from ..query import QueryEngine, QueryResult


@dataclass
class DatabaseConfig:
    """Configuration for PrometheusDB."""
    data_dir: str = "./data"
    wal_enabled: bool = True
    wal_dir: str = "./data/wal"
    retention_enabled: bool = True
    retention_check_interval: int = 3600  # seconds
    compression_enabled: bool = True
    max_memory_mb: int = 1024
    vector_dimension: int = 128
    vector_index_type: str = "auto"
    high_cardinality_threshold: int = 100000
    chunk_size: int = 1000  # Points per chunk


class PrometheusDB:
    """
    High-performance Time-Series Database with Vector Operations.
    
    Features:
    - Time-series storage with configurable retention
    - Vector similarity search (HNSW, IVF-PQ)
    - Hybrid queries (time-range + semantic)
    - PromQL-compatible query language
    - Gorilla compression for time-series
    - High-cardinality handling
    - Distributed clustering support
    
    Example:
        >>> db = PrometheusDB(data_dir="./prometheusdb_data")
        >>> db.write("cpu_usage", {"host": "server1"}, value=0.75)
        >>> results = db.query('cpu_usage{host="server1"}[5m]')
        >>> vectors = db.search_vectors(query_vector, k=10)
    """
    
    def __init__(self, config: DatabaseConfig | None = None):
        """
        Initialize PrometheusDB.
        
        Args:
            config: Database configuration
        """
        self.config = config or DatabaseConfig()
        self._lock = threading.RLock()
        
        # Initialize directories
        self._data_dir = Path(self.config.data_dir)
        self._wal_dir = Path(self.config.wal_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._wal_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._storage: dict[str, TimeSeries] = {}
        self._metrics: dict[int, Metric] = {}  # fingerprint -> Metric
        self._metric_index: dict[str, int] = {}  # metric_key -> fingerprint
        
        # Initialize compression
        self._gorilla = GorillaCompressor()
        self._delta = DeltaOfDeltaCompressor()
        
        # Initialize vector index
        self._vector_index = HybridIndex(
            dimension=self.config.vector_dimension,
            index_type=self.config.vector_index_type if self.config.vector_index_type != "auto" else "hnsw"
        )
        
        # Initialize query engine
        self._query_engine = QueryEngine(self)
        
        # Initialize WAL
        self._wal: WriteAheadLog | None = None
        if self.config.wal_enabled:
            self._wal = WriteAheadLog(str(self._wal_dir))
        
        # Initialize retention manager
        self._retention_manager = RetentionManager(
            retention_enabled=self.config.retention_enabled,
            check_interval=self.config.retention_check_interval
        )
        
        # Statistics
        self._stats = {
            "writes": 0,
            "reads": 0,
            "queries": 0,
            "start_time": int(time.time() * 1000),
        }
        
        # High-cardinality tracking
        self._high_cardinality_metrics: set[str] = set()
    
    def write(
        self,
        metric_name: str,
        labels: dict[str, str],
        value: float,
        timestamp: int | None = None,
        value_type: ValueType = ValueType.GAUGE,
    ) -> str:
        """
        Write a single data point.
        
        Args:
            metric_name: Name of the metric
            labels: Label key-value pairs
            value: Metric value
            timestamp: Unix timestamp in milliseconds (default: now)
            value_type: Type of metric value
        
        Returns:
            Time series key
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        # Create metric
        label_objects = frozenset(Label(name=k, value=v) for k, v in labels.items())
        metric = Metric(
            name=metric_name,
            labels=label_objects,
            value_type=value_type,
        )
        
        # Create data point
        point = DataPoint(timestamp=timestamp, value=value)
        
        with self._lock:
            # Store metric
            fp = metric.fingerprint
            self._metrics[fp] = metric
            self._metric_index[metric.metric_key] = fp
            
            # Get or create time series
            ts = self._storage.get(metric.metric_key)
            if ts is None:
                ts = TimeSeries(metric=metric)
            
            # Append point
            new_ts = ts.append(timestamp, value)
            self._storage[metric.metric_key] = new_ts
            
            # Write to WAL
            if self._wal:
                self._wal.write(metric.metric_key, timestamp, value)
            
            # Update stats
            self._stats["writes"] += 1
            
            # Check high cardinality
            if len(self._storage) > self.config.high_cardinality_threshold:
                self._high_cardinality_metrics.add(metric_name)
        
        return metric.metric_key
    
    def write_batch(
        self,
        data: list[dict[str, Any]],
    ) -> int:
        """
        Write multiple data points in a batch.
        
        Args:
            data: List of dicts with metric_name, labels, value, timestamp
        
        Returns:
            Number of points written
        """
        count = 0
        with self._lock:
            for item in data:
                self.write(
                    metric_name=item["metric_name"],
                    labels=item.get("labels", {}),
                    value=item["value"],
                    timestamp=item.get("timestamp"),
                    value_type=item.get("value_type", ValueType.GAUGE),
                )
                count += 1
        return count
    
    def query(
        self,
        query_str: str,
        time: int | None = None,
    ) -> list[QueryResult]:
        """
        Execute a PromQL query.
        
        Args:
            query_str: PromQL query string
            time: Optional evaluation time (Unix ms)
        
        Returns:
            List of QueryResult objects
        """
        with self._lock:
            self._stats["queries"] += 1
        
        return self._query_engine.execute(query_str, time)
    
    def read(
        self,
        metric_name: str,
        labels: dict[str, str] | None = None,
        start: int | None = None,
        end: int | None = None,
        limit: int = 10000,
    ) -> list[TimeSeries]:
        """
        Read time series data.
        
        Args:
            metric_name: Metric name to read
            labels: Optional label filters
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            limit: Maximum number of points
        
        Returns:
            List of matching TimeSeries
        """
        with self._lock:
            self._stats["reads"] += 1
            
            results = []
            
            for key, ts in self._storage.items():
                if not key.startswith(metric_name):
                    continue
                
                if labels:
                    if not ts.metric.match_labels(labels):
                        continue
                
                if start is not None or end is not None:
                    ts = ts.range_query(start or 0, end or int(time.time() * 1000))
                
                if ts.points:
                    results.append(ts)
                    if len(results) >= limit:
                        break
            
            return results
    
    def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        metric_name: str | None = None,
        time_range: tuple[int, int] | None = None,
        filter_func: callable | None = None,
    ) -> list[Any]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results
            metric_name: Optional metric name filter
            time_range: Optional (start, end) tuple for time filtering
            filter_func: Optional filter function
        
        Returns:
            List of search results
        """
        return self._vector_index.hybrid_search(
            query_vector=query_vector,
            metric_name=metric_name,
            time_range=time_range,
            k=k,
            filter_func=filter_func,
        )
    
    def add_vector(
        self,
        metric_name: str,
        labels: dict[str, str],
        vector: np.ndarray,
        timestamp: int | None = None,
        value: float | None = None,
    ) -> str:
        """
        Add a vector embedding with associated metadata.
        
        Args:
            metric_name: Associated metric name
            labels: Label key-value pairs
            vector: Embedding vector
            timestamp: Optional timestamp
            value: Optional associated value
        
        Returns:
            Vector ID
        """
        with self._lock:
            return self._vector_index.add_vector(
                metric_name=metric_name,
                labels=labels,
                vector=vector,
                timestamp=timestamp,
                value=value,
            )
    
    def series(self, match: str | None = None) -> list[str]:
        """
        List all series labels.
        
        Args:
            match: Optional series selector
        
        Returns:
            List of series keys
        """
        with self._lock:
            if match is None:
                return list(self._storage.keys())
            
            results = []
            for key in self._storage.keys():
                if match in key:
                    results.append(key)
            return results
    
    def label_values(self, label_name: str) -> set[str]:
        """
        Get all unique values for a label.
        
        Args:
            label_name: Name of the label
        
        Returns:
            Set of unique label values
        """
        with self._lock:
            values = set()
            for ts in self._storage.values():
                for label in ts.metric.labels:
                    if label.name == label_name:
                        values.add(label.value)
            return values
    
    def drop_metric(self, metric_name: str, labels: dict[str, str] | None = None) -> int:
        """
        Delete metrics matching criteria.
        
        Args:
            metric_name: Metric name to delete
            labels: Optional label filters
        
        Returns:
            Number of series deleted
        """
        with self._lock:
            deleted = 0
            keys_to_delete = []
            
            for key, ts in self._storage.items():
                if not key.startswith(metric_name):
                    continue
                
                if labels and not ts.metric.match_labels(labels):
                    continue
                
                keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._storage[key]
                deleted += 1
            
            return deleted
    
    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            total_points = sum(len(ts.points) for ts in self._storage.values())
            
            return {
                "num_series": len(self._storage),
                "num_metrics": len(self._metrics),
                "num_vectors": len(self._vector_index.id_mapping),
                "total_points": total_points,
                "writes": self._stats["writes"],
                "reads": self._stats["reads"],
                "queries": self._stats["queries"],
                "uptime_seconds": (int(time.time() * 1000) - self._stats["start_time"]) / 1000,
                "high_cardinality_metrics": list(self._high_cardinality_metrics),
                "vector_index": self._vector_index.get_stats(),
            }
    
    def export(self, format: str = "prometheus") -> str:
        """
        Export data in specified format.
        
        Args:
            format: Export format ('prometheus', 'json')
        
        Returns:
            Exported data as string
        """
        with self._lock:
            if format == "json":
                data = {
                    "metrics": [
                        {
                            "name": ts.metric.name,
                            "labels": {l.name: l.value for l in ts.metric.labels},
                            "points": [(p.timestamp, p.value) for p in ts.points],
                        }
                        for ts in self._storage.values()
                    ]
                }
                return json.dumps(data, indent=2)
            
            elif format == "prometheus":
                lines = []
                for ts in self._storage.values():
                    lines.append(ts.metric.to_prometheus_format())
                    for p in ts.points:
                        lines.append(f"  {p.value} {p.timestamp}")
                return "\n".join(lines)
            
            else:
                raise ValueError(f"Unknown format: {format}")
    
    def save(self, path: str | None = None) -> None:
        """
        Save database to disk.
        
        Args:
            path: Optional path (default: data_dir)
        """
        if path is None:
            path = str(self._data_dir)
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save metadata
            metadata = {
                "config": {
                    "data_dir": str(self._data_dir),
                    "vector_dimension": self.config.vector_dimension,
                },
                "stats": self._stats,
                "high_cardinality": list(self._high_cardinality_metrics),
            }
            
            with open(path / "metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            # Save time series
            for key, ts in self._storage.items():
                filename = self._fingerprint(key) + ".ts"
                filepath = path / filename
                
                with open(filepath, 'wb') as f:
                    f.write(ts.to_bytes())
    
    def load(self, path: str | None = None) -> None:
        """
        Load database from disk.
        
        Args:
            path: Optional path (default: data_dir)
        """
        if path is None:
            path = str(self._data_dir)
        
        path = Path(path)
        
        with self._lock:
            # Load metadata
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self._stats = metadata.get("stats", self._stats)
                    self._high_cardinality_metrics = set(metadata.get("high_cardinality", []))
            
            # Load time series
            for filepath in path.glob("*.ts"):
                with open(filepath, 'rb') as f:
                    # Read metric name from filename
                    key = filepath.stem
                    # This is simplified - in production you'd store full metric info
                    # For now, just restore from key
    
    def _fingerprint(self, key: str) -> str:
        """Generate filename fingerprint."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def close(self) -> None:
        """Close the database and flush all pending writes."""
        if self._wal:
            self._wal.flush()
        
        # Save to disk
        self.save()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
