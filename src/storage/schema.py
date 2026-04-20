"""
Core schema definitions for PrometheusDB.

This module defines the fundamental data structures:
- Label: Key-value pairs for metric metadata
- Metric: Named metrics with labels and time-series data
- TimeSeries: Sequences of timestamped values
- Vector: Embedding vectors for similarity search
- RetentionPolicy: Data retention configurations
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterator

import numpy as np


class ValueType(Enum):
    """Type of metric value."""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Label:
    """
    A label is a key-value pair that provides metadata for metrics.
    
    Labels enable multidimensional metrics in Prometheus-style databases,
    allowing flexible querying and aggregation.
    
    Example:
        >>> label = Label(name="instance", value="server01:8080")
        >>> label.name
        'instance'
        >>> label.value
        'server01:8080'
    """
    name: str
    value: str
    
    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Label name must be a non-empty string")
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Label value must be a non-empty string")
    
    @property
    def fingerprint(self) -> int:
        """Compute a 64-bit fingerprint for the label."""
        content = f"{self.name}={self.value}".encode('utf-8')
        return int.from_bytes(hashlib.fnv1a64(content), byteorder='big')
    
    def matches(self, other: Label) -> bool:
        """Check if this label matches another (same name, value can be wildcarded)."""
        if other.value == "" or other.value == "*":
            return self.name == other.name
        return self.name == other.name and self.value == other.value
    
    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {self.name: self.value}
    
    @classmethod
    def from_dict(cls, d: dict[str, str]) -> frozenset[Label]:
        """Create a set of Labels from a dictionary."""
        return frozenset(cls(name=k, value=v) for k, v in d.items())


@dataclass(frozen=True, order=True, slots=True)
class Metric:
    """
    A metric is a named time series with associated labels.
    
    The metric name describes what is being measured (e.g., 'http_requests_total').
    Labels provide dimensions for the metric (e.g., method="GET", status="200").
    
    The combination of metric name + unique label set forms a unique time series.
    """
    name: str
    labels: frozenset[Label]
    value_type: ValueType = ValueType.GAUGE
    unit: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Metric name cannot be empty")
        if not isinstance(self.labels, frozenset):
            object.__setattr__(self, 'labels', frozenset(self.labels))
    
    @property
    def fingerprint(self) -> int:
        """
        Compute a 64-bit fingerprint for the metric.
        
        Uses a combination of metric name and label fingerprint
        to create a unique identifier for this specific time series.
        """
        name_hash = int.from_bytes(hashlib.fnv1a64(self.name.encode('utf-8')), byteorder='big')
        labels_hash = hash(frozenset(l.fingerprint for l in self.labels))
        return (name_hash << 16) ^ (labels_hash & 0xFFFFFFFFFFFF)
    
    @property
    def metric_key(self) -> str:
        """Generate a canonical string key for the metric."""
        label_str = ",".join(f"{l.name}=\"{l.value}\"" for l in sorted(self.labels, key=lambda x: x.name))
        return f"{self.name}{{{label_str}}}"
    
    def with_labels(self, **labels: str) -> Metric:
        """Create a new metric with additional or modified labels."""
        new_labels = set(self.labels)
        for name, value in labels.items():
            new_labels.add(Label(name=name, value=value))
        return Metric(
            name=self.name,
            labels=frozenset(new_labels),
            value_type=self.value_type,
            unit=self.unit,
            description=self.description
        )
    
    def match_labels(self, selector: dict[str, str]) -> bool:
        """
        Check if this metric's labels match a selector.
        
        Args:
            selector: Dictionary of label name -> value pairs to match.
                     Empty value or "*" acts as wildcard.
        
        Returns:
            True if all selector labels match this metric's labels.
        """
        metric_labels = {l.name: l.value for l in self.labels}
        for name, value in selector.items():
            if name not in metric_labels:
                return False
            if value != "" and value != "*" and metric_labels[name] != value:
                return False
        return True
    
    def to_prometheus_format(self) -> str:
        """Format the metric in Prometheus exposition format."""
        label_parts = []
        for l in sorted(self.labels, key=lambda x: x.name):
            escaped_value = l.value.replace("\\", "\\\\").replace('"', '\\"')
            label_parts.append(f'{l.name}="{escaped_value}"')
        labels_str = ",".join(label_parts)
        return f"{self.name}{{{labels_str}}}"


@dataclass
class DataPoint:
    """
    A single timestamped data point in a time series.
    
    Attributes:
        timestamp: Unix timestamp in milliseconds (int64)
        value: The measured value (float64)
    """
    timestamp: int  # Unix timestamp in milliseconds
    value: float
    
    def __post_init__(self):
        if self.timestamp < 0:
            raise ValueError("Timestamp cannot be negative")
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime object."""
        return datetime.fromtimestamp(self.timestamp / 1000.0)
    
    @property
    def seconds(self) -> float:
        """Get timestamp as seconds."""
        return self.timestamp / 1000.0
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        return struct.pack('>qd', self.timestamp, self.value)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> DataPoint:
        """Deserialize from bytes."""
        timestamp, value = struct.unpack('>qd', data)
        return cls(timestamp=int(timestamp), value=value)
    
    def delta_to(self, other: DataPoint) -> float:
        """Calculate value difference between this and another point."""
        return other.value - self.value


@dataclass
class TimeSeries:
    """
    A time series containing a sequence of timestamped data points.
    
    Time series are immutable once created, but new points can be appended
    via the append() method which creates a new instance.
    
    Attributes:
        metric: The metric this time series belongs to
        points: Sorted list of data points (by timestamp)
    """
    metric: Metric
    points: list[DataPoint] = field(default_factory=list)
    
    def __post_init__(self):
        self.points.sort(key=lambda p: p.timestamp)
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __iter__(self) -> Iterator[DataPoint]:
        return iter(self.points)
    
    def __getitem__(self, index: int) -> DataPoint:
        return self.points[index]
    
    @property
    def start_time(self) -> int | None:
        """Get the timestamp of the first data point."""
        return self.points[0].timestamp if self.points else None
    
    @property
    def end_time(self) -> int | None:
        """Get the timestamp of the last data point."""
        return self.points[-1].timestamp if self.points else None
    
    @property
    def duration_ms(self) -> int | None:
        """Get the duration of the time series in milliseconds."""
        if not self.points or len(self.points) < 2:
            return None
        return self.points[-1].timestamp - self.points[0].timestamp
    
    @property
    def values(self) -> np.ndarray:
        """Get all values as a numpy array."""
        return np.array([p.value for p in self.points])
    
    @property
    def timestamps(self) -> np.ndarray:
        """Get all timestamps as a numpy array."""
        return np.array([p.timestamp for p in self.points])
    
    def append(self, timestamp: int, value: float) -> TimeSeries:
        """Create a new time series with an additional data point."""
        new_points = self.points.copy()
        new_points.append(DataPoint(timestamp=timestamp, value=value))
        new_points.sort(key=lambda p: p.timestamp)
        return TimeSeries(metric=self.metric, points=new_points)
    
    def range_query(self, start: int, end: int) -> TimeSeries:
        """
        Extract a time range from the series.
        
        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
        
        Returns:
            New TimeSeries containing only points within the range.
        """
        filtered = [p for p in self.points if start <= p.timestamp <= end]
        return TimeSeries(metric=self.metric, points=filtered)
    
    def downsampled(self, interval_ms: int, aggregator: str = "avg") -> TimeSeries:
        """
        Downsample the time series to a fixed interval.
        
        Args:
            interval_ms: Target interval in milliseconds
            aggregator: How to aggregate points ('avg', 'sum', 'min', 'max', 'first', 'last')
        
        Returns:
            New downsampled TimeSeries.
        """
        if not self.points:
            return TimeSeries(metric=self.metric)
        
        buckets: dict[int, list[float]] = {}
        for p in self.points:
            bucket = (p.timestamp // interval_ms) * interval_ms
            buckets.setdefault(bucket, []).append(p.value)
        
        aggregators = {
            'avg': lambda v: sum(v) / len(v),
            'sum': sum,
            'min': min,
            'max': max,
            'first': lambda v: v[0],
            'last': lambda v: v[-1],
        }
        
        agg_fn = aggregators.get(aggregator, aggregators['avg'])
        new_points = [
            DataPoint(timestamp=ts, value=agg_fn(vals))
            for ts, vals in sorted(buckets.items())
        ]
        
        return TimeSeries(metric=self.metric, points=new_points)
    
    def to_bytes(self) -> bytes:
        """Serialize the time series to bytes."""
        header = struct.pack('>I', len(self.points))
        points_data = b''.join(p.to_bytes() for p in self.points)
        return header + points_data
    
    @classmethod
    def from_bytes(cls, metric: Metric, data: bytes) -> TimeSeries:
        """Deserialize a time series from bytes."""
        num_points = struct.unpack('>I', data[:4])[0]
        points = []
        offset = 4
        for _ in range(num_points):
            points.append(DataPoint.from_bytes(data[offset:offset+16]))
            offset += 16
        return cls(metric=metric, points=points)


@dataclass
class Vector:
    """
    A vector embedding for similarity search.
    
    Vectors are used for semantic search and approximate nearest neighbor
    (ANN) queries. Each vector is associated with a metric and optional
    timestamp for hybrid queries.
    
    Attributes:
        id: Unique identifier for the vector
        values: The embedding values (numpy array)
        metric: Associated metric (if any)
        timestamp: Associated timestamp (if any)
        metadata: Additional metadata dict
    """
    id: str
    values: np.ndarray
    metric: Metric | None = None
    timestamp: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float32)
        if self.values.ndim != 1:
            raise ValueError("Vector must be 1-dimensional")
    
    @property
    def dimension(self) -> int:
        """Get the dimensionality of the vector."""
        return len(self.values)
    
    @property
    def norm(self) -> float:
        """Compute the L2 norm of the vector."""
        return float(np.linalg.norm(self.values))
    
    def normalized(self) -> Vector:
        """Get a normalized copy of the vector (L2)."""
        norm = self.norm
        if norm == 0:
            return Vector(id=self.id, values=self.values.copy(), metric=self.metric,
                         timestamp=self.timestamp, metadata=self.metadata.copy())
        return Vector(
            id=self.id,
            values=self.values / norm,
            metric=self.metric,
            timestamp=self.timestamp,
            metadata=self.metadata.copy()
        )
    
    def cosine_similarity(self, other: Vector) -> float:
        """Compute cosine similarity with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        dot_product = float(np.dot(self.values, other.values))
        return dot_product / (self.norm * other.norm + 1e-10)
    
    def euclidean_distance(self, other: Vector) -> float:
        """Compute Euclidean distance to another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        return float(np.linalg.norm(self.values - other.values))
    
    def dot_product(self, other: Vector) -> float:
        """Compute dot product with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        return float(np.dot(self.values, other.values))
    
    def to_bytes(self) -> bytes:
        """Serialize vector to bytes."""
        header = struct.pack('>I', self.dimension)
        values_data = self.values.tobytes()
        timestamp_data = struct.pack('>q', self.timestamp or 0)
        return header + values_data + timestamp_data
    
    @classmethod
    def from_bytes(cls, id: str, data: bytes) -> Vector:
        """Deserialize vector from bytes."""
        dim = struct.unpack('>I', data[:4])[0]
        values = np.frombuffer(data[4:4+dim*4], dtype=np.float32)
        timestamp = struct.unpack('>q', data[4+dim*4:4+dim*4+8])[0]
        timestamp = timestamp if timestamp != 0 else None
        return cls(id=id, values=values, timestamp=timestamp)


@dataclass
class RetentionPolicy:
    """
    A retention policy defines how long data is kept and how it's compressed.
    
    Policies can be applied to different metric namespaces, allowing fine-grained
    control over data retention based on metric type or importance.
    
    Attributes:
        name: Unique name for the policy
        duration: How long to retain data (in seconds)
        resolution: Sample resolution ('raw', '5m', '1h', '1d')
        compression_enabled: Whether to apply compression
        archive_enabled: Whether to move to archive storage after retention
        downsampling_rules: Rules for automatic downsampling
    """
    name: str
    duration_seconds: int  # How long to keep data
    resolution: str = "raw"  # 'raw', '5m', '1h', '1d'
    compression_enabled: bool = True
    archive_enabled: bool = False
    downsampling_rules: list[DownsamplingRule] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    
    def __post_init__(self):
        valid_resolutions = {'raw', '5m', '1h', '1d', '1w'}
        if self.resolution not in valid_resolutions:
            raise ValueError(f"Invalid resolution: {self.resolution}. Must be one of {valid_resolutions}")
    
    @property
    def duration(self) -> timedelta:
        """Get duration as timedelta."""
        return timedelta(seconds=self.duration_seconds)
    
    @property
    def resolution_ms(self) -> int:
        """Get resolution in milliseconds."""
        multipliers = {
            'raw': 1,
            '5m': 5 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        return multipliers.get(self.resolution, 1)
    
    def is_expired(self, timestamp: int) -> bool:
        """Check if a timestamp is older than the retention period."""
        cutoff = int(time.time() * 1000) - (self.duration_seconds * 1000)
        return timestamp < cutoff
    
    def should_downsample(self, interval_ms: int) -> tuple[bool, str]:
        """Check if data should be downsampled based on resolution."""
        if self.resolution == 'raw':
            return False, 'raw'
        
        if interval_ms > self.resolution_ms:
            return True, self.resolution
        return False, self.resolution
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'duration_seconds': self.duration_seconds,
            'resolution': self.resolution,
            'compression_enabled': self.compression_enabled,
            'archive_enabled': self.archive_enabled,
            'created_at': self.created_at,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetentionPolicy:
        """Create from dictionary representation."""
        return cls(
            name=d['name'],
            duration_seconds=d['duration_seconds'],
            resolution=d.get('resolution', 'raw'),
            compression_enabled=d.get('compression_enabled', True),
            archive_enabled=d.get('archive_enabled', False),
            created_at=d.get('created_at', int(time.time() * 1000)),
        )


@dataclass
class DownsamplingRule:
    """Rule for automatic downsampling of time series data."""
    source_resolution: str
    target_resolution: str
    aggregator: str  # 'avg', 'sum', 'min', 'max', 'first', 'last', 'p50', 'p95', 'p99'
    min_points_before_aggregate: int = 10
    
    @property
    def interval_ms(self) -> int:
        """Get target interval in milliseconds."""
        multipliers = {
            '5m': 5 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        return multipliers.get(self.target_resolution, 0)


# Common retention policies
DEFAULT_RETENTION = RetentionPolicy(
    name="default",
    duration_seconds=15 * 24 * 60 * 60,  # 15 days
    resolution="raw",
)

SHORT_TERM_RETENTION = RetentionPolicy(
    name="short_term",
    duration_seconds=24 * 60 * 60,  # 1 day
    resolution="raw",
)

LONG_TERM_RETENTION = RetentionPolicy(
    name="long_term",
    duration_seconds=365 * 24 * 60 * 60,  # 1 year
    resolution="1h",
)

METRICS_BUCKET_RETENTION = {
    "15d": RetentionPolicy(name="15d", duration_seconds=15 * 86400, resolution="raw"),
    "30d": RetentionPolicy(name="30d", duration_seconds=30 * 86400, resolution="5m"),
    "90d": RetentionPolicy(name="90d", duration_seconds=90 * 86400, resolution="1h"),
    "1y": RetentionPolicy(name="1y", duration_seconds=365 * 86400, resolution="1d"),
}
