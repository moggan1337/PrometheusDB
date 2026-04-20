"""
PromQL-Compatible Query Engine.

This module implements a PromQL-compatible query language for time-series data.
It supports:
- Metric selectors with label matching
- Range queries with time windows
- Aggregation operators (sum, avg, min, max, count, stddev)
- Rate and increase calculations
- Binary operators with matching
- Functions (rate, increase, irate, avg_over_time, etc.)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np

from ..storage.schema import (
    Metric, TimeSeries, DataPoint, Label,
    ValueType, Label as SchemaLabel
)


@dataclass
class QueryResult:
    """
    Result of a PromQL query.
    
    Attributes:
        metric: The metric this result belongs to
        values: List of (timestamp, value) tuples
        vector: For instant queries, single value
        type: Query type ('vector', 'matrix', 'scalar')
    """
    metric: Metric
    values: list[tuple[int, float]]
    vector: tuple[int, float] | None = None
    query_type: str = "vector"
    
    def __post_init__(self):
        if self.vector is None and self.values:
            self.vector = self.values[-1]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric": self.metric.to_prometheus_format(),
            "values": [{"timestamp": ts, "value": val} for ts, val in self.values],
            "vector": {"timestamp": self.vector[0], "value": self.vector[1]} if self.vector else None,
            "type": self.query_type,
        }
    
    def to_prometheus_format(self) -> str:
        """Format in Prometheus exposition format."""
        lines = []
        metric_name = self.metric.name
        
        for ts, val in self.values:
            label_str = ""
            if self.metric.labels:
                labels = []
                for l in sorted(self.metric.labels, key=lambda x: x.name):
                    labels.append(f'{l.name}="{l.value}"')
                label_str = "{" + ",".join(labels) + "}"
            
            lines.append(f"{metric_name}{label_str} {val} {ts}")
        
        return "\n".join(lines)


class QueryType(Enum):
    """Type of PromQL query."""
    INSTANT = "instant"      # Single value at a point in time
    RANGE = "range"         # Values over a time range
    AGGREGATION = "aggregation"  # Aggregated results


@dataclass
class Query:
    """
    Parsed PromQL query.
    
    Attributes:
        metric_name: Name of the metric
        labels: Label matchers
        range_ms: Time range in milliseconds (for range queries)
        start: Start timestamp
        end: End timestamp
        step: Query resolution step
        aggregations: List of aggregation operators
        functions: List of applied functions
    """
    metric_name: str
    labels: dict[str, str] = field(default_factory=dict)
    label_matchers: list[tuple[str, str, str]] = field(default_factory=dict)  # (name, op, value)
    range_ms: int = 0
    start: int = field(default_factory=lambda: int(time.time() * 1000))
    end: int = field(default_factory=lambda: int(time.time() * 1000))
    step_ms: int = 15000
    aggregations: list[dict[str, Any]] = field(default_factory=list)
    functions: list[tuple[str, list[Any]]] = field(default_factory=list)
    group_by_labels: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.end == 0:
            self.end = int(time.time() * 1000)
        if self.start == 0:
            self.start = self.end - 300000  # 5 minutes ago


@dataclass
class TimeSeriesIterator:
    """Wrapper for iterating over time series data."""
    series: TimeSeries
    current_index: int = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> DataPoint:
        if self.current_index >= len(self.series.points):
            raise StopIteration
        point = self.series.points[self.current_index]
        self.current_index += 1
        return point


class QueryEngine:
    """
    PromQL-compatible query engine.
    
    This engine supports a wide range of PromQL features:
    - Instant and range vector queries
    - Label matching (=, !=, =~, !~)
    - Range selectors [5m], [1h]
    - Aggregation operators
    - Rate and increase functions
    - Binary arithmetic operators
    
    Example:
        >>> engine = QueryEngine(storage)
        >>> result = engine.execute('rate(http_requests_total[5m])')
        >>> result = engine.execute('sum by (method) (rate(http_requests_total[5m]))')
    """
    
    # Regex patterns for PromQL parsing
    LABEL_MATCHER = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|!=|=~|!~)\s*"([^"]*)"')
    FUNCTION_CALL = re.compile(r'(\w+)\s*\(\s*([^)]+)\)')
    RANGE_SELECTOR = re.compile(r'\[(\d+)([smhdwy])\]')
    AGGREGATION_OP = re.compile(
        r'(sum|avg|min|max|count|stddev|stdvar|bottomk|topk|'
        r'count_values|quantile)\s*(?:by\s*\(([^)]+)\))?\s*\((.+)\)'
    )
    
    def __init__(self, storage: Any = None):
        """
        Initialize query engine.
        
        Args:
            storage: Storage backend (if None, uses in-memory storage)
        """
        self.storage = storage or InMemoryStorage()
        self._functions = self._register_functions()
        self._aggregations = self._register_aggregations()
    
    def _register_functions(self) -> dict[str, Callable]:
        """Register built-in PromQL functions."""
        return {
            "rate": self._rate,
            "increase": self._increase,
            "irate": self._irate,
            "avg_over_time": self._avg_over_time,
            "sum_over_time": self._sum_over_time,
            "min_over_time": self._min_over_time,
            "max_over_time": self._max_over_time,
            "count_over_time": self._count_over_time,
            "stddev_over_time": self._stddev_over_time,
            "absent": self._absent,
            "abs": self._abs,
            "ceil": self._ceil,
            "floor": self._floor,
            "round": self._round,
            "sqrt": self._sqrt,
            "exp": self._exp,
            "ln": self._ln,
            "log2": self._log2,
            "log10": self._log10,
            "scalar": self._scalar,
            "vector": self._vector,
            "clamp_max": self._clamp_max,
            "clamp_min": self._clamp_min,
        }
    
    def _register_aggregations(self) -> dict[str, Callable]:
        """Register aggregation operators."""
        return {
            "sum": np.sum,
            "avg": np.mean,
            "min": np.min,
            "max": np.max,
            "count": len,
            "stddev": np.std,
            "stdvar": np.var,
        }
    
    def parse(self, query: str) -> Query:
        """
        Parse a PromQL query string.
        
        Args:
            query: PromQL query string
        
        Returns:
            Parsed Query object
        """
        query = query.strip()
        
        parsed = Query(metric_name="")
        
        # Extract range selector [5m], [1h], etc.
        range_match = self.RANGE_SELECTOR.search(query)
        if range_match:
            value, unit = range_match.groups()
            multipliers = {'s': 1000, 'm': 60000, 'h': 3600000, 'd': 86400000, 'w': 604800000}
            parsed.range_ms = int(value) * multipliers.get(unit, 1)
            query = self.RANGE_SELECTOR.sub('', query)
        
        # Check for aggregation
        agg_match = self.AGGREGATION_OP.match(query)
        if agg_match:
            op, by_labels, inner_query = agg_match.groups()
            query = inner_query.strip()
            parsed.aggregations.append({
                "operator": op,
                "by_labels": [l.strip() for l in by_labels.split(',')] if by_labels else []
            })
        
        # Check for function call
        func_match = self.FUNCTION_CALL.match(query)
        if func_match:
            func_name, func_args = func_match.groups()
            parsed.functions.append((func_name, self._parse_function_args(func_args)))
            query = query[func_match.end():].strip()
        
        # Extract metric name and labels
        # Format: metric_name{label="value", ...}
        metric_pattern = re.match(r'([a-zA-Z_][a-zA-Z0-9_:]*)', query)
        if metric_pattern:
            parsed.metric_name = metric_pattern.group(1)
            query = query[metric_pattern.end():]
        
        # Parse labels
        label_match = self.LABEL_MATCHER.findall(query)
        for name, op, value in label_match:
            parsed.label_matchers.append((name, op, value))
            if op == '=':
                parsed.labels[name] = value
        
        return parsed
    
    def _parse_function_args(self, args: str) -> list[Any]:
        """Parse function arguments."""
        args = args.strip()
        if not args:
            return []
        
        # Split by comma, but respect nested brackets
        result = []
        depth = 0
        current = ""
        
        for char in args:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                result.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            result.append(current.strip())
        
        # Try to convert to appropriate types
        parsed = []
        for arg in result:
            arg = arg.strip()
            try:
                parsed.append(int(arg))
            except ValueError:
                try:
                    parsed.append(float(arg))
                except ValueError:
                    parsed.append(arg)
        
        return parsed
    
    def execute(self, query: str, time: int | None = None) -> list[QueryResult]:
        """
        Execute a PromQL query.
        
        Args:
            query: PromQL query string
            time: Optional evaluation time (Unix ms)
        
        Returns:
            List of QueryResult objects
        """
        parsed = self.parse(query)
        
        if time is None:
            time = int(time.time() * 1000)
        
        # Set time range
        if parsed.range_ms > 0:
            parsed.start = time - parsed.range_ms
            parsed.end = time
        else:
            parsed.start = time
            parsed.end = time
        
        # Fetch matching time series
        series_list = self._fetch_series(parsed)
        
        # Apply functions
        for func_name, func_args in parsed.functions:
            series_list = self._apply_function(func_name, func_args, series_list, parsed)
        
        # Apply aggregations
        for agg in parsed.aggregations:
            series_list = self._apply_aggregation(agg, series_list)
        
        # Convert to results
        results = []
        for series in series_list:
            values = [(p.timestamp, p.value) for p in series.points]
            results.append(QueryResult(
                metric=series.metric,
                values=values,
                query_type="matrix" if parsed.range_ms > 0 else "vector"
            ))
        
        return results
    
    def _fetch_series(self, query: Query) -> list[TimeSeries]:
        """Fetch time series matching the query."""
        # This would normally query the storage
        # For now, return empty list
        return []
    
    def _apply_function(
        self,
        func_name: str,
        args: list[Any],
        series_list: list[TimeSeries],
        query: Query,
    ) -> list[TimeSeries]:
        """Apply a function to time series."""
        if func_name not in self._functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        func = self._functions[func_name]
        
        if func_name in ["rate", "increase", "irate"]:
            # These require range data
            return func(series_list, query.range_ms)
        elif func_name.endswith("_over_time"):
            # Time-based functions
            return func(series_list)
        else:
            # Scalar functions
            return func(series_list, *args)
    
    def _apply_aggregation(
        self,
        agg: dict[str, Any],
        series_list: list[TimeSeries],
    ) -> list[TimeSeries]:
        """Apply aggregation to time series."""
        op = agg["operator"]
        by_labels = agg.get("by_labels", [])
        
        if op in ["sum", "avg", "min", "max", "count", "stddev", "stdvar"]:
            return self._aggregate_by_labels(series_list, op, by_labels)
        elif op == "topk":
            return self._topk(series_list, args=[agg])
        elif op == "bottomk":
            return self._bottomk(series_list, args=[agg])
        
        return series_list
    
    def _rate(self, series_list: list[TimeSeries], range_ms: int) -> list[TimeSeries]:
        """Calculate per-second rate of increase."""
        if range_ms == 0:
            range_ms = 60000  # Default 1 minute
        
        results = []
        for series in series_list:
            new_points = []
            points = series.points
            
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                delta_val = p2.value - p1.value
                delta_time = (p2.timestamp - p1.timestamp) / 1000  # Convert to seconds
                
                if delta_time > 0:
                    rate = delta_val / delta_time
                    new_points.append(DataPoint(
                        timestamp=p2.timestamp,
                        value=max(0, rate)  # Rate should be non-negative for counters
                    ))
            
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _increase(self, series_list: list[TimeSeries], range_ms: int) -> list[TimeSeries]:
        """Calculate total increase over time range."""
        if range_ms == 0:
            range_ms = 60000
        
        results = []
        for series in series_list:
            if len(series.points) < 2:
                results.append(TimeSeries(metric=series.metric))
                continue
            
            p1 = series.points[0]
            p2 = series.points[-1]
            
            increase = p2.value - p1.value
            
            new_point = DataPoint(timestamp=p2.timestamp, value=increase)
            results.append(TimeSeries(metric=series.metric, points=[new_point]))
        
        return results
    
    def _irate(self, series_list: list[TimeSeries], range_ms: int) -> list[TimeSeries]:
        """Calculate instant rate (per-second) from the last two points."""
        results = []
        for series in series_list:
            if len(series.points) < 2:
                results.append(TimeSeries(metric=series.metric))
                continue
            
            p1, p2 = series.points[-2], series.points[-1]
            delta_val = p2.value - p1.value
            delta_time = (p2.timestamp - p1.timestamp) / 1000
            
            if delta_time > 0:
                rate = delta_val / delta_time
                new_point = DataPoint(timestamp=p2.timestamp, value=max(0, rate))
            else:
                new_point = DataPoint(timestamp=p2.timestamp, value=0)
            
            results.append(TimeSeries(metric=series.metric, points=[new_point]))
        
        return results
    
    def _avg_over_time(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Calculate average over time."""
        results = []
        for series in series_list:
            if not series.points:
                results.append(TimeSeries(metric=series.metric))
                continue
            
            avg = np.mean([p.value for p in series.points])
            ts = series.points[-1].timestamp if series.points else 0
            results.append(TimeSeries(
                metric=series.metric,
                points=[DataPoint(timestamp=ts, value=avg)]
            ))
        
        return results
    
    def _sum_over_time(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Calculate sum over time."""
        results = []
        for series in series_list:
            if not series.points:
                results.append(TimeSeries(metric=series.metric))
                continue
            
            total = sum(p.value for p in series.points)
            ts = series.points[-1].timestamp if series.points else 0
            results.append(TimeSeries(
                metric=series.metric,
                points=[DataPoint(timestamp=ts, value=total)]
            ))
        
        return results
    
    def _min_over_time(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Calculate minimum over time."""
        results = []
        for series in series_list:
            if not series.points:
                results.append(TimeSeries(metric=series.metric))
                continue
            
            min_val = min(p.value for p in series.points)
            ts = series.points[-1].timestamp if series.points else 0
            results.append(TimeSeries(
                metric=series.metric,
                points=[DataPoint(timestamp=ts, value=min_val)]
            ))
        
        return results
    
    def _max_over_time(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Calculate maximum over time."""
        results = []
        for series in series_list:
            if not series.points:
                results.append(TimeSeries(metric=series.metric))
                continue
            
            max_val = max(p.value for p in series.points)
            ts = series.points[-1].timestamp if series.points else 0
            results.append(TimeSeries(
                metric=series.metric,
                points=[DataPoint(timestamp=ts, value=max_val)]
            ))
        
        return results
    
    def _count_over_time(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Count points over time."""
        results = []
        for series in series_list:
            ts = series.points[-1].timestamp if series.points else 0
            results.append(TimeSeries(
                metric=series.metric,
                points=[DataPoint(timestamp=ts, value=len(series.points))]
            ))
        
        return results
    
    def _stddev_over_time(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Calculate standard deviation over time."""
        results = []
        for series in series_list:
            if len(series.points) < 2:
                results.append(TimeSeries(metric=series.metric))
                continue
            
            values = [p.value for p in series.points]
            stddev = np.std(values)
            ts = series.points[-1].timestamp if series.points else 0
            results.append(TimeSeries(
                metric=series.metric,
                points=[DataPoint(timestamp=ts, value=stddev)]
            ))
        
        return results
    
    def _abs(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Absolute value."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=abs(p.value))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _ceil(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Ceiling function."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=np.ceil(p.value))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _floor(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Floor function."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=np.floor(p.value))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _round(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Round to nearest integer."""
        results = []
        for series in series_list:
            precision = args[0] if args else 0
            multiplier = 10 ** precision
            new_points = [
                DataPoint(timestamp=p.timestamp, value=round(p.value / multiplier) * multiplier)
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _sqrt(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Square root."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=np.sqrt(max(0, p.value)))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _exp(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Exponential."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=np.exp(p.value))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _ln(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Natural logarithm."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=np.log(p.value) if p.value > 0 else float('nan'))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _log2(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Base-2 logarithm."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=np.log2(p.value) if p.value > 0 else float('nan'))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _log10(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Base-10 logarithm."""
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=np.log10(p.value) if p.value > 0 else float('nan'))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _scalar(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Convert to scalar if single value."""
        results = []
        for series in series_list:
            if len(series.points) == 1:
                results.append(TimeSeries(metric=series.metric, points=series.points))
            else:
                # Return NaN for non-single-value series
                results.append(TimeSeries(
                    metric=series.metric,
                    points=[DataPoint(timestamp=0, value=float('nan'))]
                ))
        
        return results
    
    def _vector(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Convert scalar to vector."""
        return series_list
    
    def _clamp_max(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Clamp values to maximum."""
        max_val = args[0] if args else 0
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=min(p.value, max_val))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _clamp_min(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Clamp values to minimum."""
        min_val = args[0] if args else 0
        results = []
        for series in series_list:
            new_points = [
                DataPoint(timestamp=p.timestamp, value=max(p.value, min_val))
                for p in series.points
            ]
            results.append(TimeSeries(metric=series.metric, points=new_points))
        
        return results
    
    def _absent(self, series_list: list[TimeSeries], *args) -> list[TimeSeries]:
        """Return 1 if series is empty, 0 otherwise."""
        results = []
        for series in series_list:
            value = 1.0 if len(series.points) == 0 else 0.0
            results.append(TimeSeries(
                metric=Metric(name="absent", labels=frozenset()),
                points=[DataPoint(timestamp=int(time.time() * 1000), value=value)]
            ))
        
        return results
    
    def _aggregate_by_labels(
        self,
        series_list: list[TimeSeries],
        op: str,
        group_by: list[str],
    ) -> list[TimeSeries]:
        """Aggregate series by label groups."""
        # Group series by label values
        groups: dict[tuple, list[TimeSeries]] = {}
        
        for series in series_list:
            # Build group key from specified labels
            labels = {l.name: l.value for l in series.metric.labels}
            key = tuple(labels.get(l, "__total") for l in group_by)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(series)
        
        # Aggregate each group
        results = []
        agg_func = self._aggregations.get(op, np.mean)
        
        for key, group in groups.items():
            if not group:
                continue
            
            # Create aggregated metric
            label_dict = dict(zip(group_by, key))
            new_labels = frozenset(SchemaLabel(name=k, value=v) for k, v in label_dict.items())
            new_metric = Metric(name=group[0].metric.name, labels=new_labels)
            
            # Aggregate values at each timestamp
            if not group[0].points:
                continue
            
            aggregated_points = []
            timestamps = sorted(set(p.timestamp for s in group for p in s.points))
            
            for ts in timestamps:
                values = []
                for s in group:
                    for p in s.points:
                        if p.timestamp == ts:
                            values.append(p.value)
                            break
                
                if values:
                    agg_value = agg_func(values)
                    aggregated_points.append(DataPoint(timestamp=ts, value=agg_value))
            
            results.append(TimeSeries(metric=new_metric, points=aggregated_points))
        
        return results
    
    def _topk(self, series_list: list[TimeSeries], k: int = 5) -> list[TimeSeries]:
        """Return top k series by latest value."""
        sorted_series = sorted(
            series_list,
            key=lambda s: s.points[-1].value if s.points else 0,
            reverse=True
        )
        return sorted_series[:k]
    
    def _bottomk(self, series_list: list[TimeSeries], k: int = 5) -> list[TimeSeries]:
        """Return bottom k series by latest value."""
        sorted_series = sorted(
            series_list,
            key=lambda s: s.points[-1].value if s.points else 0
        )
        return sorted_series[:k]


class InMemoryStorage:
    """Simple in-memory storage for testing."""
    
    def __init__(self):
        self.series: dict[str, TimeSeries] = {}
    
    def add(self, metric: Metric, timestamp: int, value: float) -> None:
        """Add a data point."""
        key = metric.metric_key
        if key not in self.series:
            self.series[key] = TimeSeries(metric=metric)
        self.series[key] = self.series[key].append(timestamp, value)
    
    def query(self, metric_name: str, labels: dict[str, str], start: int, end: int) -> list[TimeSeries]:
        """Query time series."""
        results = []
        for key, series in self.series.items():
            if key.startswith(metric_name):
                if series.metric.match_labels(labels):
                    filtered = series.range_query(start, end)
                    if filtered.points:
                        results.append(filtered)
        return results
