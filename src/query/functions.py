"""
Aggregation and Rate Functions for PromQL.

This module provides the implementation of various PromQL aggregation
operators and functions.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol

import numpy as np


class AggregationFunctions:
    """
    Collection of aggregation functions for time series.
    
    These functions operate on groups of time series and reduce them
    to single values or smaller groups.
    """
    
    @staticmethod
    def sum(values: list[float]) -> float:
        """Sum all values."""
        return sum(values)
    
    @staticmethod
    def min(values: list[float]) -> float:
        """Return minimum value."""
        return min(values) if values else float('nan')
    
    @staticmethod
    def max(values: list[float]) -> float:
        """Return maximum value."""
        return max(values) if values else float('nan')
    
    @staticmethod
    def avg(values: list[float]) -> float:
        """Return average of values."""
        return np.mean(values) if values else float('nan')
    
    @staticmethod
    def count(values: list[float]) -> float:
        """Return count of values."""
        return float(len(values))
    
    @staticmethod
    def stddev(values: list[float]) -> float:
        """Return standard deviation."""
        return float(np.std(values)) if len(values) > 1 else float('nan')
    
    @staticmethod
    def stdvar(values: list[float]) -> float:
        """Return variance."""
        return float(np.var(values)) if len(values) > 1 else float('nan')
    
    @staticmethod
    def quantile(values: list[float], quantile: float) -> float:
        """Return specified quantile (0-1)."""
        if not values:
            return float('nan')
        return float(np.percentile(values, quantile * 100))
    
    @staticmethod
    def bottomk(values: list[float], k: int) -> list[float]:
        """Return k smallest values."""
        sorted_values = sorted(values)
        return sorted_values[:min(k, len(sorted_values))]
    
    @staticmethod
    def topk(values: list[float], k: int) -> list[float]:
        """Return k largest values."""
        sorted_values = sorted(values, reverse=True)
        return sorted_values[:min(k, len(sorted_values))]


class RateFunctions:
    """
    Collection of rate calculation functions.
    
    These functions calculate rates of change for counter metrics.
    """
    
    @staticmethod
    def rate(points: list[tuple[int, float]], range_ms: int) -> list[tuple[int, float]]:
        """
        Calculate per-second rate of increase.
        
        Args:
            points: List of (timestamp_ms, value) tuples
            range_ms: Time range in milliseconds
        
        Returns:
            List of (timestamp, rate) tuples
        """
        if len(points) < 2:
            return []
        
        result = []
        range_s = range_ms / 1000.0
        
        for i in range(1, len(points)):
            ts1, val1 = points[i - 1]
            ts2, val2 = points[i]
            
            delta_val = val2 - val1
            delta_time = (ts2 - ts1) / 1000.0
            
            if delta_time > 0:
                rate = delta_val / delta_time
                # Rate should be non-negative for counters
                rate = max(0, rate)
                result.append((ts2, rate))
        
        return result
    
    @staticmethod
    def increase(points: list[tuple[int, float]], range_ms: int) -> list[tuple[int, float]]:
        """
        Calculate total increase over time range.
        
        Args:
            points: List of (timestamp_ms, value) tuples
            range_ms: Time range in milliseconds
        
        Returns:
            List of (timestamp, increase) tuples
        """
        if len(points) < 2:
            return []
        
        result = []
        range_s = range_ms / 1000.0
        
        for i in range(1, len(points)):
            ts1, val1 = points[i - 1]
            ts2, val2 = points[i]
            
            delta_val = val2 - val1
            delta_time = (ts2 - ts1) / 1000.0
            
            # Extrapolate to full range
            if delta_time > 0:
                increase = delta_val * (range_s / delta_time)
                result.append((ts2, increase))
        
        return result
    
    @staticmethod
    def irate(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """
        Calculate instant rate from last two points.
        
        Uses only the last two points for calculation, giving
        a more responsive but potentially noisy result.
        
        Args:
            points: List of (timestamp_ms, value) tuples
        
        Returns:
            List of (timestamp, rate) tuples
        """
        if len(points) < 2:
            return []
        
        ts1, val1 = points[-2]
        ts2, val2 = points[-1]
        
        delta_val = val2 - val1
        delta_time = (ts2 - ts1) / 1000.0
        
        if delta_time > 0:
            rate = delta_val / delta_time
            rate = max(0, rate)
            return [(ts2, rate)]
        
        return [(ts2, 0.0)]
    
    @staticmethod
    def delta(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """
        Calculate difference between first and last values.
        
        Unlike rate/increase, this returns raw differences.
        
        Args:
            points: List of (timestamp_ms, value) tuples
        
        Returns:
            List of (timestamp, delta) tuples
        """
        if len(points) < 2:
            return []
        
        result = []
        base_val = points[0][1]
        
        for ts, val in points[1:]:
            result.append((ts, val - base_val))
        
        return result
    
    @staticmethod
    def idelta(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """
        Calculate instant difference (last - previous).
        
        Args:
            points: List of (timestamp_ms, value) tuples
        
        Returns:
            List of (timestamp, delta) tuples
        """
        if len(points) < 2:
            return []
        
        result = []
        for i in range(1, len(points)):
            ts1, val1 = points[i - 1]
            ts2, val2 = points[i]
            result.append((ts2, val2 - val1))
        
        return result


class TimeWindowFunctions:
    """
    Functions that operate over time windows.
    
    These functions apply transformations over sliding time windows.
    """
    
    @staticmethod
    def avg_over_time(
        points: list[tuple[int, float]],
        window_ms: int
    ) -> list[tuple[int, float]]:
        """Calculate average over time window."""
        if len(points) < 2:
            return [(p[0], p[1]) for p in points]
        
        result = []
        for i, (ts, _) in enumerate(points):
            window_start = ts - window_ms
            window_points = [p[1] for p in points if window_start <= p[0] <= ts]
            if window_points:
                result.append((ts, np.mean(window_points)))
        
        return result
    
    @staticmethod
    def min_over_time(
        points: list[tuple[int, float]],
        window_ms: int
    ) -> list[tuple[int, float]]:
        """Calculate minimum over time window."""
        result = []
        for i, (ts, _) in enumerate(points):
            window_start = ts - window_ms
            window_points = [p[1] for p in points if window_start <= p[0] <= ts]
            if window_points:
                result.append((ts, min(window_points)))
        
        return result
    
    @staticmethod
    def max_over_time(
        points: list[tuple[int, float]],
        window_ms: int
    ) -> list[tuple[int, float]]:
        """Calculate maximum over time window."""
        result = []
        for i, (ts, _) in enumerate(points):
            window_start = ts - window_ms
            window_points = [p[1] for p in points if window_start <= p[0] <= ts]
            if window_points:
                result.append((ts, max(window_points)))
        
        return result
    
    @staticmethod
    def sum_over_time(
        points: list[tuple[int, float]],
        window_ms: int
    ) -> list[tuple[int, float]]:
        """Calculate sum over time window."""
        result = []
        for i, (ts, _) in enumerate(points):
            window_start = ts - window_ms
            window_points = [p[1] for p in points if window_start <= p[0] <= ts]
            if window_points:
                result.append((ts, sum(window_points)))
        
        return result
    
    @staticmethod
    def stddev_over_time(
        points: list[tuple[int, float]],
        window_ms: int
    ) -> list[tuple[int, float]]:
        """Calculate standard deviation over time window."""
        result = []
        for i, (ts, _) in enumerate(points):
            window_start = ts - window_ms
            window_points = [p[1] for p in points if window_start <= p[0] <= ts]
            if len(window_points) > 1:
                result.append((ts, np.std(window_points)))
        
        return result
    
    @staticmethod
    def count_over_time(
        points: list[tuple[int, float]],
        window_ms: int
    ) -> list[tuple[int, float]]:
        """Count points over time window."""
        result = []
        for i, (ts, _) in enumerate(points):
            window_start = ts - window_ms
            count = sum(1 for p in points if window_start <= p[0] <= ts)
            result.append((ts, float(count)))
        
        return result


class PredictorFunctions:
    """
    Functions for predicting future values.
    """
    
    @staticmethod
    def predict_linear(
        points: list[tuple[int, float]],
        duration_s: float
    ) -> float:
        """
        Predict future value using linear regression.
        
        Args:
            points: List of (timestamp_ms, value) tuples
            duration_s: Seconds into the future to predict
        
        Returns:
            Predicted value
        """
        if len(points) < 2:
            return points[0][1] if points else 0.0
        
        # Convert to seconds
        x = np.array([p[0] / 1000.0 for p in points])
        y = np.array([p[1] for p in points])
        
        # Linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict
        future_ts = (points[-1][0] / 1000.0) + duration_s
        return slope * future_ts + intercept


class HistogramFunctions:
    """
    Functions for histogram metrics.
    """
    
    @staticmethod
    def histogram_quantile(
        buckets: dict[float, float],
        quantile: float
    ) -> float:
        """
        Calculate quantile from histogram buckets.
        
        Args:
            buckets: Dictionary of bucket upper bounds -> cumulative counts
            quantile: Quantile to calculate (0-1)
        
        Returns:
            Estimated value at the quantile
        """
        if not buckets:
            return float('nan')
        
        sorted_bounds = sorted(buckets.keys())
        total = buckets.get(sorted_bounds[-1], 0)
        
        if total == 0:
            return float('nan')
        
        target = quantile * total
        
        # Find the bucket containing the quantile
        cumulative = 0.0
        for bound in sorted_bounds:
            cumulative = buckets[bound]
            if cumulative >= target:
                # Linear interpolation within bucket
                prev_cumulative = 0.0
                if bound != sorted_bounds[0]:
                    prev_bound = sorted_bounds[sorted_bounds.index(bound) - 1]
                    prev_cumulative = buckets.get(prev_bound, 0)
                
                if cumulative != prev_cumulative:
                    ratio = (target - prev_cumulative) / (cumulative - prev_cumulative)
                    return bound * ratio + (sorted_bounds[sorted_bounds.index(bound) - 1] if bound != sorted_bounds[0] else 0) * (1 - ratio)
                return bound
        
        return sorted_bounds[-1]
    
    @staticmethod
    def histogram_fraction(
        lower: float,
        upper: float,
        buckets: dict[float, float]
    ) -> float:
        """
        Calculate fraction of observations within bounds.
        
        Args:
            lower: Lower bound
            upper: Upper bound
            buckets: Histogram buckets
        
        Returns:
            Fraction of observations
        """
        if not buckets:
            return float('nan')
        
        sorted_bounds = sorted(buckets.keys())
        total = buckets.get(sorted_bounds[-1], 0)
        
        if total == 0:
            return float('nan')
        
        # Count observations within bounds
        count = 0.0
        prev_bound = 0.0
        
        for bound in sorted_bounds:
            if bound <= upper:
                count += buckets[bound]
            if bound >= lower:
                break
        
        # Subtract lower bound
        for bound in sorted_bounds:
            if bound <= lower:
                count -= buckets[bound]
        
        return count / total
