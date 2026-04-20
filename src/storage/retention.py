"""
Retention Policy Manager.

This module handles automatic data retention and downsampling
based on configurable policies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import RetentionPolicy, METRICS_BUCKET_RETENTION, TimeSeries


@dataclass
class RetentionStats:
    """Statistics from retention operations."""
    points_removed: int = 0
    series_pruned: int = 0
    bytes_reclaimed: int = 0
    last_run: int = 0


class RetentionManager:
    """
    Manages data retention policies.
    
    The manager periodically checks for expired data and:
    - Removes data points older than retention period
    - Applies downsampling rules
    - Archives or deletes expired data
    
    Example:
        >>> manager = RetentionManager(retention_enabled=True)
        >>> manager.add_policy("metrics", RetentionPolicy(name="short", duration_seconds=86400))
        >>> manager.run_cleanup(storage)
    """
    
    def __init__(
        self,
        retention_enabled: bool = True,
        check_interval: int = 3600,
        default_retention: RetentionPolicy | None = None,
    ):
        """
        Initialize retention manager.
        
        Args:
            retention_enabled: Whether retention is active
            check_interval: Seconds between cleanup runs
            default_retention: Default retention policy
        """
        self.enabled = retention_enabled
        self.check_interval = check_interval
        self.default_retention = default_retention or METRICS_BUCKET_RETENTION["15d"]
        
        self._policies: dict[str, RetentionPolicy] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        
        self._stats = RetentionStats()
        
        # Callbacks for cleanup events
        self._on_expire: list[Callable[[str, int, int], None]] = []
    
    def add_policy(self, namespace: str, policy: RetentionPolicy) -> None:
        """
        Add a retention policy.
        
        Args:
            namespace: Namespace this policy applies to (e.g., metric prefix)
            policy: Retention policy
        """
        with self._lock:
            self._policies[namespace] = policy
    
    def remove_policy(self, namespace: str) -> bool:
        """
        Remove a retention policy.
        
        Args:
            namespace: Namespace to remove
        
        Returns:
            True if policy existed
        """
        with self._lock:
            if namespace in self._policies:
                del self._policies[namespace]
                return True
            return False
    
    def get_policy(self, namespace: str) -> RetentionPolicy:
        """
        Get retention policy for a namespace.
        
        Args:
            namespace: Metric namespace
        
        Returns:
            Matching retention policy or default
        """
        with self._lock:
            # Exact match
            if namespace in self._policies:
                return self._policies[namespace]
            
            # Prefix match
            for ns, policy in self._policies.items():
                if namespace.startswith(ns):
                    return policy
            
            return self.default_retention
    
    def run_cleanup(
        self,
        storage: dict[str, TimeSeries],
        dry_run: bool = False,
    ) -> RetentionStats:
        """
        Run retention cleanup.
        
        Args:
            storage: Dictionary of time series
            dry_run: If True, don't actually delete data
        
        Returns:
            Statistics from cleanup run
        """
        stats = RetentionStats()
        current_time = int(time.time() * 1000)
        
        for key, ts in list(storage.items()):
            policy = self.get_policy(key.split("{")[0])
            cutoff = current_time - (policy.duration_seconds * 1000)
            
            # Filter expired points
            original_count = len(ts.points)
            new_points = [p for p in ts.points if p.timestamp >= cutoff]
            
            if len(new_points) < original_count:
                stats.points_removed += original_count - len(new_points)
                
                if dry_run:
                    continue
                
                # Update time series
                ts.points = new_points
                
                # Remove empty series
                if not new_points:
                    del storage[key]
                    stats.series_pruned += 1
        
        stats.last_run = current_time
        self._stats = stats
        
        return stats
    
    def apply_downsampling(
        self,
        ts: TimeSeries,
        policy: RetentionPolicy,
    ) -> TimeSeries:
        """
        Apply downsampling based on retention policy.
        
        Args:
            ts: Time series to downsample
            policy: Retention policy with resolution
        
        Returns:
            Downsampled time series
        """
        if policy.resolution == "raw":
            return ts
        
        # Get target resolution in ms
        resolution_ms = policy.resolution_ms
        
        return ts.downsampled(resolution_ms)
    
    def on_expire(self, callback: Callable[[str, int, int], None]) -> None:
        """
        Register callback for expiration events.
        
        Args:
            callback: Function(key, timestamp, bytes) called when data expires
        """
        self._on_expire.append(callback)
    
    def start(self, storage: dict[str, TimeSeries]) -> None:
        """
        Start background retention thread.
        
        Args:
            storage: Reference to storage dictionary
        """
        if not self.enabled:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(storage,),
            daemon=True,
        )
        self._thread.start()
    
    def stop(self) -> None:
        """Stop background retention thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _run_loop(self, storage: dict[str, TimeSeries]) -> None:
        """Background loop for periodic cleanup."""
        while self._running:
            try:
                self.run_cleanup(storage)
            except Exception as e:
                print(f"Retention cleanup error: {e}")
            
            time.sleep(self.check_interval)
    
    def get_stats(self) -> dict[str, Any]:
        """Get retention statistics."""
        return {
            "enabled": self.enabled,
            "check_interval_seconds": self.check_interval,
            "policies_count": len(self._policies),
            "default_retention_days": self.default_retention.duration_seconds / 86400,
            "last_cleanup": self._stats.last_run,
            "points_removed": self._stats.points_removed,
            "series_pruned": self._stats.series_pruned,
        }


class TieredStorage:
    """
    Tiered storage with hot/warm/cold data management.
    
    Moves data between storage tiers based on age and access patterns.
    """
    
    def __init__(
        self,
        hot_ttl_seconds: int = 3600,
        warm_ttl_seconds: int = 86400,
        cold_enabled: bool = True,
    ):
        """
        Initialize tiered storage.
        
        Args:
            hot_ttl_seconds: Time data stays in hot tier
            warm_ttl_seconds: Time data stays in warm tier
            cold_enabled: Whether to use cold storage
        """
        self.hot_ttl_seconds = hot_ttl_seconds
        self.warm_ttl_seconds = warm_ttl_seconds
        self.cold_enabled = cold_enabled
        
        self._hot: dict[str, TimeSeries] = {}
        self._warm: dict[str, TimeSeries] = {}
        self._cold: dict[str, TimeSeries] = {}
        
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
    
    def put(self, key: str, ts: TimeSeries) -> None:
        """Add time series to hot tier."""
        with self._lock:
            self._hot[key] = ts
    
    def get(self, key: str) -> TimeSeries | None:
        """Get time series (searches all tiers)."""
        with self._lock:
            if key in self._hot:
                return self._hot[key]
            if key in self._warm:
                return self._warm[key]
            if self._cold_enabled and key in self._cold:
                return self._cold[key]
            return None
    
    def run_tiering(self) -> dict[str, int]:
        """
        Move data between tiers based on age.
        
        Returns:
            Dictionary with tier counts
        """
        current_time = int(time.time() * 1000)
        hot_cutoff = current_time - (self.hot_ttl_seconds * 1000)
        warm_cutoff = current_time - (self.warm_ttl_seconds * 1000)
        
        with self._lock:
            # Hot -> Warm
            for key, ts in list(self._hot.items()):
                if ts.end_time and ts.end_time < hot_cutoff:
                    del self._hot[key]
                    self._warm[key] = ts
            
            # Warm -> Cold/Warm
            if self.cold_enabled:
                for key, ts in list(self._warm.items()):
                    if ts.end_time and ts.end_time < warm_cutoff:
                        del self._warm[key]
                        self._cold[key] = ts
            else:
                for key, ts in list(self._warm.items()):
                    if ts.end_time and ts.end_time < warm_cutoff:
                        del self._warm[key]
                        self._warm[key] = ts  # Keep in warm
            
            return {
                "hot": len(self._hot),
                "warm": len(self._warm),
                "cold": len(self._cold),
            }
    
    def start(self) -> None:
        """Start background tiering thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _run_loop(self) -> None:
        """Background tiering loop."""
        while self._running:
            self.run_tiering()
            time.sleep(60)  # Run every minute
    
    def get_stats(self) -> dict[str, Any]:
        """Get tiering statistics."""
        with self._lock:
            hot_points = sum(len(ts.points) for ts in self._hot.values())
            warm_points = sum(len(ts.points) for ts in self._warm.values())
            cold_points = sum(len(ts.points) for ts in self._cold.values())
            
            return {
                "hot_series": len(self._hot),
                "hot_points": hot_points,
                "warm_series": len(self._warm),
                "warm_points": warm_points,
                "cold_series": len(self._cold),
                "cold_points": cold_points,
                "hot_ttl_seconds": self.hot_ttl_seconds,
                "warm_ttl_seconds": self.warm_ttl_seconds,
                "cold_enabled": self.cold_enabled,
            }
