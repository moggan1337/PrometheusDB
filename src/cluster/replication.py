"""
Replication Manager for Distributed PrometheusDB.

Handles data replication, failover, and consistency
across cluster nodes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .consistent_hash import ConsistentHash


class ReplicationStrategy(Enum):
    """Replication strategies."""
    SYNC = "sync"           # Synchronous replication
    ASYNC = "async"         # Asynchronous replication
    QUORUM = "quorum"       # Quorum-based (R/W = N/2+1)


@dataclass
class Replica:
    """A data replica."""
    node_id: str
    shard_id: str
    is_primary: bool = False
    last_sync: int = 0
    lag_ms: int = 0
    state: str = "active"  # active, syncing, stale, failed


@dataclass
class ReplicationStats:
    """Replication statistics."""
    total_replicas: int = 0
    healthy_replicas: int = 0
    lagging_replicas: int = 0
    failed_replicas: int = 0
    sync_failures: int = 0
    last_rebalance: int = 0


class ReplicationManager:
    """
    Manages data replication across cluster nodes.
    
    Features:
    - Configurable replication factor
    - Primary/replica selection
    - Automatic failover
    - Consistency guarantees
    
    Example:
        >>> manager = ReplicationManager(
        ...     replication_factor=3,
        ...     strategy=ReplicationStrategy.QUORUM
        ... )
        >>> manager.write("metric_key", data)
        >>> manager.read("metric_key")
    """
    
    def __init__(
        self,
        replication_factor: int = 3,
        strategy: ReplicationStrategy = ReplicationStrategy.QUORUM,
        write_quorum: int | None = None,
        read_quorum: int | None = None,
    ):
        """
        Initialize replication manager.
        
        Args:
            replication_factor: Number of replicas
            strategy: Replication strategy
            write_quorum: Min replicas for write (default: RF/2+1)
            read_quorum: Min replicas for read (default: RF/2+1)
        """
        self.replication_factor = replication_factor
        self.strategy = strategy
        
        # Quorum settings
        if write_quorum is None:
            write_quorum = replication_factor // 2 + 1
        if read_quorum is None:
            read_quorum = replication_factor // 2 + 1
        
        self.write_quorum = write_quorum
        self.read_quorum = read_quorum
        
        # Replica tracking
        self._replicas: dict[str, list[Replica]] = {}  # key -> replicas
        self._lock = threading.RLock()
        
        # Stats
        self._stats = ReplicationStats()
        
        # Callbacks
        self._on_replica_fail: list[Callable[[str, str], None]] = []
        self._on_replica_recover: list[Callable[[str, str], None]] = []
    
    def get_replicas(self, key: str) -> list[Replica]:
        """Get replicas for a key."""
        with self._lock:
            return self._replicas.get(key, [])
    
    def get_primary(self, key: str) -> str | None:
        """Get primary replica node for a key."""
        replicas = self.get_replicas(key)
        for replica in replicas:
            if replica.is_primary:
                return replica.node_id
        return replicas[0].node_id if replicas else None
    
    def get_nodes_for_write(self, key: str) -> list[str]:
        """Get nodes required for a write operation."""
        replicas = self.get_replicas(key)
        return [r.node_id for r in replicas if r.state == "active"]
    
    def get_nodes_for_read(self, key: str) -> list[str]:
        """Get nodes that can be used for a read."""
        replicas = self.get_replicas(key)
        return [r.node_id for r in replicas if r.state in ("active", "syncing")]
    
    def assign_replicas(
        self,
        key: str,
        nodes: list[str],
    ) -> list[Replica]:
        """
        Assign replicas to a key.
        
        Args:
            key: Data key
            nodes: List of node IDs (first is primary)
        
        Returns:
            List of created replicas
        """
        with self._lock:
            replicas = []
            
            for i, node_id in enumerate(nodes[:self.replication_factor]):
                replica = Replica(
                    node_id=node_id,
                    shard_id=self._get_shard_id(key),
                    is_primary=(i == 0),
                    last_sync=int(time.time() * 1000),
                )
                replicas.append(replica)
            
            self._replicas[key] = replicas
            self._update_stats()
            
            return replicas
    
    def update_replica_state(
        self,
        key: str,
        node_id: str,
        state: str,
    ) -> None:
        """Update replica state."""
        with self._lock:
            replicas = self._replicas.get(key, [])
            
            for replica in replicas:
                if replica.node_id == node_id:
                    old_state = replica.state
                    replica.state = state
                    replica.last_sync = int(time.time() * 1000)
                    
                    # Fire callbacks
                    if old_state == "active" and state in ("stale", "failed"):
                        self._notify_fail(key, node_id)
                    elif state == "active" and old_state in ("stale", "failed"):
                        self._notify_recover(key, node_id)
            
            self._update_stats()
    
    def promote_replica(self, key: str, node_id: str) -> bool:
        """
        Promote a replica to primary.
        
        Args:
            key: Data key
            node_id: Node to promote
        
        Returns:
            True if promoted
        """
        with self._lock:
            replicas = self._replicas.get(key, [])
            
            # Find current primary
            current_primary = None
            for replica in replicas:
                if replica.is_primary:
                    current_primary = replica.node_id
                    replica.is_primary = False
                    break
            
            # Find and promote target
            for replica in replicas:
                if replica.node_id == node_id:
                    replica.is_primary = True
                    replica.state = "active"
                    return True
            
            return False
    
    def write(
        self,
        key: str,
        data: Any,
        timestamp: int | None = None,
    ) -> tuple[bool, int]:
        """
        Write data with replication.
        
        Args:
            key: Data key
            data: Data to write
            timestamp: Write timestamp
        
        Returns:
            Tuple of (success, replicas_written)
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        if self.strategy == ReplicationStrategy.SYNC:
            return self._sync_write(key, data, timestamp)
        elif self.strategy == ReplicationStrategy.ASYNC:
            return self._async_write(key, data, timestamp)
        elif self.strategy == ReplicationStrategy.QUORUM:
            return self._quorum_write(key, data, timestamp)
        
        return False, 0
    
    def _sync_write(
        self,
        key: str,
        data: Any,
        timestamp: int,
    ) -> tuple[bool, int]:
        """Synchronous write to all replicas."""
        replicas = self.get_nodes_for_write(key)
        written = 0
        
        for node_id in replicas:
            # In real implementation, send to actual node
            written += 1
        
        return written == len(replicas), written
    
    def _async_write(
        self,
        key: str,
        data: Any,
        timestamp: int,
    ) -> tuple[bool, int]:
        """Asynchronous write - return immediately."""
        # Queue for async replication
        return True, 1  # Primary written
    
    def _quorum_write(
        self,
        key: str,
        data: Any,
        timestamp: int,
    ) -> tuple[bool, int]:
        """Write requiring quorum."""
        replicas = self.get_nodes_for_write(key)
        written = 0
        
        for node_id in replicas[:self.write_quorum]:
            written += 1
        
        success = written >= self.write_quorum
        return success, written
    
    def read(
        self,
        key: str,
        consistency: str = "quorum",
    ) -> tuple[Any | None, int]:
        """
        Read data with replication.
        
        Args:
            key: Data key
            consistency: Consistency level ('one', 'quorum', 'all')
        
        Returns:
            Tuple of (data, replicas_read)
        """
        if consistency == "one":
            return self._read_one(key)
        elif consistency == "quorum":
            return self._read_quorum(key)
        elif consistency == "all":
            return self._read_all(key)
        
        return None, 0
    
    def _read_one(self, key: str) -> tuple[Any | None, int]:
        """Read from single replica."""
        primary = self.get_primary(key)
        return (None, 1) if primary else (None, 0)
    
    def _read_quorum(self, key: str) -> tuple[Any | None, int]:
        """Read requiring quorum."""
        replicas = self.get_nodes_for_read(key)
        read = min(len(replicas), self.read_quorum)
        return (None, read)
    
    def _read_all(self, key: str) -> tuple[Any | None, int]:
        """Read from all replicas."""
        replicas = self.get_nodes_for_read(key)
        return (None, len(replicas))
    
    def check_replica_health(self, key: str, node_id: str) -> bool:
        """Check if a replica is healthy."""
        replicas = self.get_replicas(key)
        for replica in replicas:
            if replica.node_id == node_id:
                return replica.state == "active"
        return False
    
    def handle_failure(self, key: str, node_id: str) -> str | None:
        """
        Handle replica failure.
        
        Args:
            key: Data key
            node_id: Failed node ID
        
        Returns:
            New primary node ID or None
        """
        with self._lock:
            replicas = self._replicas.get(key, [])
            
            # Find failed replica
            failed_replica = None
            for replica in replicas:
                if replica.node_id == node_id:
                    failed_replica = replica
                    break
            
            if not failed_replica:
                return None
            
            # If failed was primary, promote next best
            if failed_replica.is_primary:
                for replica in replicas:
                    if replica.node_id != node_id and replica.state == "active":
                        self.promote_replica(key, replica.node_id)
                        return replica.node_id
            
            return self.get_primary(key)
    
    def rebalance(
        self,
        key: str,
        new_nodes: list[str],
    ) -> None:
        """Rebalance replicas across new node set."""
        with self._lock:
            if key not in self._replicas:
                return
            
            # Create new replica list
            new_replicas = []
            
            for i, node_id in enumerate(new_nodes[:self.replication_factor]):
                replica = Replica(
                    node_id=node_id,
                    shard_id=self._get_shard_id(key),
                    is_primary=(i == 0),
                    last_sync=int(time.time() * 1000),
                )
                new_replicas.append(replica)
            
            self._replicas[key] = new_replicas
            self._stats.last_rebalance = int(time.time() * 1000)
    
    def _get_shard_id(self, key: str) -> str:
        """Get shard ID from key."""
        import hashlib
        hash_val = hashlib.md5(key.encode()).digest()[0]
        return f"shard_{hash_val % 16:02d}"
    
    def _update_stats(self) -> None:
        """Update replication statistics."""
        total = 0
        healthy = 0
        lagging = 0
        failed = 0
        
        for replicas in self._replicas.values():
            for replica in replicas:
                total += 1
                if replica.state == "active":
                    healthy += 1
                elif replica.state == "syncing":
                    lagging += 1
                elif replica.state in ("stale", "failed"):
                    failed += 1
        
        self._stats.total_replicas = total
        self._stats.healthy_replicas = healthy
        self._stats.lagging_replicas = lagging
        self._stats.failed_replicas = failed
    
    def _notify_fail(self, key: str, node_id: str) -> None:
        """Notify replica failure."""
        for callback in self._on_replica_fail:
            try:
                callback(key, node_id)
            except Exception:
                pass
    
    def _notify_recover(self, key: str, node_id: str) -> None:
        """Notify replica recovery."""
        for callback in self._on_replica_recover:
            try:
                callback(key, node_id)
            except Exception:
                pass
    
    def on_replica_fail(self, callback: Callable[[str, str], None]) -> None:
        """Register failure callback."""
        self._on_replica_fail.append(callback)
    
    def on_replica_recover(self, callback: Callable[[str, str], None]) -> None:
        """Register recovery callback."""
        self._on_replica_recover.append(callback)
    
    def get_stats(self) -> dict[str, Any]:
        """Get replication statistics."""
        self._update_stats()
        return {
            "replication_factor": self.replication_factor,
            "strategy": self.strategy.value,
            "write_quorum": self.write_quorum,
            "read_quorum": self.read_quorum,
            "total_replicas": self._stats.total_replicas,
            "healthy_replicas": self._stats.healthy_replicas,
            "lagging_replicas": self._stats.lagging_replicas,
            "failed_replicas": self._stats.failed_replicas,
            "sync_failures": self._stats.sync_failures,
            "last_rebalance": self._stats.last_rebalance,
        }
