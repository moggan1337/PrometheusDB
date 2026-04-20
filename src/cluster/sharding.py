"""
Sharding Strategies for Distributed PrometheusDB.

This module provides different sharding strategies for distributing
data across cluster nodes.
"""

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar('T')


class ShardingStrategy(Enum):
    """Available sharding strategies."""
    CONSISTENT_HASH = "consistent_hash"
    KEY_RANGE = "key_range"
    ROUND_ROBIN = "round_robin"
    CUSTOM = "custom"


@dataclass
class Shard:
    """
    A shard represents a portion of data in the cluster.
    
    Attributes:
        shard_id: Unique shard identifier
        primary_node: Primary node for this shard
        replica_nodes: Replica nodes for redundancy
        key_range: Range of keys (for range-based sharding)
        metrics: Shard statistics
    """
    shard_id: str
    primary_node: str
    replica_nodes: list[str] = field(default_factory=list)
    key_start: int = 0
    key_end: int = 2 ** 32
    num_series: int = 0
    num_points: int = 0
    size_bytes: int = 0
    created_at: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shard_id": self.shard_id,
            "primary_node": self.primary_node,
            "replica_nodes": self.replica_nodes,
            "key_start": self.key_start,
            "key_end": self.key_end,
            "num_series": self.num_series,
            "num_points": self.num_points,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }


class ShardKey:
    """
    Represents a sharding key for data distribution.
    
    Shard keys are derived from metric metadata to ensure
    related data stays on the same node.
    """
    
    def __init__(self, metric_name: str, labels: dict[str, str] | None = None):
        """
        Initialize shard key.
        
        Args:
            metric_name: Name of the metric
            labels: Optional labels for more specific sharding
        """
        self.metric_name = metric_name
        self.labels = labels or {}
        
        # Build key components
        self._key = self._build_key()
        self._hash = self._compute_hash()
    
    def _build_key(self) -> str:
        """Build the shard key string."""
        parts = [self.metric_name]
        
        # Add sorted labels for consistent ordering
        for name in sorted(self.labels.keys()):
            parts.append(f"{name}={self.labels[name]}")
        
        return "|".join(parts)
    
    def _compute_hash(self) -> int:
        """Compute hash of the key."""
        hash_bytes = hashlib.md5(self._key.encode()).digest()
        return struct.unpack('>I', hash_bytes[:4])[0]
    
    @property
    def key(self) -> str:
        """Get the key string."""
        return self._key
    
    @property
    def hash(self) -> int:
        """Get the hash value."""
        return self._hash
    
    def __hash__(self) -> int:
        return self._hash
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardKey):
            return False
        return self._key == other._key


class ShardManager(ABC, Generic[T]):
    """
    Abstract base class for shard management.
    
    Provides interface for distributing data across shards
    and managing shard placement.
    """
    
    def __init__(
        self,
        num_shards: int = 256,
        replica_count: int = 3,
    ):
        """
        Initialize shard manager.
        
        Args:
            num_shards: Number of logical shards
            replica_count: Number of replicas per shard
        """
        self.num_shards = num_shards
        self.replica_count = replica_count
        self._shards: dict[str, Shard] = {}
        self._lock = None
        import threading
        self._lock = threading.RLock()
    
    @abstractmethod
    def get_shard_id(self, key: ShardKey) -> str:
        """Get shard ID for a key."""
        pass
    
    @abstractmethod
    def get_nodes_for_shard(self, shard_id: str) -> list[str]:
        """Get nodes that should store a shard."""
        pass
    
    def create_shard(self, shard_id: str, primary_node: str) -> Shard:
        """Create a new shard."""
        with self._lock:
            shard = Shard(
                shard_id=shard_id,
                primary_node=primary_node,
                replica_nodes=[],
                created_at=int(__import__('time').time() * 1000),
            )
            self._shards[shard_id] = shard
            return shard
    
    def get_shard(self, shard_id: str) -> Shard | None:
        """Get shard by ID."""
        return self._shards.get(shard_id)
    
    def get_all_shards(self) -> list[Shard]:
        """Get all shards."""
        return list(self._shards.values())
    
    def update_shard_stats(
        self,
        shard_id: str,
        num_series: int | None = None,
        num_points: int | None = None,
        size_bytes: int | None = None,
    ) -> None:
        """Update shard statistics."""
        with self._lock:
            if shard_id not in self._shards:
                return
            
            shard = self._shards[shard_id]
            if num_series is not None:
                shard.num_series = num_series
            if num_points is not None:
                shard.num_points = num_points
            if size_bytes is not None:
                shard.size_bytes = size_bytes


class ConsistentHashShardManager(ShardManager[T]):
    """
    Shard manager using consistent hashing.
    
    Uses a consistent hash ring to determine shard placement,
    with support for virtual nodes.
    """
    
    def __init__(
        self,
        num_shards: int = 256,
        replica_count: int = 3,
        vnode_count: int = 150,
    ):
        """Initialize consistent hash shard manager."""
        super().__init__(num_shards, replica_count)
        
        self.vnode_count = vnode_count
        
        # Import here to avoid circular imports
        from .consistent_hash import ConsistentHash
        self._ring = ConsistentHash(vnode_count=vnode_count)
        
        # Node to shard mapping
        self._node_shards: dict[str, set[str]] = {}
    
    def register_node(self, node_id: str) -> None:
        """Register a node in the cluster."""
        self._ring.add_node(node_id)
        self._node_shards[node_id] = set()
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from the cluster."""
        self._ring.remove_node(node_id)
        
        # Transfer shards from removed node
        if node_id in self._node_shards:
            shard_ids = self._node_shards[node_id]
            del self._node_shards[node_id]
            
            for shard_id in shard_ids:
                if shard_id in self._shards:
                    self._shards[shard_id].primary_node = ""
    
    def get_shard_id(self, key: ShardKey) -> str:
        """Get shard ID for a key using consistent hash."""
        # Map hash to shard
        shard_num = key.hash % self.num_shards
        return f"shard_{shard_num:04d}"
    
    def get_nodes_for_shard(self, shard_id: str) -> list[str]:
        """Get nodes for a shard using consistent hash."""
        return self._ring.get_nodes(shard_id, self.replica_count)
    
    def assign_shard_to_node(self, shard_id: str, node_id: str) -> None:
        """Assign a shard to a node."""
        with self._lock:
            if shard_id not in self._shards:
                self.create_shard(shard_id, node_id)
            else:
                self._shards[shard_id].primary_node = node_id
            
            if node_id not in self._node_shards:
                self._node_shards[node_id] = set()
            self._node_shards[node_id].add(shard_id)
    
    def get_shard_distribution(self) -> dict[str, int]:
        """Get number of shards per node."""
        return {
            node_id: len(shards)
            for node_id, shards in self._node_shards.items()
        }
    
    def find_shard_owner(self, shard_id: str) -> str | None:
        """Find the node that owns a shard."""
        shard = self.get_shard(shard_id)
        return shard.primary_node if shard else None


class KeyRangeShardManager(ShardManager[T]):
    """
    Range-based shard manager.
    
    Divides the key space into ranges, with each shard
    responsible for a specific range.
    """
    
    def __init__(
        self,
        num_shards: int = 256,
        replica_count: int = 3,
    ):
        """Initialize range-based shard manager."""
        super().__init__(num_shards, replica_count)
        
        # Calculate range size
        self._range_size = 2 ** 32 // num_shards
        
        # Shard metadata
        self._shard_ranges: dict[str, tuple[int, int]] = {}
        
        # Create initial shards
        for i in range(num_shards):
            shard_id = f"shard_{i:04d}"
            start = i * self._range_size
            end = start + self._range_size - 1
            self._shard_ranges[shard_id] = (start, end)
    
    def get_shard_id(self, key: ShardKey) -> str:
        """Get shard ID based on key range."""
        shard_num = key.hash // self._range_size
        return f"shard_{shard_num % self.num_shards:04d}"
    
    def get_nodes_for_shard(self, shard_id: str) -> list[str]:
        """Get nodes for a shard."""
        # For range-based, would need separate node assignment
        return []
    
    def get_shard_for_range(
        self,
        start_key: int,
        end_key: int,
    ) -> list[str]:
        """Get shards that cover a key range."""
        shards = []
        
        for shard_id, (shard_start, shard_end) in self._shard_ranges.items():
            if start_key <= shard_end and end_key >= shard_start:
                shards.append(shard_id)
        
        return shards


class RoundRobinShardManager(ShardManager[T]):
    """
    Round-robin shard manager.
    
    Distributes data evenly across shards using round-robin.
    Useful for write-heavy workloads.
    """
    
    def __init__(
        self,
        num_shards: int = 256,
        replica_count: int = 3,
    ):
        """Initialize round-robin shard manager."""
        super().__init__(num_shards, replica_count)
        
        self._current_shard = 0
        self._write_count = 0
        
        import threading
        self._lock = threading.Lock()
    
    def get_shard_id(self, key: ShardKey) -> str:
        """Get next shard in round-robin fashion."""
        with self._lock:
            shard_num = self._current_shard
            self._current_shard = (self._current_shard + 1) % self.num_shards
            self._write_count += 1
            return f"shard_{shard_num:04d}"
    
    def get_nodes_for_shard(self, shard_id: str) -> list[str]:
        """Get nodes for a shard."""
        return []


def create_shard_manager(
    strategy: ShardingStrategy,
    num_shards: int = 256,
    replica_count: int = 3,
    **kwargs,
) -> ShardManager:
    """
    Factory function to create a shard manager.
    
    Args:
        strategy: Sharding strategy to use
        num_shards: Number of shards
        replica_count: Replica count
        **kwargs: Additional strategy-specific arguments
    
    Returns:
        Configured ShardManager instance
    """
    if strategy == ShardingStrategy.CONSISTENT_HASH:
        return ConsistentHashShardManager(
            num_shards=num_shards,
            replica_count=replica_count,
            vnode_count=kwargs.get("vnode_count", 150),
        )
    elif strategy == ShardingStrategy.KEY_RANGE:
        return KeyRangeShardManager(
            num_shards=num_shards,
            replica_count=replica_count,
        )
    elif strategy == ShardingStrategy.ROUND_ROBIN:
        return RoundRobinShardManager(
            num_shards=num_shards,
            replica_count=replica_count,
        )
    else:
        raise ValueError(f"Unknown sharding strategy: {strategy}")
