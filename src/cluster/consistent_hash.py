"""
Consistent Hashing Implementation.

Consistent hashing is used to distribute data across cluster nodes
with minimal redistribution when nodes join or leave.
"""

from __future__ import annotations

import bisect
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np


@dataclass
class VirtualNode:
    """A virtual node in the consistent hash ring."""
    node_id: str
    vnode_id: int
    position: int
    
    @property
    def key(self) -> str:
        """Get the key for this virtual node."""
        return f"{self.node_id}:{self.vnode_id}"


class ConsistentHash:
    """
    Consistent Hashing with virtual nodes.
    
    Features:
    - Virtual nodes for better distribution
    - Configurable number of virtual nodes per physical node
    - Efficient lookup O(log N)
    - Support for replica placement
    
    Example:
        >>> ch = ConsistentHash(vnode_count=150)
        >>> ch.add_node("node1")
        >>> ch.add_node("node2")
        >>> node = ch.get_node("metric_key")
    """
    
    def __init__(
        self,
        vnode_count: int = 150,
        hash_function: Callable[[str], int] | None = None,
    ):
        """
        Initialize consistent hash ring.
        
        Args:
            vnode_count: Number of virtual nodes per physical node
            hash_function: Custom hash function (default: MD5)
        """
        self.vnode_count = vnode_count
        self._hash_func = hash_function or self._default_hash
        
        # Ring: sorted list of (position, virtual_node)
        self._ring: list[tuple[int, VirtualNode]] = []
        self._positions: dict[str, list[int]] = {}  # node_id -> positions
        
        # Lock for thread safety
        import threading
        self._lock = threading.Lock()
    
    @staticmethod
    def _default_hash(key: str) -> int:
        """Default hash function using MD5."""
        hash_bytes = hashlib.md5(key.encode()).digest()
        return struct.unpack('>I', hash_bytes[:4])[0]
    
    def _hash(self, key: str) -> int:
        """Hash a key."""
        return self._hash_func(key)
    
    def add_node(self, node_id: str) -> None:
        """
        Add a node to the ring.
        
        Args:
            node_id: Unique node identifier
        """
        with self._lock:
            if node_id in self._positions:
                return
            
            positions = []
            
            for i in range(self.vnode_count):
                vnode = VirtualNode(
                    node_id=node_id,
                    vnode_id=i,
                    position=self._hash(f"{node_id}:{i}"),
                )
                self._ring.append((vnode.position, vnode))
                positions.append(vnode.position)
            
            # Sort ring by position
            self._ring.sort(key=lambda x: x[0])
            self._positions[node_id] = positions
    
    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the ring.
        
        Args:
            node_id: Node to remove
        """
        with self._lock:
            if node_id not in self._positions:
                return
            
            positions = self._positions[node_id]
            
            # Remove all virtual nodes
            self._ring = [
                (pos, vnode) for pos, vnode in self._ring
                if vnode.node_id != node_id
            ]
            
            del self._positions[node_id]
    
    def get_node(self, key: str) -> str | None:
        """
        Get the primary node for a key.
        
        Args:
            key: Data key
        
        Returns:
            Node ID or None if ring is empty
        """
        if not self._ring:
            return None
        
        position = self._hash(key)
        return self._get_node_at_position(position)
    
    def _get_node_at_position(self, position: int) -> str | None:
        """Get node at or after a position."""
        with self._lock:
            if not self._ring:
                return None
            
            # Binary search for position
            positions = [p for p, _ in self._ring]
            
            idx = bisect.bisect_right(positions, position)
            
            if idx >= len(positions):
                # Wrap around to first
                return self._ring[0][1].node_id
            
            return self._ring[idx][1].node_id
    
    def get_nodes(self, key: str, count: int = 1) -> list[str]:
        """
        Get multiple nodes for a key (for replication).
        
        Args:
            key: Data key
            count: Number of nodes to return
        
        Returns:
            List of node IDs
        """
        if not self._ring:
            return []
        
        result = []
        position = self._hash(key)
        seen_nodes = set()
        
        with self._lock:
            positions = [p for p, _ in self._ring]
            
            idx = bisect.bisect_right(positions, position)
            
            while len(result) < count and len(result) < len(self._positions):
                if idx >= len(positions):
                    idx = 0
                
                node_id = self._ring[idx][1].node_id
                
                if node_id not in seen_nodes:
                    result.append(node_id)
                    seen_nodes.add(node_id)
                
                idx += 1
        
        return result
    
    def get_replica_positions(
        self,
        key: str,
        replica_count: int = 3,
    ) -> list[tuple[str, int]]:
        """
        Get replica positions for a key.
        
        Returns list of (node_id, position) tuples for placing replicas.
        """
        if not self._ring:
            return []
        
        result = []
        base_position = self._hash(key)
        
        with self._lock:
            positions = [p for p, _ in self._ring]
            
            for i in range(replica_count):
                # Add offset for each replica
                offset = (base_position + i * 1000) % (2 ** 32)
                idx = bisect.bisect_right(positions, offset)
                
                if idx >= len(positions):
                    idx = 0
                
                vnode = self._ring[idx][1]
                result.append((vnode.node_id, offset))
        
        return result
    
    def get_physical_node_count(self) -> int:
        """Get number of physical nodes."""
        return len(self._positions)
    
    def get_vnode_count(self) -> int:
        """Get total number of virtual nodes."""
        return len(self._ring)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the ring."""
        with self._lock:
            if not self._ring:
                return {
                    "physical_nodes": 0,
                    "virtual_nodes": 0,
                    "vnode_per_node": self.vnode_count,
                    "distribution": {},
                }
            
            # Calculate distribution
            distribution = {}
            for node_id in self._positions:
                distribution[node_id] = len([
                    vnode for _, vnode in self._ring
                    if vnode.node_id == node_id
                ])
            
            return {
                "physical_nodes": len(self._positions),
                "virtual_nodes": len(self._ring),
                "vnode_per_node": self.vnode_count,
                "distribution": distribution,
                "ring_size": len(self._ring),
            }
    
    def balance(self) -> dict[str, float]:
        """
        Calculate load balance across nodes.
        
        Returns:
            Dictionary of node_id -> load percentage
        """
        with self._lock:
            if not self._positions:
                return {}
            
            total_vnodes = len(self._ring)
            if total_vnodes == 0:
                return {}
            
            balance = {}
            for node_id, positions in self._positions.items():
                balance[node_id] = len(positions) / total_vnodes * 100
            
            return balance
    
    def find_divergent_node(
        self,
        key: str,
        primary_node: str,
    ) -> list[str]:
        """
        Find nodes that should have a copy of a key.
        
        Used for verifying replication consistency.
        """
        all_nodes = self.get_nodes(key, len(self._positions))
        return [n for n in all_nodes if n != primary_node]
    
    def serialize(self) -> dict[str, Any]:
        """Serialize ring state."""
        with self._lock:
            return {
                "vnode_count": self.vnode_count,
                "nodes": {
                    node_id: positions
                    for node_id, positions in self._positions.items()
                },
            }
    
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ConsistentHash:
        """Deserialize ring state."""
        ch = cls(vnode_count=data["vnode_count"])
        
        for node_id, positions in data["nodes"].items():
            for vnode_id, position in enumerate(positions):
                vnode = VirtualNode(
                    node_id=node_id,
                    vnode_id=vnode_id,
                    position=position,
                )
                ch._ring.append((position, vnode))
        
        ch._ring.sort(key=lambda x: x[0])
        
        for node_id in data["nodes"]:
            ch._positions[node_id] = data["nodes"][node_id]
        
        return ch


class HashRing:
    """
    Alternative hash ring implementation using numpy for performance.
    
    Optimized for high-throughput scenarios with many keys.
    """
    
    def __init__(self, vnode_count: int = 150):
        """Initialize hash ring."""
        self.vnode_count = vnode_count
        self._ring_positions: np.ndarray = np.array([], dtype=np.uint32)
        self._ring_nodes: list[str] = []
        self._node_vnodes: dict[str, np.ndarray] = {}
        
        import threading
        self._lock = threading.Lock()
    
    def add_node(self, node_id: str) -> None:
        """Add node to ring."""
        with self._lock:
            # Generate vnode positions
            positions = np.array([
                self._hash(f"{node_id}:{i}")
                for i in range(self.vnode_count)
            ], dtype=np.uint32)
            
            self._node_vnodes[node_id] = positions
            
            # Rebuild ring
            self._rebuild_ring()
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from ring."""
        with self._lock:
            if node_id in self._node_vnodes:
                del self._node_vnodes[node_id]
                self._rebuild_ring()
    
    def _rebuild_ring(self) -> None:
        """Rebuild sorted ring arrays."""
        nodes = []
        positions = []
        
        for node_id, vnode_positions in self._node_vnodes.items():
            for pos in vnode_positions:
                nodes.append(node_id)
                positions.append(pos)
        
        if positions:
            sorted_idx = np.argsort(positions)
            self._ring_positions = np.array(positions, dtype=np.uint32)[sorted_idx]
            self._ring_nodes = [nodes[i] for i in sorted_idx]
        else:
            self._ring_positions = np.array([], dtype=np.uint32)
            self._ring_nodes = []
    
    def _hash(self, key: str) -> int:
        """Hash a key."""
        hash_bytes = hashlib.md5(key.encode()).digest()
        return struct.unpack('>I', hash_bytes[:4])[0]
    
    def get_node(self, key: str) -> str | None:
        """Get primary node for key."""
        if len(self._ring_positions) == 0:
            return None
        
        position = self._hash(key)
        idx = np.searchsorted(self._ring_positions, position)
        
        if idx >= len(self._ring_positions):
            return self._ring_nodes[0]
        
        return self._ring_nodes[idx]
    
    def get_nodes(self, key: str, count: int = 3) -> list[str]:
        """Get multiple nodes for replication."""
        if len(self._ring_positions) == 0:
            return []
        
        position = self._hash(key)
        idx = np.searchsorted(self._ring_positions, position)
        
        result = []
        seen = set()
        
        for i in range(len(self._ring_nodes)):
            actual_idx = (idx + i) % len(self._ring_nodes)
            node = self._ring_nodes[actual_idx]
            
            if node not in seen:
                result.append(node)
                seen.add(node)
            
            if len(result) >= count:
                break
        
        return result
