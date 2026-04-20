"""
Cluster Node - Individual node in a distributed PrometheusDB cluster.

This module provides the ClusterNode class that represents a single
node in the cluster, handling local storage, coordination with
other nodes, and participating in consistent hashing.
"""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid

from ..storage.schema import Metric, TimeSeries, DataPoint, Label


class NodeState(Enum):
    """State of a cluster node."""
    JOINING = "joining"
    ACTIVE = "active"
    LEAVING = "leaving"
    FAILED = "failed"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    port: int
    vnode_count: int = 150  # Virtual nodes for consistent hashing
    state: NodeState = NodeState.JOINING
    version: str = "0.1.0"
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    last_seen: int = field(default_factory=lambda: int(time.time() * 1000))
    num_series: int = 0
    num_points: int = 0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "vnode_count": self.vnode_count,
            "state": self.state.value,
            "version": self.version,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "num_series": self.num_series,
            "num_points": self.num_points,
            "cpu_usage": self.cpu_usage,
            "memory_usage_mb": self.memory_usage_mb,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeInfo:
        """Create from dictionary."""
        return cls(
            node_id=d["node_id"],
            host=d["host"],
            port=d["port"],
            vnode_count=d.get("vnode_count", 150),
            state=NodeState(d.get("state", "active")),
            version=d.get("version", "0.1.0"),
            created_at=d.get("created_at", int(time.time() * 1000)),
            last_seen=d.get("last_seen", int(time.time() * 1000)),
            num_series=d.get("num_series", 0),
            num_points=d.get("num_points", 0),
            cpu_usage=d.get("cpu_usage", 0.0),
            memory_usage_mb=d.get("memory_usage_mb", 0.0),
        )
    
    @property
    def address(self) -> str:
        """Get address string."""
        return f"{self.host}:{self.port}"


class ClusterNode:
    """
    A node in the distributed PrometheusDB cluster.
    
    Each node:
    - Manages its own local storage
    - Participates in consistent hashing for data distribution
    - Handles replication and failover
    - Communicates with other nodes
    
    Example:
        >>> node = ClusterNode(host="192.168.1.10", port=9090)
        >>> node.start()
        >>> node.join_cluster(["192.168.1.1:9090"])  # Seed nodes
    """
    
    def __init__(
        self,
        host: str | None = None,
        port: int = 9090,
        node_id: str | None = None,
        vnode_count: int = 150,
        data_dir: str = "./data",
    ):
        """
        Initialize cluster node.
        
        Args:
            host: Hostname/IP (default: auto-detect)
            port: Port number
            node_id: Unique node ID (auto-generated if None)
            vnode_count: Number of virtual nodes for consistent hashing
            data_dir: Local data directory
        """
        self.node_id = node_id or self._generate_node_id()
        self.host = host or self._get_default_host()
        self.port = port
        self.vnode_count = vnode_count
        self.data_dir = data_dir
        
        # Node info
        self.info = NodeInfo(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            vnode_count=vnode_count,
        )
        
        # Cluster state
        self.state = NodeState.JOINING
        self.peers: dict[str, NodeInfo] = {}
        self._lock = threading.RLock()
        
        # Local storage (placeholder - would integrate with PrometheusDB)
        self._local_storage: dict[str, TimeSeries] = {}
        
        # Communication
        self._socket: socket.socket | None = None
        self._running = False
        
        # Failure detection
        self._failure_detector = FailureDetector(self)
        self._gossip_interval = 1.0  # seconds
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        node_uuid = uuid.uuid4().bytes
        host_hash = hashlib.sha256(f"{self.host}:{self.port}".encode()).digest()[:4]
        return node_uuid.hex() + host_hash.hex()
    
    def _get_default_host(self) -> str:
        """Get default host address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host = s.getsockname()[0]
            s.close()
            return host
        except Exception:
            return "127.0.0.1"
    
    def start(self) -> None:
        """Start the cluster node."""
        self._running = True
        self.state = NodeState.ACTIVE
        self.info.state = NodeState.ACTIVE
        
        # Start gossip protocol
        threading.Thread(target=self._gossip_loop, daemon=True).start()
        
        # Start failure detector
        self._failure_detector.start()
    
    def stop(self) -> None:
        """Stop the cluster node gracefully."""
        self._running = False
        self.state = NodeState.LEAVING
        self.info.state = NodeState.LEAVING
        
        # Notify peers of departure
        self._broadcast_leave()
        
        if self._socket:
            self._socket.close()
        
        self._failure_detector.stop()
    
    def join_cluster(self, seed_nodes: list[str]) -> bool:
        """
        Join an existing cluster.
        
        Args:
            seed_nodes: List of seed node addresses
        
        Returns:
            True if successfully joined
        """
        if not seed_nodes:
            # First node in cluster
            self.state = NodeState.ACTIVE
            return True
        
        # Try to contact seed nodes
        for seed in seed_nodes:
            try:
                host, port = seed.split(":")
                port = int(port)
                
                # Send join request
                if self._send_join_request(host, port):
                    self.state = NodeState.ACTIVE
                    return True
            except Exception as e:
                print(f"Failed to contact seed node {seed}: {e}")
        
        return False
    
    def _send_join_request(self, host: str, port: int) -> bool:
        """Send join request to a seed node."""
        # Simplified - would use actual network communication
        return True
    
    def _broadcast_leave(self) -> None:
        """Broadcast leave message to all peers."""
        # Simplified - would send actual message
        pass
    
    def _gossip_loop(self) -> None:
        """Gossip protocol loop for peer discovery."""
        while self._running:
            try:
                # Update node info
                self.info.last_seen = int(time.time() * 1000)
                self._update_stats()
                
                # Gossip with random peer
                self._gossip()
                
                time.sleep(self._gossip_interval)
            except Exception as e:
                print(f"Gossip error: {e}")
    
    def _gossip(self) -> None:
        """Exchange information with a random peer."""
        if not self.peers:
            return
        
        # Select random peer
        import random
        peer_id = random.choice(list(self.peers.keys()))
        peer = self.peers[peer_id]
        
        # Build gossip message
        message = {
            "type": "gossip",
            "from": self.node_id,
            "info": self.info.to_dict(),
            "timestamp": int(time.time() * 1000),
        }
        
        # Send to peer (simplified)
        # In reality, would use network communication
    
    def _update_stats(self) -> None:
        """Update node statistics."""
        import psutil
        try:
            process = psutil.Process()
            self.info.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.info.cpu_usage = process.cpu_percent()
        except Exception:
            pass
        
        self.info.num_series = len(self._local_storage)
        self.info.num_points = sum(len(ts.points) for ts in self._local_storage.values())
    
    def add_peer(self, peer_info: NodeInfo) -> None:
        """Add a peer node."""
        with self._lock:
            self.peers[peer_info.node_id] = peer_info
    
    def remove_peer(self, node_id: str) -> None:
        """Remove a peer node."""
        with self._lock:
            if node_id in self.peers:
                del self.peers[node_id]
    
    def get_preferred_node(self, key: str, replica: int = 0) -> str | None:
        """
        Get the preferred node for a key.
        
        Args:
            key: Data key (e.g., metric fingerprint)
            replica: Replica number (0 = primary)
        
        Returns:
            Node ID of preferred node or None
        """
        with self._lock:
            if not self.peers:
                return self.node_id if self.state == NodeState.ACTIVE else None
            
            # Get consistent hash ring positions
            positions = self._get_ring_positions(key, replica)
            
            for pos in positions:
                for peer_id, peer in self.peers.items():
                    if peer.state == NodeState.ACTIVE:
                        peer_pos = self._hash_node(peer.node_id)
                        if peer_pos >= pos:
                            return peer_id
            
            # Wrap around - return first active node
            for peer_id, peer in self.peers.items():
                if peer.state == NodeState.ACTIVE:
                    return peer_id
            
            return self.node_id if self.state == NodeState.ACTIVE else None
    
    def _get_ring_positions(self, key: str, replica: int) -> list[int]:
        """Get positions on the hash ring for a key."""
        base_hash = self._hash_key(key)
        ring_size = 2 ** 32
        
        # Return position for this replica
        return [(base_hash + replica * (ring_size // 100)) % ring_size]
    
    def _hash_key(self, key: str) -> int:
        """Hash a key to a ring position."""
        hash_bytes = hashlib.md5(key.encode()).digest()
        return struct.unpack('>I', hash_bytes[:4])[0]
    
    def _hash_node(self, node_id: str) -> int:
        """Hash a node ID to a ring position."""
        return self._hash_key(node_id)
    
    def local_write(
        self,
        metric: Metric,
        timestamp: int,
        value: float,
    ) -> bool:
        """
        Write data to local storage.
        
        Args:
            metric: Metric to write
            timestamp: Timestamp
            value: Value
        
        Returns:
            True if successful
        """
        with self._lock:
            key = metric.metric_key
            
            if key not in self._local_storage:
                self._local_storage[key] = TimeSeries(metric=metric)
            
            self._local_storage[key] = self._local_storage[key].append(timestamp, value)
            return True
    
    def local_read(
        self,
        metric_key: str,
        start: int | None = None,
        end: int | None = None,
    ) -> TimeSeries | None:
        """
        Read data from local storage.
        
        Args:
            metric_key: Metric key
            start: Start timestamp
            end: End timestamp
        
        Returns:
            TimeSeries or None
        """
        with self._lock:
            if metric_key not in self._local_storage:
                return None
            
            ts = self._local_storage[metric_key]
            
            if start is not None or end is not None:
                ts = ts.range_query(start or 0, end or int(time.time() * 1000))
            
            return ts
    
    def get_stats(self) -> dict[str, Any]:
        """Get node statistics."""
        with self._lock:
            return {
                "node_id": self.node_id,
                "address": self.info.address,
                "state": self.state.value,
                "peers": len(self.peers),
                "local_series": len(self._local_storage),
                "local_points": sum(len(ts.points) for ts in self._local_storage.values()),
                "info": self.info.to_dict(),
            }


class FailureDetector:
    """
    Phi Accrual Failure Detector.
    
    Detects node failures based on past heartbeat intervals
    and adjusts sensitivity dynamically.
    """
    
    def __init__(self, node: ClusterNode, threshold: float = 8.0):
        """
        Initialize failure detector.
        
        Args:
            node: Cluster node
            threshold: Failure threshold (higher = more tolerant)
        """
        self.node = node
        self.threshold = threshold
        
        # Heartbeat history
        self._heartbeats: dict[str, list[float]] = {}
        self._last_heartbeat: dict[str, float] = {}
        
        self._running = False
        self._lock = threading.Lock()
    
    def record_heartbeat(self, node_id: str) -> None:
        """Record a heartbeat from a node."""
        with self._lock:
            now = time.time()
            
            if node_id not in self._heartbeats:
                self._heartbeats[node_id] = []
            
            if node_id in self._last_heartbeat:
                interval = now - self._last_heartbeat[node_id]
                self._heartbeats[node_id].append(interval)
                
                # Keep last 100 intervals
                if len(self._heartbeats[node_id]) > 100:
                    self._heartbeats[node_id] = self._heartbeats[node_id][-100:]
            
            self._last_heartbeat[node_id] = now
    
    def phi(self, node_id: str) -> float:
        """
        Calculate phi (suspicion level) for a node.
        
        Returns:
            Phi value (higher = more likely failed)
        """
        with self._lock:
            if node_id not in self._heartbeats:
                return 0.0
            
            now = time.time()
            last = self._last_heartbeat.get(node_id, now)
            elapsed = now - last
            
            intervals = self._heartbeats[node_id]
            if not intervals:
                return 0.0
            
            mean = sum(intervals) / len(intervals)
            variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
            stddev = variance ** 0.5 if variance > 0 else 1.0
            
            # Phi calculation (simplified)
            import math
            phi = (elapsed - mean) / stddev if stddev > 0 else elapsed - mean
            
            return max(0.0, phi)
    
    def is_available(self, node_id: str) -> bool:
        """Check if node is considered available."""
        return self.phi(node_id) < self.threshold
    
    def start(self) -> None:
        """Start failure detector."""
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()
    
    def stop(self) -> None:
        """Stop failure detector."""
        self._running = False
    
    def _run(self) -> None:
        """Run failure detection loop."""
        while self._running:
            with self._lock:
                now = time.time()
                
                for peer_id in list(self.node.peers.keys()):
                    phi = self.phi(peer_id)
                    
                    if phi >= self.threshold:
                        # Mark peer as failed
                        self.node.peers[peer_id].state = NodeState.FAILED
            
            time.sleep(1.0)
