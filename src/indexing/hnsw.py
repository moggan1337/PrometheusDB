"""
Hierarchical Navigable Small World (HNSW) Index Implementation.

HNSW is a graph-based approximate nearest neighbor (ANN) algorithm that
achieves excellent search performance with high recall. It works by:

1. Building a multi-layer graph where higher layers have fewer connections
2. Starting search from entry points at the top layer
3. Greedily traversing connections to find nearest neighbors
4. Gradually descending to lower layers for finer search

Reference: Efficient and robust approximate nearest neighbor search
           using Hierarchical Navigable Small World graphs (Malkov et al., 2018)
"""

from __future__ import annotations

import math
import random
import heapq
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np


@dataclass
class HNSWNode:
    """
    A node in the HNSW graph.
    
    Attributes:
        id: Unique identifier for the node
        vector: The embedding vector
        neighbors: List of neighbor lists per layer
    """
    id: str
    vector: np.ndarray
    neighbors: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)


@dataclass(order=True)
class SearchResult:
    """Result of an ANN search with distance."""
    distance: float
    id: str = field(compare=False)
    vector: np.ndarray = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=False, compare=False)


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) index for vector similarity search.
    
    This implementation supports:
    - L2 (Euclidean) distance
    - Cosine similarity (via normalized vectors)
    - Configurable M (connections per layer)
    - Configurable ef (search width)
    - Incremental insertion
    
    Example:
        >>> index = HNSWIndex(dimension=128, M=16, ef_construction=200)
        >>> index.add("doc1", np.random.randn(128))
        >>> index.add("doc2", np.random.randn(128))
        >>> results = index.search(np.random.randn(128), k=5)
    """
    
    def __init__(
        self,
        dimension: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_layers: int | None = None,
        distance: str = "l2",
        seed: int = 42,
    ):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Vector dimensionality
            M: Number of connections per layer (higher = better recall, more memory)
            ef_construction: Search width during construction (higher = better quality, slower)
            ef_search: Search width during search (higher = better recall, slower)
            max_layers: Maximum number of layers (default: log(N))
            distance: Distance metric ('l2', 'cosine', 'dot')
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.distance = distance
        self.seed = seed
        
        # Calculate max layers if not specified
        self.max_layers = max_layers or int(math.log2(100000))  # Default to ~17 for 100k vectors
        
        # Node storage
        self.nodes: dict[str, HNSWNode] = {}
        self.entry_point: str | None = None
        self.max_level = 0
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Distance function
        self._distance_func = self._get_distance_func(distance)
    
    def _get_distance_func(self, distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
        """Get the distance function for the specified metric."""
        if distance == "l2":
            return lambda a, b: float(np.linalg.norm(a - b))
        elif distance == "cosine":
            def cosine_dist(a, b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 1.0
                return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))
            return cosine_dist
        elif distance == "dot":
            return lambda a, b: -float(np.dot(a, b))  # Negative because we want min distance
        else:
            raise ValueError(f"Unknown distance metric: {distance}")
    
    def _calculate_level(self) -> int:
        """
        Calculate the level for a new node using exponential distribution.
        
        Higher levels are exponentially less probable.
        """
        # Level = floor(-ln(uniform(0,1)) * level_mult)
        # where level_mult controls the probability
        level_mult = 1.0 / math.log(self.M)
        
        while True:
            r = -math.log(random.random())
            level = int(r * level_mult)
            if level < self.max_layers:
                return level
            # Retry with small probability to avoid too many top-level nodes
    
    def _get_layer(self, level: int) -> list[str]:
        """Get neighbors at a specific level."""
        return []
    
    def add(self, id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """
        Add a vector to the index.
        
        Args:
            id: Unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata dictionary
        """
        if id in self.nodes:
            raise ValueError(f"Vector with id '{id}' already exists")
        
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match "
                f"index dimension {self.dimension}"
            )
        
        vector = np.array(vector, dtype=np.float32)
        if metadata is None:
            metadata = {}
        
        # Create node
        node = HNSWNode(
            id=id,
            vector=vector,
            neighbors=[],
            metadata=metadata,
        )
        
        # Determine node level
        level = self._calculate_level()
        
        # Initialize neighbor lists for all layers up to max level
        for _ in range(self.max_layers):
            node.neighbors.append([])
        
        # If this is the first node
        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            self.nodes[id] = node
            return
        
        # Search for insertion point starting from top layer
        current_level = self.max_level
        candidates = [(0, self.entry_point)]
        visited = {self.entry_point}
        
        # Search from top layer down to find closest neighbors
        while current_level > level:
            found = True
            while found:
                found = False
                current_id = candidates[0][1]
                current_node = self.nodes[current_id]
                
                if current_level >= len(current_node.neighbors):
                    continue
                
                # Get neighbors at current level
                for neighbor_id in current_node.neighbors[current_level]:
                    if neighbor_id in visited:
                        continue
                    
                    visited.add(neighbor_id)
                    dist = self._distance_func(vector, self.nodes[neighbor_id].vector)
                    
                    # Insert into candidates heap (by distance)
                    heapq.heappush(candidates, (dist, neighbor_id))
                
                # Check if we should continue searching
                if len(candidates) > 1 and candidates[0][0] < candidates[-1][0]:
                    # Continue searching
                    found = True
                else:
                    break
            
            # Move to next level
            while candidates and candidates[0][1] != self._get_nearest(candidates, vector):
                heapq.heappop(candidates)
            
            current_level -= 1
        
        # Insert at current level and below
        while current_level >= 0:
            # Get ef_construction nearest neighbors
            neighbors = self._search_layer(
                vector, self.ef_construction, current_level, visited
            )
            
            # Select M nearest neighbors
            selected = self._select_neighbors(neighbors, self.M)
            
            # Add connections
            for neighbor_id in selected:
                if level <= current_level:
                    # Add bidirectional connections
                    node.neighbors[current_level].append(neighbor_id)
                    self.nodes[neighbor_id].neighbors[current_level].append(id)
            
            current_level -= 1
        
        # Update max level if needed
        if level > self.max_level:
            self.max_level = level
        
        self.nodes[id] = node
    
    def _get_nearest(self, candidates: list[tuple[float, str]], vector: np.ndarray) -> str | None:
        """Get the nearest candidate to the vector."""
        if not candidates:
            return None
        
        nearest_id = candidates[0][1]
        nearest_dist = candidates[0][0]
        
        for dist, cid in candidates:
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = cid
        
        return nearest_id
    
    def _search_layer(
        self,
        query: np.ndarray,
        ef: int,
        level: int,
        visited: set[str] | None = None,
        start_id: str | None = None,
    ) -> list[tuple[float, str]]:
        """
        Search a single layer of the HNSW graph.
        
        Args:
            query: Query vector
            ef: Number of candidates to maintain
            level: Layer to search
            visited: Set of visited node IDs
            start_id: Starting node ID (uses entry point if None)
        
        Returns:
            List of (distance, node_id) tuples
        """
        if visited is None:
            visited = set()
        
        if start_id is None:
            start_id = self.entry_point
            if start_id is None:
                return []
        
        visited.add(start_id)
        
        # Priority queue of (distance, node_id) for candidates
        candidates = [(self._distance_func(query, self.nodes[start_id].vector), start_id)]
        
        # Priority queue of (distance, node_id) for results
        results = candidates.copy()
        heapq.heapify(results)
        
        # Dynamic candidate list for traversal
        frontier = [(self._distance_func(query, self.nodes[start_id].vector), start_id)]
        
        while frontier:
            # Get current candidate with maximum distance in results
            if len(results) >= ef:
                max_dist = -results[0][0]  # Negate because heapq is min-heap
            else:
                max_dist = float('inf')
            
            # Pop from frontier (sorted by distance to query)
            current_dist, current_id = heapq.heappop(frontier)
            
            # Prune if distance is already worse than worst result
            if current_dist > max_dist and len(results) >= ef:
                break
            
            # Get neighbors
            current_node = self.nodes[current_id]
            if level >= len(current_node.neighbors):
                continue
            
            for neighbor_id in current_node.neighbors[level]:
                if neighbor_id in visited:
                    continue
                
                visited.add(neighbor_id)
                dist = self._distance_func(query, self.nodes[neighbor_id].vector)
                
                # Add to results if good enough
                if len(results) < ef or dist < -results[0][0]:
                    heapq.heappush(results, (-dist, neighbor_id))
                    heapq.heappush(frontier, (dist, neighbor_id))
        
        # Return top ef results (negate distances back)
        return [(-d, i) for d, i in results][:ef]
    
    def _select_neighbors(
        self,
        candidates: list[tuple[float, str]],
        m: int,
    ) -> list[str]:
        """
        Select M nearest neighbors from candidates using heuristic.
        
        Uses a simplified version of the select neighbors heuristic
        that avoids adding redundant connections.
        """
        if len(candidates) <= m:
            return [cid for _, cid in candidates]
        
        # Sort by distance
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        
        selected = []
        selected_ids = set()
        
        for dist, cid in sorted_candidates:
            if cid in selected_ids:
                continue
            
            # Check distance to already selected neighbors
            too_close = False
            for sel_id in selected_ids:
                sel_node = self.nodes[sel_id]
                d = self._distance_func(sel_node.vector, self.nodes[cid].vector)
                if d < dist:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(cid)
                selected_ids.add(cid)
            
            if len(selected) >= m:
                break
        
        return selected
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int | None = None,
        filter_func: Callable[[str], bool] | None = None,
    ) -> list[SearchResult]:
        """
        Search for k nearest neighbors to the query vector.
        
        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Search width (uses ef_search if None)
            filter_func: Optional function to filter candidate IDs
        
        Returns:
            List of SearchResult objects sorted by distance
        """
        if ef is None:
            ef = self.ef_search
        
        if self.entry_point is None:
            return []
        
        query = np.array(query, dtype=np.float32)
        if len(query) != self.dimension:
            raise ValueError(f"Query dimension {len(query)} does not match {self.dimension}")
        
        visited = set()
        
        # Start from top layer
        current_level = self.max_level
        
        # Search down to level 1
        while current_level > 0:
            neighbors = self._search_layer(query, ef, current_level, visited)
            if neighbors:
                # Get nearest entry point for next level
                current_id = neighbors[0][1]
                current_dist = neighbors[0][0]
            current_level -= 1
        
        # Final search at level 0
        neighbors = self._search_layer(query, ef, 0, visited)
        
        # Apply filter if provided
        if filter_func:
            neighbors = [(d, cid) for d, cid in neighbors if filter_func(cid)]
        
        # Sort and return top k
        neighbors.sort(key=lambda x: x[0])
        
        results = []
        for dist, cid in neighbors[:k]:
            node = self.nodes[cid]
            results.append(SearchResult(
                distance=dist,
                id=cid,
                vector=node.vector,
                metadata=node.metadata,
            ))
        
        return results
    
    def remove(self, id: str) -> bool:
        """
        Remove a vector from the index.
        
        Note: This is a lazy deletion - the node is removed from
        neighbor lists but not fully cleaned up.
        
        Args:
            id: Node ID to remove
        
        Returns:
            True if removed, False if not found
        """
        if id not in self.nodes:
            return False
        
        node = self.nodes[id]
        
        # Remove from neighbor lists
        for level in range(len(node.neighbors)):
            neighbors = self.nodes
            for neighbor_id in node.neighbors[level]:
                if neighbor_id in self.nodes:
                    neighbor = self.nodes[neighbor_id]
                    if level < len(neighbor.neighbors):
                        if id in neighbor.neighbors[level]:
                            neighbor.neighbors[level].remove(id)
        
        del self.nodes[id]
        
        # Update entry point if necessary
        if self.entry_point == id:
            if self.nodes:
                self.entry_point = next(iter(self.nodes.keys()))
            else:
                self.entry_point = None
                self.max_level = 0
        
        return True
    
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        total_connections = 0
        for node in self.nodes.values():
            for neighbors in node.neighbors:
                total_connections += len(neighbors)
        
        return {
            "num_vectors": len(self.nodes),
            "dimension": self.dimension,
            "max_level": self.max_level,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "distance_metric": self.distance,
            "total_connections": total_connections,
            "avg_connections_per_node": total_connections / len(self.nodes) if self.nodes else 0,
        }
    
    def save(self, filepath: str) -> None:
        """Save index to file."""
        import pickle
        
        data = {
            "nodes": self.nodes,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "dimension": self.dimension,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "distance": self.distance,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> HNSWIndex:
        """Load index from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        index = cls(
            dimension=data["dimension"],
            M=data["M"],
            ef_construction=data["ef_construction"],
            ef_search=data["ef_search"],
            distance=data["distance"],
        )
        
        index.nodes = data["nodes"]
        index.entry_point = data["entry_point"]
        index.max_level = data["max_level"]
        
        return index
