"""
ANN Registry - Factory for Approximate Nearest Neighbor Indexes.

This module provides a unified interface for creating and managing
different ANN index types, with automatic selection based on
dataset characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .hnsw import HNSWIndex
from .ivf_pq import IVFPQIndex
from .hybrid_index import HybridIndex


class ANNIndex(Protocol):
    """Protocol defining the ANN index interface."""
    
    def add(self, id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None: ...
    def search(self, query: np.ndarray, k: int, **kwargs) -> list[Any]: ...
    def remove(self, id: str) -> bool: ...
    def get_stats(self) -> dict[str, Any]: ...


@dataclass
class ANNConfig:
    """Configuration for ANN index selection."""
    index_type: str = "auto"  # 'auto', 'hnsw', 'ivf_pq', 'hybrid'
    dimension: int = 128
    num_vectors_estimate: int = 100000
    memory_limit_mb: int = 1024
    latency_priority: str = "balanced"  # 'recall', 'balanced', 'speed'
    distance: str = "cosine"
    
    # HNSW parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    
    # IVF-PQ parameters
    ivf_nlist: int = 256
    ivf_nprobe: int = 8
    pq_m: int = 8
    pq_k: int = 256


class ANNRegistry:
    """
    Registry for creating and managing ANN indexes.
    
    This registry provides:
    - Automatic index type selection based on dataset characteristics
    - Unified interface for different index types
    - Index caching and reuse
    - Performance optimization hints
    
    Example:
        >>> registry = ANNRegistry()
        >>> index = registry.create_index(dimension=128, num_vectors=1000000)
        >>> registry.add("doc1", np.random.randn(128))
        >>> results = registry.search(np.random.randn(128), k=10)
    """
    
    # Index type recommendations based on dataset size
    INDEX_THRESHOLDS = {
        "small": 10000,       # < 10k vectors
        "medium": 100000,     # 10k - 100k
        "large": 1000000,     # 100k - 1M
        "xlarge": 10000000,   # 1M - 10M
        "huge": 100000000,    # > 10M
    }
    
    # Latency presets
    LATENCY_PRESETS = {
        "recall": {"hnsw_ef": 200, "ivf_nprobe": 32, "strategy": "high_recall"},
        "balanced": {"hnsw_ef": 50, "ivf_nprobe": 8, "strategy": "balanced"},
        "speed": {"hnsw_ef": 20, "ivf_nprobe": 4, "strategy": "high_speed"},
    }
    
    def __init__(self):
        """Initialize the registry."""
        self._indexes: dict[str, ANNIndex] = {}
        self._default_index: ANNIndex | None = None
    
    def _estimate_memory(self, config: ANNConfig) -> dict[str, float]:
        """
        Estimate memory usage for different index types.
        
        Returns:
            Dictionary of index_type -> estimated memory in MB
        """
        n = config.num_vectors_estimate
        d = config.dimension
        
        estimates = {}
        
        # HNSW memory: ~16-32 bytes per connection + vector storage
        hnsw_connections = config.hnsw_m * 2  # Bidirectional
        hnsw_memory = (d * 4 + hnsw_connections * 8 + 64) * n / (1024 * 1024)
        estimates["hnsw"] = hnsw_memory
        
        # IVF-PQ memory: PQ codes + centroids
        pq_bits = config.pq_m * config.pq_k.bit_length()  # Bits for PQ code
        ivf_memory = (pq_bits / 8 + d * 4 / 10) * n / (1024 * 1024)  # 10% of original for centroids
        estimates["ivf_pq"] = ivf_memory
        
        # Hybrid
        estimates["hybrid"] = estimates["hnsw"] * 0.7 + estimates["ivf_pq"] * 0.3
        
        return estimates
    
    def _select_index_type(self, config: ANNConfig) -> str:
        """
        Automatically select the best index type.
        
        Selection criteria:
        - Dataset size
        - Memory constraints
        - Latency requirements
        - Recall requirements
        """
        if config.index_type != "auto":
            return config.index_type
        
        memory_estimates = self._estimate_memory(config)
        
        # Check memory constraints
        for index_type, memory in sorted(memory_estimates.items(), key=lambda x: x[1]):
            if memory <= config.memory_limit_mb:
                # Apply latency preferences
                preset = self.LATENCY_PRESETS.get(config.latency_priority, {})
                strategy = preset.get("strategy", "balanced")
                
                if strategy == "high_speed":
                    # Prefer IVF-PQ for speed
                    return "ivf_pq"
                elif strategy == "high_recall":
                    # Prefer HNSW for recall
                    return "hnsw"
                else:
                    # Balanced: use size-based selection
                    if config.num_vectors_estimate < self.INDEX_THRESHOLDS["medium"]:
                        return "hnsw"
                    else:
                        return "ivf_pq"
        
        # Default to most memory-efficient
        return min(memory_estimates, key=memory_estimates.get)
    
    def create_index(self, config: ANNConfig | None = None, **kwargs) -> ANNIndex:
        """
        Create an ANN index based on configuration.
        
        Args:
            config: ANNConfig object or None
            **kwargs: Alternative configuration parameters
        
        Returns:
            Configured ANN index
        """
        if config is None:
            config = ANNConfig(**kwargs)
        elif isinstance(config, dict):
            config = ANNConfig(**config)
        
        index_type = self._select_index_type(config)
        
        # Apply latency presets
        if config.latency_priority != "balanced":
            preset = self.LATENCY_PRESETS.get(config.latency_priority, {})
            if "hnsw_ef" in preset:
                config.hnsw_ef_search = preset["hnsw_ef"]
            if "ivf_nprobe" in preset:
                config.ivf_nprobe = preset["ivf_nprobe"]
        
        # Create index
        if index_type == "hnsw":
            index = HNSWIndex(
                dimension=config.dimension,
                M=config.hnsw_m,
                ef_construction=config.hnsw_ef_construction,
                ef_search=config.hnsw_ef_search,
                distance=config.distance,
            )
        elif index_type == "ivf_pq":
            index = IVFPQIndex(
                dimension=config.dimension,
                nlist=config.ivf_nlist,
                pq_m=config.pq_m,
                pq_k=config.pq_k,
                nprobe=config.ivf_nprobe,
                distance=config.distance,
            )
        elif index_type == "hybrid":
            index = HybridIndex(
                dimension=config.dimension,
                index_type="hnsw",
                hnsw_m=config.hnsw_m,
                hnsw_ef=config.hnsw_ef_search,
                distance=config.distance,
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self._default_index = index
        return index
    
    def add(self, id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Add a vector to the default index."""
        if self._default_index is None:
            raise ValueError("No index created. Call create_index first.")
        self._default_index.add(id, vector, metadata)
    
    def search(self, query: np.ndarray, k: int = 10, **kwargs) -> list[Any]:
        """Search the default index."""
        if self._default_index is None:
            raise ValueError("No index created. Call create_index first.")
        return self._default_index.search(query, k, **kwargs)
    
    def remove(self, id: str) -> bool:
        """Remove a vector from the default index."""
        if self._default_index is None:
            raise ValueError("No index created. Call create_index first.")
        return self._default_index.remove(id)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics from the default index."""
        if self._default_index is None:
            return {}
        return self._default_index.get_stats()
    
    @classmethod
    def benchmark(
        cls,
        dimension: int,
        num_vectors: int,
        query_vectors: np.ndarray,
        true_neighbors: list[list[int]],
        k: int = 10,
    ) -> dict[str, Any]:
        """
        Benchmark different index types.
        
        Args:
            dimension: Vector dimension
            num_vectors: Number of vectors to index
            query_vectors: Query vectors
            true_neighbors: Ground truth neighbors for recall calculation
            k: Number of neighbors to retrieve
        
        Returns:
            Benchmark results for each index type
        """
        import time
        
        results = {}
        
        for index_type in ["hnsw", "ivf_pq"]:
            try:
                if index_type == "hnsw":
                    index = HNSWIndex(dimension=dimension)
                else:
                    index = IVFPQIndex(dimension=dimension)
                
                # Add vectors
                start = time.time()
                for i in range(num_vectors):
                    vector = np.random.randn(dimension).astype(np.float32)
                    index.add(f"vec_{i}", vector)
                insert_time = time.time() - start
                
                # Search
                start = time.time()
                search_results = []
                for qv in query_vectors:
                    res = index.search(qv, k=k)
                    search_results.append([r.id for r in res])
                search_time = time.time() - start
                
                # Calculate recall
                recall = 0.0
                for pred, truth in zip(search_results, true_neighbors):
                    hits = len(set(pred) & set(truth))
                    recall += hits / k
                recall /= len(query_vectors)
                
                results[index_type] = {
                    "insert_time_s": insert_time,
                    "search_time_s": search_time,
                    "avg_search_time_ms": (search_time / len(query_vectors)) * 1000,
                    "recall": recall,
                    "throughput": num_vectors / insert_time,
                }
            except Exception as e:
                results[index_type] = {"error": str(e)}
        
        return results


# Global registry instance
_default_registry: ANNRegistry | None = None


def get_registry() -> ANNRegistry:
    """Get the global registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ANNRegistry()
    return _default_registry


def create_index(**kwargs) -> ANNIndex:
    """Convenience function to create an index from the global registry."""
    return get_registry().create_index(**kwargs)


def add_vector(id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
    """Convenience function to add a vector to the global registry."""
    get_registry().add(id, vector, metadata)


def search_vectors(query: np.ndarray, k: int = 10, **kwargs) -> list[Any]:
    """Convenience function to search the global registry."""
    return get_registry().search(query, k, **kwargs)
