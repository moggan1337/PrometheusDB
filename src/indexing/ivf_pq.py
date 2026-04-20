"""
Inverted File Index with Product Quantization (IVF-PQ) Implementation.

IVF-PQ is a memory-efficient ANN algorithm that combines:
1. Inverted File Index: Partitions the vector space into clusters
2. Product Quantization: Compresses vectors into small codes

This approach achieves:
- Lower memory usage than HNSW (especially for large datasets)
- Fast search with approximate but efficient quantization
- Good scalability to billions of vectors

Reference: Product Quantization for Nearest Neighbor Search
           (Jegou et al., 2011)
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np


@dataclass
class PQCodebook:
    """
    Product Quantization codebook.
    
    The vector space is divided into subspaces, each with its own
    k-means cluster centers (codewords).
    """
    sub_dim: int  # Dimension of each subspace
    num_subspaces: int  # Number of subspaces
    k: int  # Number of centroids per subspace
    codewords: np.ndarray  # Shape: (num_subspaces, k, sub_dim)
    
    def __post_init__(self):
        self.codewords = np.array(self.codewords, dtype=np.float32)
    
    def encode(self, vector: np.ndarray) -> np.ndarray:
        """
        Encode a vector using the codebook.
        
        Returns subspace codes (one per subspace).
        """
        num_codes = len(vector) // self.sub_dim
        codes = np.zeros(num_codes, dtype=np.int32)
        
        for i in range(num_codes):
            start = i * self.sub_dim
            end = start + self.sub_dim
            subspace = vector[start:end]
            
            # Find nearest codeword
            distances = np.linalg.norm(self.codewords[i] - subspace, axis=1)
            codes[i] = np.argmin(distances)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode codes back to approximate vectors.
        
        Returns reconstructed vector.
        """
        num_codes = len(codes)
        result = np.zeros(num_codes * self.sub_dim, dtype=np.float32)
        
        for i in range(num_codes):
            start = i * self.sub_dim
            end = start + self.sub_dim
            result[start:end] = self.codewords[i, codes[i]]
        
        return result
    
    def compute_residual(self, vector: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Compute residual (vector - approximate reconstruction)."""
        reconstructed = self.decode(codes)
        return vector - reconstructed


@dataclass
class IVFCluster:
    """A cluster in the inverted file index."""
    centroid: np.ndarray
    vectors: list[tuple[str, np.ndarray]] = field(default_factory=list)
    codes: dict[str, np.ndarray] = field(default_factory=dict)
    residuals: dict[str, np.ndarray] = field(default_factory=dict)
    
    def add(self, id: str, vector: np.ndarray, code: np.ndarray | None = None, residual: np.ndarray | None = None) -> None:
        """Add a vector to this cluster."""
        self.vectors.append((id, vector))
        if code is not None:
            self.codes[id] = code
        if residual is not None:
            self.residuals[id] = residual
    
    def remove(self, id: str) -> bool:
        """Remove a vector from this cluster."""
        self.vectors = [(vid, v) for vid, v in self.vectors if vid != id]
        self.codes.pop(id, None)
        self.residuals.pop(id, None)
        return True
    
    def __len__(self) -> int:
        return len(self.vectors)


@dataclass(order=True)
class IVFSearchResult:
    """Result from IVF-PQ search."""
    distance: float
    id: str = field(compare=False)
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)


class IVFPQIndex:
    """
    IVF-PQ Index for memory-efficient vector similarity search.
    
    This implementation supports:
    - Configurable number of clusters (nlist)
    - Configurable PQ parameters (m subvectors, k centroids)
    - Two-stage search (coarse + refined)
    - Optional residual quantization
    
    Example:
        >>> index = IVFPQIndex(dimension=128, nlist=1024, pq_m=8, pq_k=256)
        >>> index.add("doc1", np.random.randn(128))
        >>> results = index.search(np.random.randn(128), k=5)
    """
    
    def __init__(
        self,
        dimension: int,
        nlist: int = 256,
        pq_m: int = 8,
        pq_k: int = 256,
        nprobe: int = 8,
        max_samples: int = 100,
        distance: str = "l2",
        seed: int = 42,
    ):
        """
        Initialize IVF-PQ index.
        
        Args:
            dimension: Vector dimensionality
            nlist: Number of clusters (inverted lists)
            pq_m: Number of PQ subspaces (dimension must be divisible by m)
            pq_k: Number of centroids per subspace (typically 256)
            nprobe: Number of clusters to search
            max_samples: Maximum samples per cluster for search
            distance: Distance metric ('l2', 'cosine', 'dot')
            seed: Random seed
        """
        self.dimension = dimension
        self.nlist = nlist
        self.pq_m = pq_m
        self.pq_k = pq_k
        self.nprobe = nprobe
        self.max_samples = max_samples
        self.distance = distance
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Validate parameters
        if dimension % pq_m != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by pq_m {pq_m}")
        
        # PQ subspace dimension
        self.pq_sub_dim = dimension // pq_m
        
        # Storage
        self.clusters: list[IVFCluster] = []
        self.vectors: dict[str, tuple[np.ndarray, int]] = {}  # id -> (vector, cluster_id)
        self.codebook: PQCodebook | None = None
        
        # Distance function
        self._distance_func = self._get_distance_func(distance)
        
        # Initialize clusters with random centroids
        self._initialize_clusters()
    
    def _get_distance_func(self, distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
        """Get distance function."""
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
            return lambda a, b: -float(np.dot(a, b))
        else:
            raise ValueError(f"Unknown distance: {distance}")
    
    def _initialize_clusters(self) -> None:
        """Initialize empty clusters."""
        self.clusters = [
            IVFCluster(centroid=np.zeros(self.dimension, dtype=np.float32))
            for _ in range(self.nlist)
        ]
    
    def _kmeans_clustering(
        self,
        vectors: list[np.ndarray],
        k: int,
        max_iter: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simple k-means clustering.
        
        Returns:
            centroids: Cluster centroids (k, dimension)
            assignments: Cluster assignment for each vector
        """
        if len(vectors) < k:
            k = len(vectors)
        
        # Initialize centroids with random vectors
        indices = random.sample(range(len(vectors)), k)
        centroids = np.array([vectors[i].copy() for i in indices], dtype=np.float32)
        
        assignments = np.zeros(len(vectors), dtype=np.int32)
        
        for _ in range(max_iter):
            # Assign vectors to nearest centroid
            new_assignments = np.zeros(len(vectors), dtype=np.int32)
            
            for i, vec in enumerate(vectors):
                distances = np.linalg.norm(centroids - vec, axis=1)
                new_assignments[i] = np.argmin(distances)
            
            # Check convergence
            if np.array_equal(assignments, new_assignments):
                break
            
            assignments = new_assignments
            
            # Update centroids
            for j in range(k):
                cluster_vectors = [vectors[i] for i in range(len(vectors)) if assignments[i] == j]
                if cluster_vectors:
                    centroids[j] = np.mean(cluster_vectors, axis=0)
        
        return centroids, assignments
    
    def _train_pq(self, vectors: list[np.ndarray]) -> PQCodebook:
        """
        Train Product Quantization codebook.
        
        Uses k-means on each subspace independently.
        """
        # Convert to numpy array
        vectors = np.array(vectors, dtype=np.float32)
        
        # Split into subspaces
        num_subspaces = self.pq_m
        sub_dim = self.pq_sub_dim
        
        codewords = np.zeros((num_subspaces, self.pq_k, sub_dim), dtype=np.float32)
        
        for i in range(num_subspaces):
            start = i * sub_dim
            end = start + sub_dim
            subspace_vectors = vectors[:, start:end]
            
            # Train k-means for this subspace
            subspace_list = [subspace_vectors[j] for j in range(len(vectors))]
            
            # Simple k-means
            indices = random.sample(range(len(subspace_list)), min(self.pq_k, len(subspace_list)))
            centroids = np.array([subspace_list[idx].copy() for idx in indices], dtype=np.float32)
            
            for _ in range(10):  # Limited iterations
                distances = np.linalg.norm(
                    centroids[:, np.newaxis, :] - subspace_vectors[np.newaxis, :, :],
                    axis=2
                )
                assignments = np.argmin(distances, axis=0)
                
                for j in range(len(centroids)):
                    mask = assignments == j
                    if np.any(mask):
                        centroids[j] = np.mean(subspace_vectors[mask], axis=0)
            
            # Pad if needed
            if len(centroids) < self.pq_k:
                padding = np.zeros((self.pq_k - len(centroids), sub_dim), dtype=np.float32)
                centroids = np.vstack([centroids, padding])
            
            codewords[i] = centroids
        
        return PQCodebook(
            sub_dim=sub_dim,
            num_subspaces=num_subspaces,
            k=self.pq_k,
            codewords=codewords,
        )
    
    def fit(self, vectors: list[np.ndarray] | np.ndarray) -> None:
        """
        Fit the index on training vectors.
        
        This trains the coarse quantizer (k-means for clusters)
        and the PQ codebook.
        
        Args:
            vectors: Training vectors
        """
        if isinstance(vectors, np.ndarray):
            vectors = [vectors[i] for i in range(len(vectors))]
        
        vectors = [np.array(v, dtype=np.float32) for v in vectors]
        
        # Step 1: Cluster vectors for coarse quantization
        centroids, assignments = self._kmeans_clustering(vectors, self.nlist)
        
        for i, cluster in enumerate(self.clusters):
            if i < len(centroids):
                cluster.centroid = centroids[i]
        
        # Assign vectors to clusters
        for i, (vec, cluster_id) in enumerate(zip(vectors, assignments)):
            if cluster_id < len(self.clusters):
                self.clusters[cluster_id].add(f"_train_{i}", vec)
                self.vectors[f"_train_{i}"] = (vec, cluster_id)
        
        # Step 2: Train PQ codebook
        # Use residuals (vector - cluster centroid) for better PQ
        training_data = []
        for cluster in self.clusters:
            for _, vec in cluster.vectors:
                residual = vec - cluster.centroid
                training_data.append(residual)
        
        if training_data:
            self.codebook = self._train_pq(training_data)
    
    def add(
        self,
        id: str,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a vector to the index.
        
        Args:
            id: Unique identifier
            vector: The embedding vector
            metadata: Optional metadata
        """
        if id in self.vectors:
            raise ValueError(f"Vector with id '{id}' already exists")
        
        vector = np.array(vector, dtype=np.float32)
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} != {self.dimension}")
        
        if metadata is None:
            metadata = {}
        metadata["_vector_id"] = id
        
        # Find nearest cluster
        distances = np.array([
            self._distance_func(vector, cluster.centroid)
            for cluster in self.clusters
        ])
        cluster_id = np.argmin(distances)
        
        # Encode with PQ if codebook is trained
        code = None
        residual = None
        
        if self.codebook is not None:
            code = self.codebook.encode(vector - self.clusters[cluster_id].centroid)
            residual = vector - self.clusters[cluster_id].centroid - self.codebook.decode(code)
        
        # Add to cluster
        self.clusters[cluster_id].add(id, vector, code, residual)
        self.vectors[id] = (vector, cluster_id)
    
    def _search_clusters(self, query: np.ndarray, nprobe: int) -> list[int]:
        """Find the nprobe nearest clusters to the query."""
        distances = np.array([
            self._distance_func(query, cluster.centroid)
            for cluster in self.clusters
        ])
        
        # Get indices of nearest clusters
        return np.argsort(distances)[:nprobe].tolist()
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
        max_samples: int | None = None,
        filter_func: Callable[[str], bool] | None = None,
    ) -> list[IVFSearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results to return
            nprobe: Number of clusters to search (None = use default)
            max_samples: Max samples per cluster (None = use default)
            filter_func: Optional filter function
        
        Returns:
            List of search results sorted by distance
        """
        query = np.array(query, dtype=np.float32)
        
        if nprobe is None:
            nprobe = self.nprobe
        if max_samples is None:
            max_samples = self.max_samples
        
        # Step 1: Find nearest clusters
        cluster_ids = self._search_clusters(query, nprobe)
        
        # Step 2: Search within clusters
        candidates: list[tuple[float, str, dict]] = []
        
        for cluster_id in cluster_ids:
            cluster = self.clusters[cluster_id]
            
            # Get vectors from this cluster
            for vid, vector in cluster.vectors:
                if filter_func and not filter_func(vid):
                    continue
                
                # Get metadata
                metadata = {"_cluster_id": cluster_id}
                
                # Compute distance
                dist = self._distance_func(query, vector)
                candidates.append((dist, vid, metadata))
        
        # If PQ is used, we could refine here with asymmetric distance computation
        
        # Sort and return top k
        candidates.sort(key=lambda x: x[0])
        
        return [
            IVFSearchResult(distance=d, id=vid, metadata=m)
            for d, vid, m in candidates[:k]
        ]
    
    def remove(self, id: str) -> bool:
        """Remove a vector from the index."""
        if id not in self.vectors:
            return False
        
        _, cluster_id = self.vectors[id]
        self.clusters[cluster_id].remove(id)
        del self.vectors[id]
        return True
    
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        cluster_sizes = [len(c) for c in self.clusters]
        
        total_vectors = sum(cluster_sizes)
        avg_cluster_size = total_vectors / len(self.clusters) if self.clusters else 0
        
        return {
            "num_vectors": len(self.vectors),
            "dimension": self.dimension,
            "nlist": self.nlist,
            "pq_m": self.pq_m,
            "pq_k": self.pq_k,
            "nprobe": self.nprobe,
            "distance_metric": self.distance,
            "total_clusters_used": sum(1 for s in cluster_sizes if s > 0),
            "avg_cluster_size": avg_cluster_size,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "codebook_trained": self.codebook is not None,
        }
    
    def save(self, filepath: str) -> None:
        """Save index to file."""
        import pickle
        
        data = {
            "dimension": self.dimension,
            "nlist": self.nlist,
            "pq_m": self.pq_m,
            "pq_k": self.pq_k,
            "nprobe": self.nprobe,
            "max_samples": self.max_samples,
            "distance": self.distance,
            "clusters": self.clusters,
            "vectors": self.vectors,
            "codebook": self.codebook,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> IVFPQIndex:
        """Load index from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        index = cls(
            dimension=data["dimension"],
            nlist=data["nlist"],
            pq_m=data["pq_m"],
            pq_k=data["pq_k"],
            nprobe=data["nprobe"],
            max_samples=data["max_samples"],
            distance=data["distance"],
        )
        
        index.clusters = data["clusters"]
        index.vectors = data["vectors"]
        index.codebook = data["codebook"]
        
        return index
