"""
Zstandard (Zstd) Compression Compatibility Layer.

Zstd provides excellent compression ratios with good speed,
making it ideal for archival storage and network compression.
"""

from __future__ import annotations

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    zstd = None

from typing import BinaryIO


class ZstdCompressor:
    """
    Zstandard compression for time-series data.
    
    Zstd offers:
    - Excellent compression ratios (comparable to LZMA)
    - Very fast decompression (faster than LZ4)
    - Reasonable compression speed
    - Dictionary training for better compression of similar data
    - Streaming support
    
    Best used for:
    - Long-term archival storage
    - Data transfer between nodes
    - Snapshots and backups
    """
    
    def __init__(
        self,
        compression_level: int = 3,
        window_log: int = 0,
        dict_path: str | None = None,
    ):
        """
        Initialize Zstd compressor.
        
        Args:
            compression_level: Compression level (-7 to 22, default 3)
            window_log: Maximum window log (0 = default)
            dict_path: Path to pre-trained dictionary (optional)
        """
        if not ZSTD_AVAILABLE:
            raise ImportError(
                "zstandard package not installed. "
                "Install with: pip install zstandard"
            )
        
        self.compression_level = compression_level
        self.window_log = window_log
        self._dictionary = None
        
        if dict_path:
            self._load_dictionary(dict_path)
    
    def _load_dictionary(self, path: str) -> None:
        """Load a compression dictionary from file."""
        with open(path, 'rb') as f:
            dict_data = f.read()
        self._dictionary = zstd.ZstdCompressionDict(dict_data)
    
    def compress(self, data: bytes) -> bytes:
        """
        Compress data using Zstd.
        
        Args:
            data: Bytes to compress
        
        Returns:
            Compressed bytes
        """
        if not data:
            return b''
        
        ctx = zstd.ZstdCompressor(
            level=self.compression_level,
            window_log=self.window_log,
        )
        
        if self._dictionary:
            ctx = zstd.ZstdCompressor(
                level=self.compression_level,
                window_log=self.window_log,
                dict=self._dictionary,
            )
        
        return ctx.compress(data)
    
    def decompress(self, data: bytes) -> bytes:
        """
        Decompress Zstd data.
        
        Args:
            data: Compressed bytes
        
        Returns:
            Decompressed bytes
        """
        if not data:
            return b''
        
        ctx = zstd.ZstdDecompressor()
        
        if self._dictionary:
            ctx = zstd.ZstdDecompressor(dict_data=self._dictionary)
        
        return ctx.decompress(data)
    
    def compress_to_file(self, data: bytes, filepath: str) -> int:
        """
        Compress and write to file with size header.
        
        Args:
            data: Bytes to compress
            filepath: Output file path
        
        Returns:
            Number of bytes written
        """
        compressed = self.compress(data)
        
        with open(filepath, 'wb') as f:
            # Write size header for verification
            f.write(len(data).to_bytes(8, 'big'))
            f.write(compressed)
        
        return len(data).bit_length() // 8 + len(compressed)
    
    def decompress_from_file(self, filepath: str) -> bytes:
        """
        Read and decompress from file.
        
        Args:
            filepath: Input file path
        
        Returns:
            Decompressed bytes
        """
        with open(filepath, 'rb') as f:
            # Read size header
            expected_size = int.from_bytes(f.read(8), 'big')
            
            compressed = f.read()
        
        decompressed = self.decompress(compressed)
        
        if len(decompressed) != expected_size:
            raise ValueError(
                f"Decompressed size mismatch: expected {expected_size}, "
                f"got {len(decompressed)}"
            )
        
        return decompressed
    
    def streaming_compress(self, input_stream: BinaryIO, output_stream: BinaryIO) -> int:
        """
        Stream compression from input to output.
        
        Args:
            input_stream: Input file-like object
            output_stream: Output file-like object
        
        Returns:
            Total compressed bytes written
        """
        ctx = zstd.ZstdCompressor(level=self.compression_level)
        
        total_bytes = 0
        
        while True:
            chunk = input_stream.read(65536)
            if not chunk:
                break
            
            compressed = ctx.compress(chunk)
            output_stream.write(compressed)
            total_bytes += len(compressed)
        
        final = ctx.flush(zstd.FLUSH_FRAME)
        output_stream.write(final)
        total_bytes += len(final)
        
        return total_bytes
    
    def streaming_decompress(self, input_stream: BinaryIO, output_stream: BinaryIO) -> int:
        """
        Stream decompression from input to output.
        
        Args:
            input_stream: Input file-like object
            output_stream: Output file-like object
        
        Returns:
            Total decompressed bytes written
        """
        ctx = zstd.ZstdDecompressor()
        reader = ctx.stream_reader(input_stream)
        
        total_bytes = 0
        
        while True:
            chunk = reader.read(65536)
            if not chunk:
                break
            
            output_stream.write(chunk)
            total_bytes += len(chunk)
        
        return total_bytes
    
    @classmethod
    def train_dictionary(cls, samples: list[bytes], dict_size: int = 112640) -> bytes:
        """
        Train a compression dictionary on sample data.
        
        Dictionary training improves compression ratios when
        compressing many similar small chunks.
        
        Args:
            samples: List of sample data chunks
            dict_size: Target dictionary size in bytes
        
        Returns:
            Trained dictionary bytes
        """
        if not ZSTD_AVAILABLE:
            raise ImportError("zstandard package not installed")
        
        dict_data = zstd.train_dictionary(dict_size, samples)
        return bytes(dict_data)
    
    @classmethod
    def create_compressor_with_dict(cls, dict_data: bytes, compression_level: int = 3):
        """
        Create a compressor with a pre-trained dictionary.
        
        Args:
            dict_data: Dictionary bytes
            compression_level: Compression level
        
        Returns:
            ZstdCompressor instance with dictionary
        """
        compressor = cls(compression_level=compression_level)
        compressor._dictionary = zstd.ZstdCompressionDict(dict_data)
        return compressor


class HybridCompressor:
    """
    Hybrid compression combining Gorilla + Zstd for maximum compression.
    
    This provides excellent compression for time-series data:
    1. Gorilla exploits time-series specific patterns
    2. Zstd provides additional general-purpose compression
    
    Compression ratio is typically 2-4x better than either method alone.
    """
    
    def __init__(self, zstd_level: int = 3):
        """
        Initialize hybrid compressor.
        
        Args:
            zstd_level: Zstd compression level
        """
        self.zstd = ZstdCompressor(compression_level=zstd_level)
    
    def compress(self, values: list[float], timestamps: list[int] | None = None) -> bytes:
        """
        Compress values with Gorilla, then Zstd.
        
        Args:
            values: List of float values
            timestamps: Optional list of timestamps
        
        Returns:
            Double-compressed bytes
        """
        from .gorilla import GorillaCompressor
        
        gorilla = GorillaCompressor()
        gorilla_data = gorilla.compress(values, timestamps)
        
        return self.zstd.compress(gorilla_data)
    
    def decompress(self, data: bytes) -> tuple[list[float], list[int]]:
        """
        Decompress Zstd, then Gorilla.
        
        Args:
            data: Double-compressed bytes
        
        Returns:
            Tuple of (values, timestamps)
        """
        from .gorilla import GorillaCompressor
        
        gorilla_data = self.zstd.decompress(data)
        
        gorilla = GorillaCompressor()
        return gorilla.decompress(gorilla_data)
