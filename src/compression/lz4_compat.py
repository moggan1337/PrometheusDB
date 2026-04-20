"""
LZ4 Compression Compatibility Layer.

Provides LZ4-based compression for additional data compression
on top of the Gorilla/Delta-of-Delta algorithms.
"""

from __future__ import annotations

import lz4.frame
import lz4.block
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from io import BytesIO


class LZ4Compressor:
    """
    LZ4 compression for time-series data.
    
    LZ4 provides very fast compression/decompression with good ratios,
    making it suitable for:
    - Block-level compression of time series chunks
    - Network transmission compression
    - On-disk storage compression
    
    Supports both frame (streaming) and block (random access) modes.
    """
    
    def __init__(self, compression_level: int = 9, block_size: int = 65536):
        """
        Initialize LZ4 compressor.
        
        Args:
            compression_level: LZ4 compression level (0-12, higher = more compression)
            block_size: Block size for block mode compression
        """
        self.compression_level = compression_level
        self.block_size = block_size
    
    def compress(self, data: bytes) -> bytes:
        """
        Compress data using LZ4 frame format.
        
        Frame format is suitable for streaming and provides
        better compatibility.
        
        Args:
            data: Bytes to compress
        
        Returns:
            Compressed bytes
        """
        if not data:
            return b''
        
        return lz4.frame.compress(
            data,
            compression_level=self.compression_level,
            block_size=self.block_size,
        )
    
    def decompress(self, data: bytes) -> bytes:
        """
        Decompress LZ4 frame data.
        
        Args:
            data: Compressed bytes
        
        Returns:
            Decompressed bytes
        """
        if not data:
            return b''
        
        return lz4.frame.decompress(data)
    
    def compress_block(self, data: bytes) -> bytes:
        """
        Compress data using LZ4 block format.
        
        Block format provides random access within compressed data.
        
        Args:
            data: Bytes to compress
        
        Returns:
            Compressed bytes
        """
        if not data:
            return b''
        
        return lz4.block.compress(
            data,
            mode='fast',
            compression=self.compression_level,
            store_size=False,
        )
    
    def decompress_block(self, data: bytes, uncompressed_size: int) -> bytes:
        """
        Decompress LZ4 block data.
        
        Args:
            data: Compressed bytes
            uncompressed_size: Original size of the data
        
        Returns:
            Decompressed bytes
        """
        if not data:
            return b''
        
        return lz4.block.decompress(data, uncompressed_size=uncompressed_size)
    
    def compress_stream(self, input_stream, output_stream) -> int:
        """
        Compress a stream using LZ4 frame format.
        
        Args:
            input_stream: File-like object to read from
            output_stream: File-like object to write to
        
        Returns:
            Total bytes written
        """
        total_bytes = 0
        
        with lz4.frame.LZ4FrameCompressor(
            compression_level=self.compression_level,
            block_size=self.block_size,
        ) as compressor:
            while True:
                chunk = input_stream.read(self.block_size)
                if not chunk:
                    break
                
                compressed = compressor.compress(chunk)
                output_stream.write(compressed)
                total_bytes += len(compressed)
            
            final = compressor.flush()
            output_stream.write(final)
            total_bytes += len(final)
        
        return total_bytes
    
    def decompress_stream(self, input_stream, output_stream) -> int:
        """
        Decompress a stream using LZ4 frame format.
        
        Args:
            input_stream: File-like object to read from
            output_stream: File-like object to write to
        
        Returns:
            Total bytes written
        """
        total_bytes = 0
        
        with lz4.frame.LZ4FrameDecompressor() as decompressor:
            while True:
                chunk = input_stream.read(self.block_size)
                if not chunk:
                    break
                
                decompressed = decompressor.decompress(chunk)
                output_stream.write(decompressed)
                total_bytes += len(decompressed)
        
        return total_bytes


class StreamingCompressor:
    """
    Chained compression combining Gorilla + LZ4.
    
    This provides two layers of compression:
    1. Gorilla for the time-series specific compression
    2. LZ4 for general-purpose block compression
    
    This hybrid approach often achieves 10-20% better compression
    than either method alone.
    """
    
    def __init__(self, lz4_level: int = 9):
        """
        Initialize the chained compressor.
        
        Args:
            lz4_level: LZ4 compression level
        """
        self.lz4 = LZ4Compressor(compression_level=lz4_level)
    
    def compress(self, values: list[float], timestamps: list[int] | None = None) -> bytes:
        """
        Compress values with Gorilla, then LZ4.
        
        Args:
            values: List of float values
            timestamps: Optional list of timestamps
        
        Returns:
            Double-compressed bytes
        """
        from .gorilla import GorillaCompressor
        
        # First layer: Gorilla compression
        gorilla = GorillaCompressor()
        gorilla_data = gorilla.compress(values, timestamps)
        
        # Second layer: LZ4 compression
        return self.lz4.compress(gorilla_data)
    
    def decompress(self, data: bytes) -> tuple[list[float], list[int]]:
        """
        Decompress LZ4, then Gorilla.
        
        Args:
            data: Double-compressed bytes
        
        Returns:
            Tuple of (values, timestamps)
        """
        from .gorilla import GorillaCompressor
        
        # First layer: LZ4 decompression
        gorilla_data = self.lz4.decompress(data)
        
        # Second layer: Gorilla decompression
        gorilla = GorillaCompressor()
        return gorilla.decompress(gorilla_data)
    
    def compress_to_file(
        self, 
        values: list[float], 
        timestamps: list[int] | None,
        filepath: str
    ) -> None:
        """
        Compress and write directly to a file.
        
        Args:
            values: List of float values
            timestamps: Optional list of timestamps
            filepath: Output file path
        """
        compressed = self.compress(values, timestamps)
        
        with open(filepath, 'wb') as f:
            f.write(compressed)
    
    def decompress_from_file(self, filepath: str) -> tuple[list[float], list[int]]:
        """
        Read and decompress from a file.
        
        Args:
            filepath: Input file path
        
        Returns:
            Tuple of (values, timestamps)
        """
        with open(filepath, 'rb') as f:
            data = f.read()
        
        return self.decompress(data)
