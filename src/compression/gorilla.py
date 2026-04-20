"""
Gorilla Compression Algorithm Implementation.

Gorilla is a lossless compression algorithm designed specifically for
time-series data. It achieves excellent compression ratios for metrics
that exhibit temporal patterns by exploiting:

1. XOR-based value encoding: Similar consecutive values have similar binary representations
2. Delta encoding: Timestamps are stored as deltas from the previous
3. Variable-length encoding: Uses the minimum bits needed for each value

Reference: Gorilla: A Fast, Scalable, In-Memory Time Series Database
           (Dubey et al., 2015)
"""

from __future__ import annotations

import struct
import math
from typing import BinaryIO


class GorillaCompressor:
    """
    Implements the Gorilla compression algorithm for time-series data.
    
    Gorilla compression works by:
    1. Storing the first value as-is (64-bit float)
    2. For subsequent values:
       - XOR with the previous value
       - If XOR is 0, store a single '0' bit
       - If XOR fits in 64 bits, store '10' + value
       - If XOR fits in fewer bits at a later position, store '110' + position + bits
    
    Attributes:
        block_size: Size of compression blocks in bits (default 64)
    """
    
    BLOCK_SIZE = 64  # bits
    
    def __init__(self):
        self._buffer = 0
        self._bits_in_buffer = 0
        self._previous_value: float | None = None
        self._previous_timestamp: int | None = None
        self._previous_timestamp_delta: int | None = None
        
    def reset(self) -> None:
        """Reset the compressor state."""
        self._buffer = 0
        self._bits_in_buffer = 0
        self._previous_value = None
        self._previous_timestamp = None
        self._previous_timestamp_delta = None
    
    def _write_bits(self, value: int, num_bits: int) -> None:
        """Write bits to the internal buffer."""
        self._buffer = (self._buffer << num_bits) | (value & ((1 << num_bits) - 1))
        self._bits_in_buffer += num_bits
        
        while self._bits_in_buffer >= 8:
            self._bits_in_buffer -= 8
            yield (self._buffer >> self._bits_in_buffer) & 0xFF
    
    def _write_control(self, control: int) -> None:
        """Write a control bit."""
        yield from self._write_bits(control, 1)
    
    def _float_to_bits(self, value: float) -> int:
        """Convert float to its IEEE 754 bit representation."""
        return struct.unpack('<Q', struct.pack('<d', value))[0]
    
    def _bits_to_float(self, bits: int) -> float:
        """Convert IEEE 754 bits back to float."""
        return struct.unpack('<d', struct.pack('<Q', bits))[0]
    
    def _count_leading_zeros(self, value: int) -> int:
        """Count leading zeros in a 64-bit integer."""
        if value == 0:
            return 64
        return value.bit_length()
    
    def _count_trailing_zeros(self, value: int) -> int:
        """Count trailing zeros in a 64-bit integer."""
        if value == 0:
            return 64
        return (value & -value).bit_length() - 1
    
    def compress_value(self, value: float) -> list[int]:
        """
        Compress a single float value using Gorilla encoding.
        
        Returns list of compressed bytes.
        """
        output = []
        self.reset()
        
        if self._previous_value is None:
            # First value: store as-is with control '0'
            bits = self._float_to_bits(value)
            output.extend(self._write_control(0))
            output.extend(self._write_bits(bits, 64))
        else:
            xor = self._float_to_bits(value) ^ self._float_to_bits(self._previous_value)
            
            if xor == 0:
                # Same value: control '1' + '0'
                output.extend(self._write_control(1))
                output.extend(self._write_control(0))
            else:
                leading = self._count_leading_zeros(xor)
                trailing = self._count_trailing_zeros(xor)
                significant = 64 - leading - trailing
                
                if leading >= 16 or significant > 21:
                    # Cannot compress efficiently: control '1' + '01' + full value
                    output.extend(self._write_control(1))
                    output.extend(self._write_control(1))
                    output.extend(self._write_bits(0, 1))  # marker
                    bits = self._float_to_bits(value)
                    output.extend(self._write_bits(bits, 64))
                else:
                    # Control '1' + '00' + meaningful bits
                    output.extend(self._write_control(1))
                    output.extend(self._write_control(0))
                    output.extend(self._write_bits(leading, 6))
                    output.extend(self._write_bits(significant - 1, 6))
                    output.extend(self._write_bits(
                        (xor >> trailing) & ((1 << significant) - 1),
                        significant
                    ))
        
        self._previous_value = value
        
        # Flush remaining bits
        if self._bits_in_buffer > 0:
            output.append((self._buffer << (8 - self._bits_in_buffer)) & 0xFF)
        
        return output
    
    def compress_timestamp(self, timestamp: int) -> list[int]:
        """
        Compress a timestamp using delta-of-delta encoding.
        
        Delta-of-delta encoding:
        - First timestamp stored as-is
        - Subsequent: compute delta = current - previous
        - Then encode delta_of_delta = delta - previous_delta
        """
        output = []
        
        if self._previous_timestamp is None:
            # First timestamp: store as 14 bits (enough for ~9 hours at 1ms resolution)
            output.extend(self._write_bits(timestamp, 14))
        else:
            delta = timestamp - self._previous_timestamp
            
            if self._previous_timestamp_delta is None:
                # Second timestamp: store delta
                if delta < 0:
                    delta = 0  # clamp to positive
                output.extend(self._write_bits(delta, 14))
                self._previous_timestamp_delta = delta
            else:
                delta_of_delta = delta - self._previous_timestamp_delta
                
                if delta_of_delta == 0:
                    output.extend(self._write_bits(0b10, 2))  # control 2
                elif -63 <= delta_of_delta <= 64:
                    output.extend(self._write_bits(0b110, 3))  # control 6
                    output.extend(self._write_bits(delta_of_delta + 63, 7))
                elif -255 <= delta_of_delta <= 256:
                    output.extend(self._write_bits(0b1110, 4))  # control 14
                    output.extend(self._write_bits(delta_of_delta + 255, 9))
                else:
                    output.extend(self._write_bits(0b1111, 4))  # control 15
                    output.extend(self._write_bits(delta_of_delta, 14))
                
                self._previous_timestamp_delta = delta
        
        self._previous_timestamp = timestamp
        return output
    
    def compress(self, values: list[float], timestamps: list[int] | None = None) -> bytes:
        """
        Compress a series of values.
        
        Args:
            values: List of float values to compress
            timestamps: Optional list of timestamps (milliseconds)
        
        Returns:
            Compressed bytes
        """
        if len(values) == 0:
            return b''
        
        if timestamps is None:
            # Generate synthetic timestamps at 1-second intervals
            timestamps = [i * 1000 for i in range(len(values))]
        elif len(timestamps) != len(values):
            raise ValueError("Values and timestamps must have same length")
        
        self.reset()
        output = bytearray()
        
        # Write header with number of points
        output.extend(struct.pack('>I', len(values)))
        
        for value, ts in zip(values, timestamps):
            # Write compressed value
            value_bytes = self.compress_value(value)
            output.extend(value_bytes)
            
            # Write compressed timestamp
            ts_bytes = self.compress_timestamp(ts)
            output.extend(ts_bytes)
        
        # Flush any remaining bits
        if self._bits_in_buffer > 0:
            output.append((self._buffer << (8 - self._bits_in_buffer)) & 0xFF)
        
        return bytes(output)
    
    def decompress(self, data: bytes) -> tuple[list[float], list[int]]:
        """
        Decompress data back to values and timestamps.
        
        Returns:
            Tuple of (values, timestamps)
        """
        if len(data) == 0:
            return [], []
        
        self.reset()
        
        # Read header
        num_points = struct.unpack('>I', data[:4])[0]
        offset = 4
        
        values = []
        timestamps = []
        
        for _ in range(num_points):
            # Read value
            value, read = self._read_value(data, offset)
            values.append(value)
            offset += read
            
            # Read timestamp
            ts, read = self._read_timestamp(data, offset)
            timestamps.append(ts)
            offset += read
        
        return values, timestamps
    
    def _read_value(self, data: bytes, offset: int) -> tuple[float, int]:
        """Read and decompress a single value."""
        bit_pos = offset * 8
        
        # Read first bit for control
        control = self._read_bits(data, bit_pos, 1)
        bit_pos += 1
        
        if control == 0:
            # First value or same value
            if self._previous_value is None:
                # First value: read 64 bits
                bits = self._read_bits(data, bit_pos, 64)
                self._previous_value = self._bits_to_float(bits)
                return self._previous_value, 9  # 1 + 64 bits = 9 bytes
            else:
                # Same value
                return self._previous_value, 1  # Just control bit
        else:
            # Second control bit
            control2 = self._read_bits(data, bit_pos, 1)
            bit_pos += 1
            
            if control2 == 0:
                # Compressed value
                leading = self._read_bits(data, bit_pos, 6)
                bit_pos += 6
                significant = self._read_bits(data, bit_pos, 6)
                bit_pos += 6
                significant += 1
                
                bits = self._read_bits(data, bit_pos, significant)
                bit_pos += significant
                
                # Reconstruct from previous value
                if self._previous_value is not None:
                    prev_bits = self._float_to_bits(self._previous_value)
                    bits = prev_bits ^ (bits << (64 - leading - significant))
                else:
                    bits = bits << (64 - leading - significant)
                
                self._previous_value = self._bits_to_float(bits)
                return self._previous_value, (bit_pos - offset * 8 + 7) // 8
            else:
                # Check marker
                marker = self._read_bits(data, bit_pos, 1)
                bit_pos += 1
                
                if marker == 0:
                    # Stored value
                    bits = self._read_bits(data, bit_pos, 64)
                    self._previous_value = self._bits_to_float(bits)
                    return self._previous_value, 10  # 1 + 1 + 1 + 64 bits
                else:
                    raise ValueError("Invalid Gorilla encoding")
    
    def _read_timestamp(self, data: bytes, offset: int) -> tuple[int, int]:
        """Read and decompress a single timestamp."""
        bit_pos = offset * 8
        
        if self._previous_timestamp is None:
            ts = self._read_bits(data, bit_pos, 14)
            self._previous_timestamp = ts
            self._previous_timestamp_delta = ts
            return ts, 2
        elif self._previous_timestamp_delta is None:
            delta = self._read_bits(data, bit_pos, 14)
            self._previous_timestamp = self._previous_timestamp + delta
            self._previous_timestamp_delta = delta
            return self._previous_timestamp, 2
        else:
            control = self._read_bits(data, bit_pos, 2)
            bit_pos += 2
            
            if control == 0b10:
                delta = self._previous_timestamp_delta
            elif control == 0b110:
                dod = self._read_bits(data, bit_pos, 7) - 63
                bit_pos += 7
                delta = self._previous_timestamp_delta + dod
            elif control == 0b1110:
                dod = self._read_bits(data, bit_pos, 9) - 255
                bit_pos += 9
                delta = self._previous_timestamp_delta + dod
            else:  # 0b1111
                dod = self._read_bits(data, bit_pos, 14)
                if dod >= 8192:
                    dod -= 16384
                bit_pos += 14
                delta = self._previous_timestamp_delta + dod
            
            self._previous_timestamp = self._previous_timestamp + delta
            self._previous_timestamp_delta = delta
            
            return self._previous_timestamp, (bit_pos - offset * 8 + 7) // 8
    
    def _read_bits(self, data: bytes, bit_pos: int, num_bits: int) -> int:
        """Read bits from byte array."""
        result = 0
        byte_pos = bit_pos // 8
        bit_offset = bit_pos % 8
        
        for i in range(num_bits):
            if byte_pos >= len(data):
                break
            bit = (data[byte_pos] >> (7 - bit_offset)) & 1
            result = (result << 1) | bit
            bit_offset += 1
            if bit_offset == 8:
                bit_offset = 0
                byte_pos += 1
        
        return result
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        if original_size == 0:
            return 0.0
        return 1 - (compressed_size / original_size)
    
    def estimate_compressed_size(self, values: list[float]) -> int:
        """Estimate the compressed size for a list of values."""
        # Rough estimation based on typical compression ratios
        avg_bits_per_value = 8  # Conservative estimate
        return (len(values) * avg_bits_per_value + 32) // 8  # +4 for header
