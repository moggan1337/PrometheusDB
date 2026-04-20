"""
Delta-of-Delta Compression for Timestamp Encoding.

Delta-of-delta encoding is particularly effective for timestamps that
have regular intervals with small variations. It works by:

1. Computing delta = t[i] - t[i-1]
2. Computing delta_of_delta = delta - previous_delta
3. Encoding delta_of_delta with variable-length encoding

This approach is especially effective for:
- Sensor data with regular sampling intervals
- Metrics with consistent scrape intervals
- Events with predictable timing patterns
"""

from __future__ import annotations

import struct
from typing import BinaryIO


class DeltaOfDeltaCompressor:
    """
    Compress timestamps using delta-of-delta encoding.
    
    Delta-of-delta encoding achieves better compression for regular time series
    where the interval between samples is consistent.
    
    Encoding scheme:
    - 0: delta_of_delta = 0 (same as previous interval)
    - 1: delta_of_delta in [-63, 64] (stored in 7 bits with offset)
    - 2: delta_of_delta in [-255, 256] (stored in 9 bits with offset)
    - 3: delta_of_delta in [-2047, 2048] (stored in 12 bits with offset)
    - 4: raw delta_of_delta in 32 bits
    """
    
    ENCODING_THRESHOLDS = [
        (0, 0, 1),
        (-63, 64, 7),
        (-255, 256, 9),
        (-2047, 2048, 12),
    ]
    
    def __init__(self, base_interval_ms: int = 1000):
        """
        Initialize the compressor.
        
        Args:
            base_interval_ms: Expected base interval between samples in milliseconds.
                              This helps optimize encoding for the common case.
        """
        self.base_interval_ms = base_interval_ms
        self._previous_timestamp: int | None = None
        self._previous_delta: int | None = None
        self._buffer = bytearray()
        self._bit_buffer = 0
        self._bits_in_buffer = 0
        
    def reset(self) -> None:
        """Reset compressor state."""
        self._previous_timestamp = None
        self._previous_delta = None
        self._buffer = bytearray()
        self._bit_buffer = 0
        self._bits_in_buffer = 0
    
    def _write_bits(self, value: int, num_bits: int) -> None:
        """Write bits to the internal buffer."""
        self._bit_buffer = (self._bit_buffer << num_bits) | (value & ((1 << num_bits) - 1))
        self._bits_in_buffer += num_bits
        
        while self._bits_in_buffer >= 8:
            self._bits_in_buffer -= 8
            self._buffer.append((self._bit_buffer >> self._bits_in_buffer) & 0xFF)
    
    def _flush_bits(self) -> None:
        """Flush remaining bits in the buffer."""
        if self._bits_in_buffer > 0:
            self._buffer.append((self._bit_buffer << (8 - self._bits_in_buffer)) & 0xFF)
            self._bits_in_buffer = 0
            self._bit_buffer = 0
    
    def _encode(self, delta: int) -> list[int]:
        """Encode a single delta value."""
        if self._previous_delta is None:
            # First delta: store as-is with 14 bits (supports up to ~4.6 hours at 1ms)
            return [14, delta]
        
        dod = delta - self._previous_delta
        
        if dod == 0:
            return [0]
        
        # Try to find the smallest encoding that fits
        for offset, (min_val, max_val, num_bits) in enumerate(self.ENCODING_THRESHOLDS[1:], 1):
            if min_val <= dod <= max_val:
                return [offset, dod - offset]
        
        # Fallback: raw 32-bit encoding
        return [4, dod]
    
    def compress(self, timestamps: list[int]) -> bytes:
        """
        Compress a list of timestamps.
        
        Args:
            timestamps: List of timestamps in milliseconds
        
        Returns:
            Compressed bytes
        """
        self.reset()
        
        if not timestamps:
            return b''
        
        # Write header: number of timestamps
        output = bytearray(struct.pack('>I', len(timestamps)))
        output.extend(struct.pack('>Q', self.base_interval_ms))
        
        for ts in sorted(timestamps):
            if self._previous_timestamp is None:
                # First timestamp: store as-is
                output.extend(struct.pack('>Q', ts))
            else:
                delta = ts - self._previous_timestamp
                encoded = self._encode(delta)
                
                # Write encoding type and value
                for val in encoded:
                    if val < 128:
                        output.append(val)
                    else:
                        # Use escape sequence for larger values
                        output.append(0x80 | (val >> 8))
                        output.append(val & 0xFF)
            
            self._previous_timestamp = ts
            if self._previous_timestamp is not None:
                if self._previous_delta is None:
                    self._previous_delta = ts - self._previous_timestamp
        
        return bytes(output)
    
    def decompress(self, data: bytes) -> list[int]:
        """
        Decompress timestamps from bytes.
        
        Args:
            data: Compressed bytes
        
        Returns:
            List of timestamps in milliseconds
        """
        if len(data) < 12:
            return []
        
        num_timestamps = struct.unpack('>I', data[:4])[0]
        self.base_interval_ms = struct.unpack('>Q', data[4:12])[0]
        
        offset = 12
        timestamps = []
        prev_ts: int | None = None
        prev_delta: int | None = None
        
        for _ in range(num_timestamps):
            if prev_ts is None:
                # First timestamp
                if offset + 8 > len(data):
                    break
                ts = struct.unpack('>Q', data[offset:offset+8])[0]
                offset += 8
                prev_delta = 0
            else:
                # Read encoding type
                encoding_type = data[offset]
                offset += 1
                
                if encoding_type == 0:
                    # Same as previous delta
                    ts = prev_ts + (prev_delta or 0)
                elif encoding_type < 4:
                    # Variable length encoding
                    offset_idx = encoding_type
                    _, _, num_bits = self.ENCODING_THRESHOLDS[offset_idx]
                    
                    if offset + ((num_bits + 7) // 8) > len(data):
                        break
                    
                    dod = 0
                    for i in range(num_bits):
                        byte_val = data[offset + i // 8]
                        bit_val = (byte_val >> (7 - (i % 8))) & 1
                        dod = (dod << 1) | bit_val
                    
                    dod = dod - self.ENCODING_THRESHOLDS[offset_idx][1]  # Apply offset
                    delta = prev_delta + dod if prev_delta else dod
                    ts = prev_ts + delta
                    offset += (num_bits + 7) // 8
                elif encoding_type == 4:
                    # Raw 32-bit
                    if offset + 4 > len(data):
                        break
                    dod = struct.unpack('>i', data[offset:offset+4])[0]
                    offset += 4
                    delta = prev_delta + dod if prev_delta else dod
                    ts = prev_ts + delta
                elif encoding_type >= 0x80:
                    # Escape sequence
                    if encoding_type & 0x80:
                        val = ((encoding_type & 0x7F) << 8) | data[offset]
                        offset += 1
                        if val >= 16384:
                            val -= 32768
                        dod = val
                        delta = prev_delta + dod if prev_delta else dod
                        ts = prev_ts + delta
                    else:
                        break
                else:
                    break
            
            timestamps.append(ts)
            prev_ts = ts
            if prev_delta is not None or len(timestamps) > 1:
                if len(timestamps) >= 2:
                    prev_delta = timestamps[-1] - timestamps[-2]
        
        return timestamps


class DoubleDeltaCompressor:
    """
    Alternative implementation using a simplified double-delta approach.
    
    This is optimized for Prometheus-style data where scrape intervals
    are typically consistent (e.g., 15s default).
    """
    
    def __init__(self, expected_interval_ms: int = 15000):
        self.expected_interval_ms = expected_interval_ms
        self.timestamps: list[int] = []
    
    def add(self, timestamp: int) -> None:
        """Add a timestamp to the compressor."""
        self.timestamps.append(timestamp)
    
    def compress(self) -> bytes:
        """Compress all stored timestamps."""
        if not self.timestamps:
            return b''
        
        output = bytearray()
        output.extend(struct.pack('>I', len(self.timestamps)))
        output.extend(struct.pack('>Q', self.expected_interval_ms))
        
        self.timestamps.sort()
        
        prev = self.timestamps[0]
        output.extend(struct.pack('>Q', prev))
        
        prev_delta = 0
        
        for ts in self.timestamps[1:]:
            delta = ts - prev
            dod = delta - prev_delta
            
            # Encode based on common patterns
            if dod == 0:
                output.append(0)  # Single byte for no change
            elif -7 <= dod <= 8:
                # Fit in 4 bits
                output.append(0x10 | (dod + 7))
            elif -127 <= dod <= 128:
                # Fit in 8 bits
                output.append(0x20)
                output.append((dod + 127) & 0xFF)
            else:
                # Store raw delta (16 bits)
                output.append(0x30)
                output.extend(struct.pack('>h', dod))
            
            prev_delta = delta
            prev = ts
        
        return bytes(output)
    
    def decompress(self, data: bytes) -> list[int]:
        """Decompress timestamps."""
        if len(data) < 12:
            return []
        
        num = struct.unpack('>I', data[:4])[0]
        expected = struct.unpack('>Q', data[4:12])[0]
        
        result = []
        offset = 12
        prev_ts = struct.unpack('>Q', data[offset:offset+8])[0]
        result.append(prev_ts)
        offset += 8
        
        prev_delta = 0
        
        for _ in range(num - 1):
            if offset >= len(data):
                break
            
            code = data[offset]
            offset += 1
            
            if code == 0:
                dod = 0
            elif (code & 0xF0) == 0x10:
                dod = (code & 0x0F) - 7
            elif code == 0x20:
                dod = struct.unpack('>b', bytes([data[offset]]))[0]
                offset += 1
            elif code == 0x30:
                dod = struct.unpack('>h', data[offset:offset+2])[0]
                offset += 2
            else:
                dod = 0  # Fallback
            
            delta = prev_delta + dod
            ts = prev_ts + delta
            result.append(ts)
            
            prev_delta = delta
            prev_ts = ts
        
        return result
    
    def reset(self) -> None:
        """Reset the compressor."""
        self.timestamps = []
