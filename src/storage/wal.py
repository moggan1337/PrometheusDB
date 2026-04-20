"""
Write-Ahead Log (WAL) Implementation.

The WAL provides durability for writes by logging all operations
before they're applied to the main storage. This ensures data
integrity in case of crashes.
"""

from __future__ import annotations

import os
import struct
import threading
import time
from pathlib import Path
from typing import BinaryIO, Iterator


class WriteAheadLog:
    """
    Write-Ahead Log for durability.
    
    The WAL logs all write operations to disk before they're
    applied to the main storage. On startup, the WAL can be
    replayed to recover any uncommitted writes.
    
    Operations:
    - write(): Append a write record
    - flush(): Force write to disk
    - replay(): Replay WAL to recover data
    - truncate(): Clear old WAL segments
    """
    
    HEADER_SIZE = 16
    RECORD_HEADER_SIZE = 12  # type(1) + key_len(2) + timestamp(8) + value_len(1)
    
    # Record types
    WRITE = 0x01
    DELETE = 0x02
    BATCH = 0x03
    
    def __init__(self, directory: str, segment_size: int = 64 * 1024 * 1024):
        """
        Initialize WAL.
        
        Args:
            directory: Directory for WAL files
            segment_size: Size of each segment file (default 64MB)
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        
        self.segment_size = segment_size
        self._lock = threading.Lock()
        
        # Current segment
        self._current_segment = 0
        self._segment_offset = 0
        self._current_file: BinaryIO | None = None
        
        # Find last segment
        self._find_last_segment()
        
        # Open or create new segment
        self._open_segment()
    
    def _find_last_segment(self) -> None:
        """Find the last segment number."""
        segments = list(self.directory.glob("*.wal"))
        if segments:
            max_num = max(int(s.stem) for s in segments)
            self._current_segment = max_num
    
    def _segment_path(self, segment_num: int) -> Path:
        """Get path for a segment."""
        return self.directory / f"{segment_num:08d}.wal"
    
    def _open_segment(self) -> None:
        """Open a new segment or continue existing one."""
        path = self._segment_path(self._current_segment)
        
        # Open in append mode
        self._current_file = open(path, 'ab')
        self._segment_offset = self._current_file.tell()
        
        # If new segment, write header
        if self._segment_offset == 0:
            self._write_segment_header()
            self._segment_offset = self.HEADER_SIZE
    
    def _write_segment_header(self) -> None:
        """Write segment header."""
        if self._current_file:
            header = struct.pack('>Q', int(time.time() * 1000))
            self._current_file.write(header)
            self._current_file.flush()
    
    def write(self, key: str, timestamp: int, value: float) -> None:
        """
        Write a record to the WAL.
        
        Args:
            key: Time series key
            timestamp: Unix timestamp in milliseconds
            value: Metric value
        """
        with self._lock:
            if self._current_file is None:
                self._open_segment()
            
            # Encode record
            key_bytes = key.encode('utf-8')
            key_len = len(key_bytes)
            value_bytes = struct.pack('>d', value)
            value_len = len(value_bytes)
            
            record_size = self.RECORD_HEADER_SIZE + key_len + value_len
            
            # Check if we need a new segment
            if self._segment_offset + record_size > self.segment_size:
                self._rotate_segment()
            
            # Write record header
            header = struct.pack(
                '>BHHq',
                self.WRITE,
                key_len,
                timestamp,
                value_len
            )
            self._current_file.write(header)
            
            # Write key and value
            self._current_file.write(key_bytes)
            self._current_file.write(value_bytes)
            
            self._segment_offset += record_size
    
    def write_batch(self, records: list[tuple[str, int, float]]) -> None:
        """
        Write multiple records in a batch.
        
        Args:
            records: List of (key, timestamp, value) tuples
        """
        with self._lock:
            for key, timestamp, value in records:
                self.write(key, timestamp, value)
    
    def _rotate_segment(self) -> None:
        """Rotate to a new segment."""
        if self._current_file:
            self._current_file.close()
        
        self._current_segment += 1
        self._segment_offset = 0
        self._open_segment()
    
    def flush(self) -> None:
        """Force flush to disk."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
    
    def replay(self) -> Iterator[tuple[str, int, float]]:
        """
        Replay the WAL and yield all write records.
        
        Yields:
            Tuples of (key, timestamp, value)
        """
        for segment_path in sorted(self.directory.glob("*.wal")):
            yield from self._replay_segment(segment_path)
    
    def _replay_segment(self, path: Path) -> Iterator[tuple[str, int, float]]:
        """Replay a single segment."""
        with open(path, 'rb') as f:
            # Skip header
            f.read(self.HEADER_SIZE)
            
            while True:
                header = f.read(self.RECORD_HEADER_SIZE)
                if len(header) < self.RECORD_HEADER_SIZE:
                    break
                
                record_type, key_len, timestamp, value_len = struct.unpack(
                    '>BHHq', header
                )
                
                if record_type == self.WRITE:
                    key = f.read(key_len).decode('utf-8')
                    value_bytes = f.read(value_len)
                    value = struct.unpack('>d', value_bytes)[0]
                    yield (key, timestamp, value)
                
                elif record_type == self.DELETE:
                    f.read(key_len)
                
                elif record_type == self.BATCH:
                    # Batch header - read count
                    count_data = f.read(4)
                    if len(count_data) < 4:
                        break
                    count = struct.unpack('>I', count_data)[0]
                    
                    for _ in range(count):
                        batch_header = f.read(self.RECORD_HEADER_SIZE)
                        if len(batch_header) < self.RECORD_HEADER_SIZE:
                            break
                        
                        _, key_len, timestamp, value_len = struct.unpack(
                            '>BHHq', batch_header
                        )
                        key = f.read(key_len).decode('utf-8')
                        value_bytes = f.read(value_len)
                        value = struct.unpack('>d', value_bytes)[0]
                        yield (key, timestamp, value)
    
    def truncate(self, keep_segments: int = 2) -> int:
        """
        Truncate old segments, keeping the most recent ones.
        
        Args:
            keep_segments: Number of recent segments to keep
        
        Returns:
            Number of segments deleted
        """
        with self._lock:
            segments = sorted(int(s.stem) for s in self.directory.glob("*.wal"))
            
            if len(segments) <= keep_segments:
                return 0
            
            deleted = 0
            for seg_num in segments[:-keep_segments]:
                path = self._segment_path(seg_num)
                if path.exists():
                    path.unlink()
                    deleted += 1
            
            return deleted
    
    def close(self) -> None:
        """Close the WAL."""
        with self._lock:
            if self._current_file:
                self.flush()
                self._current_file.close()
                self._current_file = None
    
    def get_segment_count(self) -> int:
        """Get number of WAL segments."""
        return len(list(self.directory.glob("*.wal")))
    
    def get_size_bytes(self) -> int:
        """Get total WAL size in bytes."""
        total = 0
        for path in self.directory.glob("*.wal"):
            total += path.stat().st_size
        return total
