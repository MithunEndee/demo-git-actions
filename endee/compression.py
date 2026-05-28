"""
Compression Utilities for Endee Vector Database Client

This module provides lightweight compression utilities for vector
metadata. Metadata is serialized to JSON and compressed using zlib
to reduce storage size, memory usage, and network transfer overhead.
"""

import zlib

import orjson


def json_zip(data: dict) -> bytes:
    """
    Compress a dictionary using JSON serialization and zlib.

    Args:
        data: Dictionary to compress

    Returns:
        bytes: Compressed data, or empty bytes if input is empty

    Example:
        >>> compressed = json_zip({"user": "alice", "age": 30})
    """
    if not data:
        return b""

    return zlib.compress(orjson.dumps(data))


def json_unzip(compressed_data: bytes) -> dict:
    """
    Decompress data produced by json_zip back into a dictionary.

    Args:
        compressed_data: Compressed bytes

    Returns:
        dict: Decompressed dictionary, or empty dict if input is empty

    Example:
        >>> original = json_unzip(compressed_data)
    """
    if not compressed_data:
        return {}

    return orjson.loads(zlib.decompress(compressed_data))
