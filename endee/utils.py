"""
Utils Module for Endee Vector Database Client

This module provides utility functions for validation and helper operations
used throughout the Endee client library.
"""

import re

from .constants import MAX_INDEX_NAME_LENGTH_ALLOWED


def is_valid_index_name(index_name):
    """
    Validate an index name according to Endee naming rules.

    Index names must:
    - Contain only alphanumeric characters (a-z, A-Z, 0-9) and underscores (_)
    - Be no longer than MAX_INDEX_NAME_LENGTH_ALLOWED characters

    Args:
        index_name (str): The index name to validate

    Returns:
        bool: True if the index name is valid, False otherwise

    Example:
        >>> is_valid_index_name("my_index_123")
        True
        >>> is_valid_index_name("my-index")  # Hyphens not allowed
        False
        >>> is_valid_index_name("my index")  # Spaces not allowed
        False
        >>> is_valid_index_name("a" * 100)  # Too long
        False
    """
    # Pattern matches alphanumeric characters and underscores only
    pattern = re.compile(r"^[a-zA-Z0-9_]+$")

    # Check both pattern match and length constraint
    return (
        pattern.match(index_name) is not None
        and len(index_name) <= MAX_INDEX_NAME_LENGTH_ALLOWED
    )
