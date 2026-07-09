"""
Utils Module for Endee Vector Database Client

This module provides utility functions for validation and helper operations
used throughout the Endee client library.
"""

import re

from .constants import MAX_COLLECTION_NAME_LENGTH_ALLOWED


def validate_collection_name(name: str) -> None:
    """
    Validate a collection name and raise ValueError if invalid.

    Collection names must:
    - Be non-empty
    - Contain only alphanumeric characters (a-z, A-Z, 0-9) and underscores (_)
    - Not start with double underscore ('__')
    - Be no longer than MAX_COLLECTION_NAME_LENGTH_ALLOWED characters
    """
    if not name:
        raise ValueError("Collection name cannot be empty.")
    pattern = re.compile(r"^[a-zA-Z0-9_]+$")
    if not pattern.match(name):
        raise ValueError(
            "Collection name can only contain alphanumeric characters and underscores."
        )
    if name.startswith("__"):
        raise ValueError(
            "Collection name cannot start with double underscores ('__')."
        )
    if len(name) > MAX_COLLECTION_NAME_LENGTH_ALLOWED:
        raise ValueError(
            f"Collection name is too long. Maximum allowed length is "
            f"{MAX_COLLECTION_NAME_LENGTH_ALLOWED} characters."
        )
