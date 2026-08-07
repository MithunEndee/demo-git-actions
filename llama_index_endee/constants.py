"""Constants for EndeeVectorStore.

Fallback defaults are defined first, then overridden from the endee SDK
when available. This keeps the wrapper functional even if the SDK is an
older version that doesn't expose every constant.
"""

from llama_index.core.vector_stores.types import FilterOperator

# ---------------------------------------------------------------------------
# Fallback defaults (overridden from SDK below when possible)
# ---------------------------------------------------------------------------

# HNSW defaults
DEFAULT_EF_SEARCH = 128
DEFAULT_M = 16
DEFAULT_EF_CON = 128

# Limits
MAX_VECTORS_PER_BATCH = 10_000
MAX_DIMENSION_ALLOWED = 10_000
MAX_EF_SEARCH_ALLOWED = 1024
MAX_TOP_K_ALLOWED = 512
MAX_KEY_BYTES = 128
MAX_VALUE_BYTES = 1024

# Query-tuning defaults
DEFAULT_PREFILTER_CARDINALITY_THRESHOLD = 10_000
DEFAULT_FILTER_BOOST_PERCENTAGE = 0
DEFAULT_DENSE_RRF_WEIGHT = 0.5
DEFAULT_RRF_RANK_CONSTANT = 60

# Sparse model
SPARSE_MODE_TYPES_SUPPORTED = ["default", "endee_bm25"]

# ---------------------------------------------------------------------------
# Override from endee SDK (each import is independent so partial failures
# don't skip everything)
# ---------------------------------------------------------------------------
import importlib as _importlib

if _importlib.util.find_spec("endee.constants"):
    from endee import constants as _ec

    for _name in (
        "DEFAULT_EF_SEARCH", "DEFAULT_M", "DEFAULT_EF_CON",
        "MAX_VECTORS_PER_BATCH", "MAX_DIMENSION_ALLOWED",
        "MAX_EF_SEARCH_ALLOWED", "MAX_TOP_K_ALLOWED",
        "MAX_KEY_BYTES", "MAX_VALUE_BYTES",
        "SPARSE_MODE_TYPES_SUPPORTED",
    ):
        if hasattr(_ec, _name):
            globals()[_name] = getattr(_ec, _name)


# ---------------------------------------------------------------------------
# Precision enum (from SDK, with fallback)
# ---------------------------------------------------------------------------
try:
    from endee.constants import Precision
except ImportError:
    from enum import Enum

    class Precision(str, Enum):  # type: ignore[no-redef]
        """Fallback Precision enum when endee SDK is too old."""

        BINARY2 = "binary"
        FLOAT16 = "float16"
        FLOAT32 = "float32"
        INT16 = "int16"
        INT8 = "int8"


# ---------------------------------------------------------------------------
# Integration-level constants (not from SDK)
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 100

# LlamaIndex FilterOperator -> Endee API symbol
SUPPORTED_FILTER_OPERATORS = (
    FilterOperator.EQ,
    FilterOperator.IN,
)

REVERSE_OPERATOR_MAP = {
    FilterOperator.EQ: "$eq",
    FilterOperator.IN: "$in",
}
