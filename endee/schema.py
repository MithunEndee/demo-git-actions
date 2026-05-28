import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ._pydantic_compat import (
    PYDANTIC_V2,
    ConfigDict,
    field_validator,
    root_validator_compat,
)
from .constants import (
    DEFAULT_DENSE_RRF_WEIGHT,
    DEFAULT_EF_SEARCH,
    DEFAULT_FILTER_BOOST_PERCENTAGE,
    DEFAULT_PREFILTER_CARDINALITY_THRESHOLD,
    DEFAULT_RRF_RANK_CONSTANT,
    DEFAULT_TOPK,
    MAX_DIMENSION_ALLOWED,
    MAX_EF_SEARCH_ALLOWED,
    MAX_INDEX_NAME_LENGTH_ALLOWED,
    MAX_TOP_K_ALLOWED,
    PRECISION_TYPES_SUPPORTED,
    SPACE_TYPES_SUPPORTED,
    SPARSE_MODE_TYPES_SUPPORTED,
    Precision,
)


class VectorItem(BaseModel):
    """Model for a single vector item in an upsert operation."""

    id: str = Field(..., min_length=1)
    vector: List[float]
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)
    filter: Optional[Dict[str, Any]] = Field(default_factory=dict)
    sparse_indices: Optional[List[int]] = None
    sparse_values: Optional[List[float]] = None

    @root_validator_compat
    def validate_sparse_data(cls, values):
        sparse_indices = values.get("sparse_indices")
        sparse_values = values.get("sparse_values")
        if (sparse_indices is None) != (sparse_values is None):
            raise ValueError(
                "Both sparse_indices and sparse_values must be provided together"
            )
        if sparse_indices is not None and len(sparse_indices) != len(sparse_values):
            raise ValueError("sparse_indices and sparse_values must match in length")
        return values


class QueryRequest(BaseModel):
    """Model for query parameters."""

    vector: Optional[List[float]] = None
    top_k: int = Field(default=DEFAULT_TOPK, gt=0, le=MAX_TOP_K_ALLOWED)
    filter: Optional[List[Dict[str, Any]]] = None
    ef: int = Field(default=DEFAULT_EF_SEARCH, le=MAX_EF_SEARCH_ALLOWED)
    include_vectors: bool = False
    sparse_indices: Optional[List[int]] = None
    sparse_values: Optional[List[float]] = None
    prefilter_cardinality_threshold: int = Field(
        default=DEFAULT_PREFILTER_CARDINALITY_THRESHOLD, ge=1_000, le=1_000_000
    )
    filter_boost_percentage: int = Field(
        default=DEFAULT_FILTER_BOOST_PERCENTAGE, ge=0, le=400
    )
    dense_rrf_weight: float = Field(default=DEFAULT_DENSE_RRF_WEIGHT, ge=0.0, le=1.0)
    rrf_rank_constant: int = Field(default=DEFAULT_RRF_RANK_CONSTANT, ge=1)

    @root_validator_compat
    def validate_query_type(cls, values):
        vector = values.get("vector")
        sparse_indices = values.get("sparse_indices")
        sparse_values = values.get("sparse_values")

        has_dense = vector is not None
        has_sparse = sparse_indices is not None or sparse_values is not None

        if not has_dense and not has_sparse:
            raise ValueError(
                "At least one of 'vector' or 'sparse_indices'/'sparse_values'"
                " must be provided."
            )

        if (sparse_indices is None) != (sparse_values is None):
            raise ValueError(
                "Both sparse_indices and sparse_values must be provided together"
            )

        if sparse_indices is not None and len(sparse_indices) != len(sparse_values):
            raise ValueError("sparse_indices and sparse_values must match in length")

        return values


class IndexCreateRequest(BaseModel):
    """Model for index creation parameters."""

    name: str
    dimension: int = Field(..., ge=2, le=MAX_DIMENSION_ALLOWED)
    space_type: str
    M: int = Field(..., gt=0)
    ef_con: int = Field(..., gt=0)
    precision: Union[str, Precision]
    version: Optional[int] = None
    sparse_model: Optional[str] = None

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError(
                "Index name must be alphanumeric and can contain underscores"
            )
        if len(v) > MAX_INDEX_NAME_LENGTH_ALLOWED:
            raise ValueError(
                f"Index name should be less than {MAX_INDEX_NAME_LENGTH_ALLOWED}"
                " characters"
            )
        return v

    @field_validator("space_type")
    def validate_space_type(cls, v: str) -> str:
        v = v.lower()
        if v not in SPACE_TYPES_SUPPORTED:
            raise ValueError(
                f"Invalid space type: {v}. Must be one of {SPACE_TYPES_SUPPORTED}"
            )
        return v

    @field_validator("precision")
    def validate_precision(cls, v: Union[str, Precision]) -> Union[str, Precision]:
        if isinstance(v, Precision):
            return v
        if v not in PRECISION_TYPES_SUPPORTED:
            raise ValueError(
                f"Invalid precision: {v}. Must be one of {PRECISION_TYPES_SUPPORTED}"
            )
        return v

    @field_validator("sparse_model")
    def validate_sparse_model(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None

        normalized = v.strip().lower()
        if normalized == "none":
            return None

        if normalized not in SPARSE_MODE_TYPES_SUPPORTED:
            supported_models = "', '".join(["None", *SPARSE_MODE_TYPES_SUPPORTED])
            raise ValueError(
                f"Invalid sparse_model: {v}. Must be one of '{supported_models}'"
            )

        return normalized


class _IndexMetadataBase(BaseModel):
    """Base fields for IndexMetadata (shared between v1 and v2)."""

    name: Optional[str] = Field(None, alias="name")
    lib_token: str
    total_elements: int = Field(..., alias="total_elements")
    space_type: str = Field(..., alias="space_type")
    dimension: int = Field(..., alias="dimension")
    precision: Optional[str] = Field(None, alias="precision")
    M: int = Field(..., alias="M")
    ef_con: int = Field(..., alias="ef_con")
    sparse_model: Optional[str] = Field(None, alias="sparse_model")


if PYDANTIC_V2:

    class IndexMetadata(_IndexMetadataBase):
        """Model for index metadata returned by the server."""

        model_config = ConfigDict(populate_by_name=True)

else:

    class IndexMetadata(_IndexMetadataBase):  # type: ignore[no-redef]
        """Model for index metadata returned by the server."""

        class Config:
            allow_population_by_field_name = True


class FilterUpdateItem(BaseModel):
    """Model for a single filter update item."""

    id: str = Field(..., min_length=1)
    filter: Dict[str, Any]


class FilterUpdateRequest(BaseModel):
    """Model for multiple filter update requests."""

    updates: List[FilterUpdateItem]


class IndexRebuildRequest(BaseModel):
    """
    Model for index rebuild request parameters.
    Only M and ef_con can be updated.
    """

    M: int = Field(..., gt=0)
    ef_con: int = Field(..., gt=0)
