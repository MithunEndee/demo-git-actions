from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ._pydantic_compat import field_validator, root_validator_compat
from .constants import (
    PRECISION_TYPES_SUPPORTED,
    SPACE_TYPES_SUPPORTED,
    SPARSE_MODE_TYPES_SUPPORTED,
)

# ─── Collection field config ───────────────────────────────────────────────────

COLLECTION_FIELD_TYPES = ("vector", "sparse", "multi_vector")

_POOLING_METHOD_MAP = {
    "average_pooling": "mean",
    "max_pooling": "max",
    "mean": "mean",
    "max": "max",
}


class CollectionFieldParams(BaseModel):
    """Parameters for vector and multi_vector fields."""

    dimension: Optional[int] = Field(None, ge=2)
    space_type: Optional[str] = Field(None)
    precision: Optional[str] = Field(None)
    m: Optional[int] = Field(None, gt=0)               # HNSW M (bi-directional links)
    ef_construct: Optional[int] = Field(None, gt=0)    # HNSW ef_construction

    @field_validator("space_type")
    def validate_space_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower()
        if v not in SPACE_TYPES_SUPPORTED:
            raise ValueError(
                f"Invalid space type: {v}. Must be one of {SPACE_TYPES_SUPPORTED}"
            )
        return v

    @field_validator("precision")
    def validate_precision(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in PRECISION_TYPES_SUPPORTED:
            raise ValueError(
                f"Invalid precision: {v}. Must be one of {PRECISION_TYPES_SUPPORTED}"
            )
        return v


class CollectionFieldConfig(BaseModel):
    """Configuration for a single field in a collection.

    Field types
    -----------
    ``vector``
        Single dense vector per object. Requires ``params``.
    ``sparse``
        Sparse vector per object. Requires ``sparse_model``, no ``params``.
    ``multi_vector``
        Multiple dense vectors per object.
        Requires ``params`` AND ``pooling_method``.

    Examples::

        CollectionFieldConfig(
            name="embedding", type="vector",
            params=CollectionFieldParams(dimension=768, space_type="cosine", precision="int8"),
        )

        CollectionFieldConfig(name="keywords", type="sparse", sparse_model="default")

        CollectionFieldConfig(
            name="multivec", type="multi_vector",
            pooling_method="average_pooling",
            params=CollectionFieldParams(dimension=128, space_type="cosine", precision="int8"),
        )
    """

    name: str = Field(..., min_length=1)
    type: str = Field(...)
    params: Optional[CollectionFieldParams] = None
    sparse_model: Optional[str] = Field(None)
    pooling_method: Optional[str] = Field(None)

    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        if v not in COLLECTION_FIELD_TYPES:
            raise ValueError(
                f"Invalid field type: {v}. Must be one of {COLLECTION_FIELD_TYPES}"
            )
        return v

    @field_validator("sparse_model")
    def validate_sparse_model(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        normalized = v.strip().lower()
        if normalized == "none":
            return None
        if normalized not in SPARSE_MODE_TYPES_SUPPORTED:
            raise ValueError(
                f"Invalid sparse_model: {v}. Must be one of {['None', *SPARSE_MODE_TYPES_SUPPORTED]}"
            )
        return normalized

    @root_validator_compat
    def validate_field_rules(cls, values):
        field_type = values.get("type")
        sparse_model = values.get("sparse_model")
        pooling_method = values.get("pooling_method")

        if field_type == "sparse":
            if sparse_model is None:
                raise ValueError("sparse fields require sparse_model")
            if pooling_method is not None:
                raise ValueError("pooling_method is only valid for multi_vector fields")

        elif field_type == "multi_vector":
            if pooling_method is None:
                raise ValueError("multi_vector fields require pooling_method")
            if sparse_model is not None:
                raise ValueError("sparse_model is only valid for sparse fields")

        elif field_type == "vector":
            if sparse_model is not None:
                raise ValueError("sparse_model is only valid for sparse fields")
            if pooling_method is not None:
                raise ValueError("pooling_method is only valid for multi_vector fields")

        return values

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "type": self.type}
        params_d: dict = {}
        if self.params is not None:
            if self.params.dimension is not None:
                params_d["dimension"] = self.params.dimension
            if self.params.space_type is not None:
                params_d["space_type"] = self.params.space_type
            if self.params.precision is not None:
                params_d["precision"] = self.params.precision
            if self.params.m is not None:
                params_d["m"] = self.params.m
            if self.params.ef_construct is not None:
                params_d["ef_construct"] = self.params.ef_construct
        if self.pooling_method is not None:
            params_d["pooling"] = _POOLING_METHOD_MAP.get(
                self.pooling_method, self.pooling_method
            )
        if params_d:
            d["params"] = params_d
        if self.sparse_model is not None:
            d["sparse_model"] = self.sparse_model
        return d


# ─── Collection metadata ───────────────────────────────────────────────────────

class CollectionMetadata(BaseModel):
    """Metadata returned by the server for a collection."""

    name: str
    fields: List[Any] = Field(default_factory=list)
    created_at: Optional[Any] = None
    layout_version: Optional[int] = None


# ─── Object upsert ────────────────────────────────────────────────────────────

class ObjectFieldInput(BaseModel):
    """Wire format for a single field value within an upserted object."""

    vector: Optional[List[float]] = None
    vectors: Optional[List[List[float]]] = None
    sparse_indices: Optional[List[int]] = None
    sparse_values: Optional[List[float]] = None

    @root_validator_compat
    def validate_sparse(cls, values):
        si = values.get("sparse_indices")
        sv = values.get("sparse_values")
        if (si is None) != (sv is None):
            raise ValueError("sparse_indices and sparse_values must be provided together")
        if si is not None and len(si) != len(sv):
            raise ValueError("sparse_indices and sparse_values must match in length")
        return values

    def to_dict(self) -> dict:
        d = {}
        if self.vector is not None:
            d["vector"] = self.vector
        if self.vectors is not None:
            d["vectors"] = self.vectors
        if self.sparse_indices is not None:
            d["sparse_indices"] = self.sparse_indices
            d["sparse_values"] = self.sparse_values
        return d


class ObjectInput(BaseModel):
    """Input for upserting a single object into a collection."""

    id: str = Field(..., min_length=1)
    meta: Optional[Any] = None
    filter: Optional[Union[Dict[str, Any], str]] = None
    fields: Optional[Dict[str, ObjectFieldInput]] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        import json as _json
        d: dict = {"id": self.id}
        if self.meta is not None:
            d["meta"] = self.meta
        if self.filter is not None:
            d["filter"] = (
                _json.dumps(self.filter, separators=(",", ":"))
                if isinstance(self.filter, dict)
                else self.filter
            )
        if self.fields:
            d["fields"] = {k: v.to_dict() for k, v in self.fields.items()}
        return d
