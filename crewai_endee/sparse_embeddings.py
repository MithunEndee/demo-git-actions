"""Sparse embedding models for hybrid search with Endee."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class SparseVector(BaseModel):
    """Sparse vector with non-zero indices and their values."""

    indices: list[int] = Field(..., description="indices must be unique")
    values: list[float] = Field(
        ..., description="values and indices must be the same length"
    )


class SparseEmbeddings(ABC):
    """Interface for sparse embedding models used with Endee."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Embed a list of documents as sparse vectors."""

    @abstractmethod
    def embed_query(self, text: str) -> SparseVector:
        """Embed a single query as a sparse vector."""


class SparseModelAdapter(SparseEmbeddings):
    """Wraps any sparse model with ``.embed()`` and ``.query_embed()``
    into the ``SparseEmbeddings`` interface.

    Example::

        from fastembed import SparseTextEmbedding
        model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

        store = EndeeVectorStore(
            ...,
            sparse_embedding=model,   # auto-wrapped internally
        )
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
            for r in self._model.embed(texts)
        ]

    def embed_query(self, text: str) -> SparseVector:
        r = next(self._model.query_embed(text))
        return SparseVector(indices=r.indices.tolist(), values=r.values.tolist())


def wrap_sparse_model(model: Any) -> SparseEmbeddings:
    """If ``model`` is already a ``SparseEmbeddings``, return it as-is.
    Otherwise wrap it in ``SparseModelAdapter``."""
    if isinstance(model, SparseEmbeddings):
        return model
    if hasattr(model, "embed") and hasattr(model, "query_embed"):
        return SparseModelAdapter(model)
    msg = (
        "sparse_embedding must be a SparseEmbeddings instance or an object "
        "with .embed() and .query_embed() methods "
        f"(got {type(model).__name__})"
    )
    raise TypeError(msg)


class EndeeModelSparse(SparseEmbeddings):
    """Sparse embeddings using endee_model's native BM25.

    Computes BM25 term-frequency weights on the client. The Endee server
    applies IDF weighting when ``sparse_model="endee_bm25"`` is set on the
    collection field.

    Args:
        model_name: Model identifier. Default: ``"endee/bm25"``.
        k: BM25 saturation parameter. Default: 1.2.
        b: BM25 length normalisation parameter. Default: 0.75.
        avg_len: Expected average document length in tokens. Default: 256.0.
        language: Language for NLTK stopwords / Snowball stemmer.
        cache_dir: Optional cache directory for NLTK data.
        kwargs: Forwarded to ``endee_model.SparseModel``.
    """

    def __init__(
        self,
        model_name: str = "endee/bm25",
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 256.0,
        language: str = "english",
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from endee_model import SparseModel
        except ImportError as err:
            msg = (
                "The 'endee_model' package is not installed. "
                "Install it with: pip install endee-model"
            )
            raise ImportError(msg) from err

        self._model = SparseModel(
            model_name=model_name,
            cache_dir=cache_dir,
            k=k,
            b=b,
            avg_len=avg_len,
            language=language,
            **kwargs,
        )

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Embed documents as sparse vectors (BM25 TF weights)."""
        return [
            SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
            for r in self._model.embed(texts)
        ]

    def embed_query(self, text: str) -> SparseVector:
        """Embed a query as a sparse vector (unique term IDs, unit weights)."""
        r = next(self._model.query_embed(text))
        return SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
