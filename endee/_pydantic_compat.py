from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION

PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")
Model = TypeVar("Model", bound=BaseModel)


if PYDANTIC_V2:
    from pydantic import ConfigDict
    from pydantic import field_validator as _pydantic_field_validator
    from pydantic import model_validator as _model_validator

    def field_validator(*fields: str, **kwargs: Any):
        """
        Pydantic v2 branch: wraps field_validator + classmethod internally
        so callers don't need @classmethod in their code.
        """
        pydantic_dec = _pydantic_field_validator(*fields, **kwargs)

        def decorator(func: Any) -> Any:
            return pydantic_dec(classmethod(func))

        return decorator

    def root_validator_compat(func: Any) -> Any:
        """
        Pydantic v2 branch: accepts a v1-style (cls, values: dict) -> dict
        validator and wraps it as model_validator(mode='after').
        """

        def v2_func(self: BaseModel) -> BaseModel:
            values = dict(self)
            func(type(self), values)
            return self

        return _model_validator(mode="after")(v2_func)

    def to_dict(model: BaseModel, **kwargs: Any) -> dict:
        return model.model_dump(**kwargs)

else:
    from pydantic import root_validator as _root_validator
    from pydantic import validator as _validator

    class ConfigDict(dict):  # type: ignore[no-redef]
        """Dummy ConfigDict shim for pydantic v1 — unused at runtime."""

        pass

    def field_validator(*fields: str, **kwargs: Any):  # type: ignore[misc]
        """
        Pydantic v1/v2 compatible field_validator decorator.
        Drops v2-only kwargs (e.g. mode) before delegating to v1 validator.
        """
        kwargs.pop("mode", None)
        return _validator(*fields, **kwargs)

    root_validator_compat = _root_validator  # type: ignore[assignment]

    def to_dict(model: BaseModel, **kwargs: Any) -> dict:  # type: ignore[misc]
        return model.dict(**kwargs)
