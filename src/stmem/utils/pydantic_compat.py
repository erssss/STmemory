from __future__ import annotations

from typing import Any, Dict, Optional


class AliasChoices:
    def __init__(self, *choices: Any):
        self.choices = choices


def Field(*, default: Any = None, default_factory: Optional[Any] = None, **kwargs: Any) -> Any:
    if default_factory is not None:
        try:
            return default_factory()
        except Exception:
            return default
    return default


def field_validator(*fields: str, mode: str = "after", **kwargs: Any):
    def decorator(fn):
        return fn

    return decorator


def model_validator(*, mode: str = "after", **kwargs: Any):
    def decorator(fn):
        return fn

    return decorator


class BaseModel:
    def __init__(self, **kwargs: Any):
        annotations = getattr(self.__class__, "__annotations__", {}) or {}
        for name in annotations.keys():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif hasattr(self.__class__, name):
                setattr(self, name, getattr(self.__class__, name))
            else:
                setattr(self, name, None)

    def model_dump(self, exclude_none: bool = False, **kwargs: Any) -> Dict[str, Any]:
        data = dict(self.__dict__)
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


class BaseSettings(BaseModel):
    pass


class SettingsConfigDict(dict):
    pass


class ConfigDict(dict):
    pass
