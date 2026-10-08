"""SurgMLLM training, conversion, inference, and evaluation package."""

__all__ = ["SurgMLLMModel"]


def __getattr__(name):
    if name == "SurgMLLMModel":
        from .models import SurgMLLMModel

        return SurgMLLMModel
    raise AttributeError(name)
