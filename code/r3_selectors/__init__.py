"""Frozen-input selectors for the October 2026 R3 experiment."""
from .models import ModelConfig, build_model

def load_checkpoint(*args, **kwargs):
    from .pipeline import load_checkpoint as implementation
    return implementation(*args, **kwargs)


def predict_arrays(*args, **kwargs):
    from .pipeline import predict_arrays as implementation
    return implementation(*args, **kwargs)


def predict_npz(*args, **kwargs):
    from .pipeline import predict_npz as implementation
    return implementation(*args, **kwargs)

__all__ = ["ModelConfig", "build_model", "load_checkpoint", "predict_arrays", "predict_npz"]
