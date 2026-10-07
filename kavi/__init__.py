"""Kavi — a GPT written from scratch in NumPy, with hand-derived backpropagation."""
from . import backend
from .model import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig", "backend"]
