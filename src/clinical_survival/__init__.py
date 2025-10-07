"""Clinical Survival ML package."""

from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development installations
    __version__ = "0.1.0"

__all__ = [
    "__version__",
]
