"""Command line interface for the clinical survival pipeline."""

from __future__ import annotations

# Import the main CLI application from the modular structure
from clinical_survival.cli.main import app

# For backward compatibility, expose the app at module level
__all__ = ["app"]
