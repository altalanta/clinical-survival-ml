"""Placeholder script for exporting trained artifacts."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    results_dir = Path("results")
    print(f"Artifacts available under {results_dir.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
