"""HTML report assembly utilities."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json
import jinja2

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from clinical_survival.utils import ensure_dir, load_json
from clinical_survival.config import ParamsConfig


def generate_report(params_config: ParamsConfig) -> None:
    """Generates the final HTML report from artifacts."""

    outdir = ensure_dir(params_config.paths.outdir)
    artifacts_dir = ensure_dir(outdir / "artifacts")
    metrics_dir = ensure_dir(artifacts_dir / "metrics")

    leaderboard_path = metrics_dir / "leaderboard.csv"
    dataset_meta_path = artifacts_dir / "dataset_metadata.json"

    dataset_meta = {}
    if dataset_meta_path.exists():
        with open(dataset_meta_path, "r") as f:
            dataset_meta = json.load(f)

    leaderboard_html = ""
    if leaderboard_path.exists():
        import pandas as pd
        leaderboard_df = pd.read_csv(leaderboard_path)
        leaderboard_html = leaderboard_df.to_html(
            index=False, classes="table table-striped", justify="center"
        )

    template_path = Path("configs/report_template.html.j2")
    if not template_path.exists():
        raise FileNotFoundError(f"Report template not found at {template_path}")

    with open(template_path, "r") as f:
        template = jinja2.Template(f.read())

    rendered_html = template.render(
        leaderboard_table=leaderboard_html,
        dataset_meta=dataset_meta,
        params=params_config.model_dump(),
    )

    report_path = outdir / "report.html"
    with open(report_path, "w") as f:
        f.write(rendered_html)

    print(f"Report generated at: {report_path}")


def build_report(
    template_path: str | Path,
    leaderboard_csv: str | Path,
    dataset_meta: dict[str, object],
    output_path: str | Path,
    *,
    calibration_figs: dict[str, Path | None] | None = None,
    decision_figs: dict[str, Path | None] | None = None,
    shap_figs: Iterable[Path] | None = None,
    external_metrics_csv: str | Path | None = None,
    best_model: str | None = None,
    extra_context: dict[str, object] | None = None,
) -> Path:
    """Render the HTML report using the provided artefacts."""

    template_path = Path(template_path)
    leaderboard_path = Path(leaderboard_csv)
    leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
    external_metrics = (
        pd.read_csv(external_metrics_csv)
        if external_metrics_csv and Path(external_metrics_csv).exists()
        else pd.DataFrame()
    )

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_path.name)

    def _clean_paths(fig_map: dict[str, Path | None] | None) -> dict[str, str]:
        if not fig_map:
            return {}
        cleaned: dict[str, str] = {}
        for key, value in fig_map.items():
            if value and Path(value).exists():
                cleaned[key] = str(value)
        return cleaned

    context = {
        "dataset": dataset_meta,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "external_metrics": external_metrics.to_dict(orient="records"),
        "generated": datetime.utcnow().isoformat(),
        "calibration_figs": _clean_paths(calibration_figs),
        "decision_figs": _clean_paths(decision_figs),
        "shap_figs": [str(path) for path in shap_figs or [] if Path(path).exists()],
        "best_model": best_model,
    }
    if extra_context:
        context.update(extra_context)

    ensure_dir(Path(output_path).parent)
    html = template.render(**context)
    Path(output_path).write_text(html, encoding="utf-8")
    return Path(output_path)


def load_best_model(metrics_dir: Path) -> str | None:
    info_path = metrics_dir / "best_model.json"
    if info_path.exists():
        info = load_json(info_path)
        return info.get("best_model")
    return None
