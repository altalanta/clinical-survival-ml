"""Command line interface for the clinical survival pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
import typer
from sksurv.util import Surv

from clinical_survival import __version__
from clinical_survival.eval import evaluate
from clinical_survival.explain import explain_model
from clinical_survival.io import load_dataset
from clinical_survival.preprocess import build_transformer, fit_transform, transform
from clinical_survival.report import build_report, load_best_model
from clinical_survival.tuning import nested_cv
from clinical_survival.utils import (
    ensure_dir,
    load_json,
    load_yaml,
    prepare_features,
    save_json,
    set_global_seed,
    stratified_event_split,
)

app = typer.Typer(help="Clinical survival modeling pipeline")


@app.callback()
def main(version: bool = typer.Option(False, "--version", help="Show version and exit")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def load(
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),
    time_col: str = typer.Option("time"),
    event_col: str = typer.Option("event"),
) -> None:
    train_split, external_split, metadata = load_dataset(
        data,
        meta,
        time_col=time_col,
        event_col=event_col,
    )
    X_train, y_train = train_split
    summary: Dict[str, object] = {
        "train_rows": len(X_train),
        "train_columns": list(X_train.columns),
        "metadata": metadata,
    }
    if external_split is not None:
        X_external, _ = external_split
        summary["external_rows"] = len(X_external)
    typer.echo(json.dumps(summary, indent=2, default=str))


def _prepare_data(config_path: Path):
    params = load_yaml(config_path)
    feature_spec = load_yaml(Path("configs/features.yaml"))
    external_cfg = params.get("external", {}).copy()
    external_csv = params.get("paths", {}).get("external_csv")
    if external_csv:
        external_cfg["csv"] = external_csv
    train_split, external_split, metadata = load_dataset(
        params["paths"]["data_csv"],
        params["paths"]["metadata"],
        time_col=params["time_col"],
        event_col=params["event_col"],
        external_config=external_cfg,
    )
    return train_split, external_split, metadata, params, feature_spec


def _to_surv(target: pd.DataFrame, event_col: str, time_col: str):
    return Surv.from_arrays(target[event_col].astype(bool), target[time_col].astype(float))


@app.command()
def train(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),
    grid: Path = typer.Option(Path("configs/model_grid.yaml"), exists=True),
) -> None:
    train_split, external_split, metadata, params, feature_spec = _prepare_data(config)
    grid_config = load_yaml(grid)

    set_global_seed(params.get("seed", 42))
    outdir = ensure_dir(params["paths"]["outdir"])
    artifacts_dir = ensure_dir(outdir / "artifacts")
    models_root = ensure_dir(artifacts_dir / "models")
    metrics_dir = ensure_dir(artifacts_dir / "metrics")

    X_train_raw, y_train_df = train_split
    external_present = external_split is not None

    if external_present:
        X_eval_raw, y_eval_df = external_split  # external holdout
    else:
        combined = pd.concat([X_train_raw, y_train_df], axis=1)
        train_df, holdout_df = stratified_event_split(
            combined,
            params["event_col"],
            params.get("test_split", 0.2),
            params.get("seed", 42),
        )
        X_train_raw = train_df.drop(columns=[params["time_col"], params["event_col"]])
        y_train_df = train_df[[params["time_col"], params["event_col"]]]
        X_eval_raw = holdout_df.drop(columns=[params["time_col"], params["event_col"]])
        y_eval_df = holdout_df[[params["time_col"], params["event_col"]]]

    X_train_features, feature_spec = prepare_features(X_train_raw, feature_spec)
    transformer = build_transformer(feature_spec, **params.get("missing", {}))
    X_train, transformer = fit_transform(transformer, X_train_features)
    joblib.dump(transformer, models_root / "transformer.joblib")

    time_col = params["time_col"]
    event_col = params["event_col"]
    y_train_struct = _to_surv(y_train_df, event_col, time_col)

    # prepare evaluation set using same transformer
    X_eval_features, _ = prepare_features(X_eval_raw, feature_spec)
    X_eval = transform(transformer, X_eval_features)
    y_eval_struct = _to_surv(y_eval_df, event_col, time_col)

    times = params.get("calibration", {}).get("times_days", [90, 180, 365])
    leaderboard_rows: List[Dict[str, object]] = []
    external_rows: List[Dict[str, object]] = []
    trained_models: Dict[str, object] = {}

    for model_name in params.get("models", []):
        typer.echo(f"Training {model_name} ...")
        time_series = pd.Series(y_train_df[time_col].values, index=X_train.index)
        event_series = pd.Series(y_train_df[event_col].values, index=X_train.index)
        best_model, fold_results = nested_cv(
            model_name,
            X_train,
            time_series,
            event_series,
            params.get("n_splits", 3),
            params.get("inner_splits", 2),
            grid_config.get(model_name, {}),
            times,
            random_state=params.get("seed", 42),
        )
        trained_models[model_name] = best_model
        joblib.dump(best_model, models_root / f"{model_name}.joblib")
        pd.DataFrame([result.__dict__ for result in fold_results]).to_csv(
            metrics_dir / f"cv_{model_name}.csv",
            index=False,
        )

        cv_eval = evaluate(
            best_model,
            model_name,
            X_train,
            y_train_struct,
            X_train,
            y_train_struct,
            times,
            metrics_dir,
            label="cv",
        )
        cv_metrics = {k: v for k, v in cv_eval.metrics.items() if k != "label"}
        leaderboard_rows.append({"model": model_name, **cv_metrics})

        eval_label = "external" if external_present else "holdout"
        eval_result = evaluate(
            best_model,
            model_name,
            X_train,
            y_train_struct,
            X_eval,
            y_eval_struct,
            times,
            metrics_dir,
            label=eval_label,
        )
        ext_metrics = {k: v for k, v in eval_result.metrics.items() if k != "label"}
        external_rows.append({"model": model_name, **ext_metrics})


    leaderboard_df = pd.DataFrame(leaderboard_rows)
    leaderboard_path = metrics_dir / "leaderboard.csv"
    if not leaderboard_df.empty:
        leaderboard_df.to_csv(leaderboard_path, index=False)
        typer.echo(f"Saved leaderboard to {leaderboard_path}")

    external_summary_path = metrics_dir / "external_summary.csv"
    if external_rows:
        pd.DataFrame(external_rows).to_csv(external_summary_path, index=False)

    save_json(metadata, artifacts_dir / "dataset_metadata.json")

    if not leaderboard_df.empty:
        best_model_row = leaderboard_df.sort_values(
            by=["concordance", "ibs"], ascending=[False, True]
        ).iloc[0]
        best_model_name = best_model_row["model"]
        save_json({"best_model": best_model_name}, metrics_dir / "best_model.json")

        external_label = params.get("external", {}).get("label", "holdout")
        final_model_dir = ensure_dir(models_root / external_label)
        joblib.dump(transformer, final_model_dir / "transformer.joblib")
        joblib.dump(trained_models[best_model_name], final_model_dir / f"{best_model_name}.joblib")


@app.command()
def evaluate_command(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),
) -> None:
    params = load_yaml(config)
    metrics_dir = Path(params["paths"]["outdir"]) / "artifacts" / "metrics"
    leaderboard_path = metrics_dir / "leaderboard.csv"
    external_path = metrics_dir / "external_summary.csv"
    if leaderboard_path.exists():
        typer.echo("Leaderboard:\n" + leaderboard_path.read_text())
    if external_path.exists():
        typer.echo("\nExternal validation:\n" + external_path.read_text())


@app.command()
def explain(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),
    model_name: str = typer.Option("coxph"),
) -> None:
    train_split, _, _, params, feature_spec = _prepare_data(config)
    X_train_raw, y_train_df = train_split
    X_features, feature_spec = prepare_features(X_train_raw, feature_spec)

    outdir = Path(params["paths"]["outdir"])
    artifacts_dir = outdir / "artifacts"
    models_root = artifacts_dir / "models"
    transformer_path = models_root / "transformer.joblib"

    if transformer_path.exists():
        transformer = joblib.load(transformer_path)
        X_transformed = transform(transformer, X_features)
    else:
        transformer = build_transformer(feature_spec, **params.get("missing", {}))
        X_transformed, transformer = fit_transform(transformer, X_features)

    y_struct = _to_surv(y_train_df, params["event_col"], params["time_col"])

    model = joblib.load(models_root / f"{model_name}.joblib")
    explain_dir = ensure_dir(artifacts_dir / "explain" / model_name)
    explain_paths = explain_model(
        model,
        X_transformed,
        y_struct,
        params.get("calibration", {}).get("times_days", [365]),
        params.get("explain", {}).get("shap_samples", 200),
        params.get("explain", {}).get("pdp_features", []),
        explain_dir,
    )
    typer.echo(json.dumps({k: [str(p) for p in v if p] for k, v in explain_paths.items()}, indent=2))


@app.command()
def report(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),
    out: Path = typer.Option(Path("results/report.html")),
) -> None:
    params = load_yaml(config)
    outdir = Path(params["paths"]["outdir"])
    artifacts_dir = outdir / "artifacts"
    metrics_dir = artifacts_dir / "metrics"
    leaderboard_path = metrics_dir / "leaderboard.csv"
    dataset_meta = load_json(artifacts_dir / "dataset_metadata.json") if (artifacts_dir / "dataset_metadata.json").exists() else {}
    best_model = load_best_model(metrics_dir)

    calibration_figs: Dict[str, Optional[Path]] = {}
    decision_figs: Dict[str, Optional[Path]] = {}
    if best_model:
        calibration_dir = metrics_dir / "calibration"
        decision_dir = metrics_dir / "decision_curves"
        calibration_figs["cv"] = calibration_dir / f"reliability_{best_model}_cv.png"
        for label in ("external", "holdout"):
            candidate = calibration_dir / f"reliability_{best_model}_{label}.png"
            if candidate.exists():
                calibration_figs["external"] = candidate
                break
        decision_figs["cv"] = decision_dir / f"net_benefit_{best_model}_cv.png"
        for label in ("external", "holdout"):
            candidate = decision_dir / f"net_benefit_{best_model}_{label}.png"
            if candidate.exists():
                decision_figs["external"] = candidate
                break

    shap_figs: List[Path] = []
    if best_model:
        explain_dir = artifacts_dir / "explain" / best_model
        if explain_dir.exists():
            shap_figs.extend(sorted(explain_dir.glob("*.png")))

    build_report(
        Path("configs/report_template.html.j2"),
        leaderboard_path,
        dataset_meta,
        out,
        calibration_figs=calibration_figs,
        decision_figs=decision_figs,
        shap_figs=shap_figs,
        external_metrics_csv=metrics_dir / "external_summary.csv",
        best_model=best_model,
    )
    typer.echo(f"Report written to {out}")


@app.command()
def run(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),
    grid: Path = typer.Option(Path("configs/model_grid.yaml"), exists=True),
) -> None:
    train(config=config, grid=grid)
    params = load_yaml(config)
    report_path = Path(params["paths"]["outdir"]) / "report.html"
    report(config=config, out=report_path)
    typer.echo(f"Pipeline completed -> {report_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
