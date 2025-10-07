"""Command line interface for the clinical survival pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer
from sksurv.util import Surv

from sklearn.pipeline import Pipeline

from clinical_survival import __version__
from clinical_survival.eval import (
    EvaluationResult,
    compute_metrics,
    decision_curve_ipcw,
    evaluate_model,
    ipcw_reliability_curve,
    plot_decision,
    plot_reliability,
)
from clinical_survival.explain import explain_model
from clinical_survival.io import load_dataset
from clinical_survival.models import PipelineModel, make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.report import build_report, load_best_model
from clinical_survival.tuning import NestedCVResult, nested_cv
from clinical_survival.utils import (
    ensure_dir,
    load_yaml,
    prepare_features,
    save_json,
    set_global_seed,
    stratified_event_split,
)

app = typer.Typer(help="Clinical survival modeling pipeline")


def _load_feature_spec(path: Path) -> dict[str, list[str]]:
    if not path.exists():  # pragma: no cover - configuration error
        raise FileNotFoundError(path)
    spec = load_yaml(path)
    return spec


def _prepare_data(
    config_path: Path,
    features_path_override: Path | None = None,
    seed_override: int | None = None,
    horizons_override: list[int] | None = None,
    thresholds_override: list[float] | None = None,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame] | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, list[str]],
]:
    params = load_yaml(config_path)
    paths = params.setdefault("paths", {})
    if seed_override is not None:
        params["seed"] = seed_override
    if horizons_override is not None:
        params.setdefault("calibration", {})["times_days"] = horizons_override
        params.setdefault("decision_curve", {}).setdefault("times_days", horizons_override)
    if thresholds_override is not None:
        params.setdefault("decision_curve", {})["thresholds"] = thresholds_override

    feature_spec_path = features_path_override or Path(paths.get("features", "configs/features.yaml"))
    feature_spec = _load_feature_spec(Path(feature_spec_path))

    external_cfg = params.get("external", {}).copy()
    external_csv = paths.get("external_csv")
    if external_csv:
        external_cfg["csv"] = external_csv

    train_split, external_split, metadata = load_dataset(
        paths["data_csv"],
        paths["metadata"],
        time_col=params["time_col"],
        event_col=params["event_col"],
        external_config=external_cfg,
    )
    return train_split, external_split, metadata, params, feature_spec


def _build_pipeline_factory(
    model_name: str,
    feature_spec: dict[str, list[str]],
    missing_cfg: dict[str, Any],
    seed: int,
) -> callable[[dict[str, Any]], PipelineModel]:
    def factory(model_params: dict[str, Any]) -> PipelineModel:
        transformer = build_preprocessor(feature_spec, missing_cfg, random_state=seed)
        estimator = make_model(model_name, random_state=seed, **model_params)
        pipeline = Pipeline([("pre", transformer), ("est", estimator)])
        wrapped = PipelineModel(pipeline)
        wrapped.name = model_name
        return wrapped

    return factory


def _collect_oof_predictions(
    result: NestedCVResult,
    n_samples: int,
    times: list[float],
    sample_ids: pd.Series,
    time_col: str,
    event_col: str,
    y_train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    risk = np.zeros(n_samples, dtype=float)
    survival = np.zeros((n_samples, len(times)), dtype=float)
    for fold in result.folds:
        idx = fold.test_indices
        risk[idx] = fold.risk
        survival[idx, :] = fold.survival

    oof_df = pd.DataFrame(
        {
            "sample_index": np.arange(n_samples),
            "sample_id": sample_ids,
            "time": y_train_df[time_col].reset_index(drop=True),
            "event": y_train_df[event_col].reset_index(drop=True),
            "risk": risk,
        }
    )
    for t_idx, horizon in enumerate(times):
        oof_df[f"surv@{int(horizon)}"] = survival[:, t_idx]
    return oof_df, risk, survival


def _save_fold_predictions(result: NestedCVResult, cv_dir: Path, times: list[float]) -> None:
    ensure_dir(cv_dir)
    for fold in result.folds:
        fold_df = pd.DataFrame({
            "sample_index": fold.test_indices,
            "risk": fold.risk,
        })
        for t_idx, horizon in enumerate(times):
            fold_df[f"surv@{int(horizon)}"] = fold.survival[:, t_idx]
        fold_df.to_csv(cv_dir / f"fold_{fold.fold}.csv", index=False)


def _metrics_to_json(metrics: dict[str, Any], path: Path) -> None:
    payload = {name: interval.as_dict() for name, interval in metrics.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.callback()
def main(version: bool = typer.Option(False, "--version", help="Show version and exit")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def load(
    data: Path = typer.Option(Path("data/toy/toy_survival.csv"), exists=True),  # noqa: B008
    meta: Path = typer.Option(Path("data/toy/metadata.yaml"), exists=True),  # noqa: B008
    time_col: str = typer.Option("time"),  # noqa: B008
    event_col: str = typer.Option("event"),  # noqa: B008
) -> None:
    train_split, external_split, metadata = load_dataset(
        data,
        meta,
        time_col=time_col,
        event_col=event_col,
    )
    X_train, _ = train_split
    summary: dict[str, Any] = {
        "train_rows": len(X_train),
        "train_columns": list(X_train.columns),
        "metadata": metadata,
    }
    if external_split is not None:
        X_external, _ = external_split
        summary["external_rows"] = len(X_external)
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command()
def train(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    grid: Path = typer.Option(Path("configs/model_grid.yaml"), exists=True),  # noqa: B008
    features_yaml: Path | None = typer.Option(None, help="Override features YAML"),  # noqa: B008
    seed: int | None = typer.Option(None, help="Override random seed"),  # noqa: B008
    horizons: list[int] | None = typer.Option(None, help="Override evaluation horizons (days)"),  # noqa: B008
    thresholds: list[float] | None = typer.Option(None, help="Override decision thresholds"),  # noqa: B008
) -> None:
    train_split, external_split, metadata, params, feature_spec = _prepare_data(
        config,
        features_path_override=features_yaml,
        seed_override=seed,
        horizons_override=horizons,
        thresholds_override=thresholds,
    )
    grid_config = load_yaml(grid)

    eval_times = list(map(float, params.get("calibration", {}).get("times_days", [90, 180, 365])))
    decision_thresholds = params.get("decision_curve", {}).get("thresholds", [0.05, 0.1, 0.2, 0.3])
    seed_value = int(params.get("seed", 42))
    bootstrap_reps = int(params.get("evaluation", {}).get("bootstrap", 200))

    set_global_seed(seed_value)
    outdir = ensure_dir(params["paths"]["outdir"])
    artifacts_dir = ensure_dir(outdir / "artifacts")
    models_root = ensure_dir(artifacts_dir / "models")
    metrics_dir = ensure_dir(artifacts_dir / "metrics")
    cv_root = ensure_dir(metrics_dir / "cv")

    X_train_raw, y_train_df = train_split
    external_present = external_split is not None

    if external_present:
        X_eval_raw, y_eval_df = external_split
    else:
        combined = pd.concat([X_train_raw, y_train_df], axis=1)
        train_df, holdout_df = stratified_event_split(
            combined,
            params["event_col"],
            params.get("test_split", 0.2),
            seed_value,
        )
        X_train_raw = train_df.drop(columns=[params["time_col"], params["event_col"]])
        y_train_df = train_df[[params["time_col"], params["event_col"], *([params.get("id_col")] if params.get("id_col") in train_df.columns else [])]]
        X_eval_raw = holdout_df.drop(columns=[params["time_col"], params["event_col"]])
        y_eval_df = holdout_df[[params["time_col"], params["event_col"]]]

    X_train_features, feature_spec = prepare_features(X_train_raw, feature_spec)
    X_eval_features, _ = prepare_features(X_eval_raw, feature_spec)

    X_train_features = X_train_features.reset_index(drop=True)
    y_train_df = y_train_df.reset_index(drop=True)
    X_eval_features = X_eval_features.reset_index(drop=True)
    y_eval_df = y_eval_df.reset_index(drop=True)

    sample_ids = (
        y_train_df[params["id_col"]]
        if params.get("id_col") and params["id_col"] in y_train_df
        else pd.Series(np.arange(len(y_train_df)))
    )
    sample_ids = pd.Series(sample_ids).reset_index(drop=True)

    leaderboard_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    trained_models: dict[str, PipelineModel] = {}

    y_train_eval = y_train_df[[params["time_col"], params["event_col"]]].rename(
        columns={params["time_col"]: "time", params["event_col"]: "event"}
    )
    y_eval_eval = y_eval_df[[params["time_col"], params["event_col"]]].rename(
        columns={params["time_col"]: "time", params["event_col"]: "event"}
    )

    for model_name in params.get("models", []):
        typer.echo(f"Training {model_name} ...")
        pipeline_factory = _build_pipeline_factory(model_name, feature_spec, params.get("missing", {}), seed_value)
        result = nested_cv(
            model_name,
            X_train_features,
            y_train_df[params["time_col"]],
            y_train_df[params["event_col"]],
            params.get("n_splits", 3),
            params.get("inner_splits", 2),
            grid_config.get(model_name, {}),
            eval_times,
            random_state=seed_value,
            pipeline_builder=pipeline_factory,
        )

        trained_models[model_name] = result.estimator
        model_dir = ensure_dir(models_root / model_name)
        joblib.dump(result.estimator.pipeline, model_dir / "pipeline.joblib")

        cv_dir = ensure_dir(cv_root / model_name)
        _save_fold_predictions(result, cv_dir, eval_times)

        oof_df, oof_risk, oof_surv = _collect_oof_predictions(
            result,
            len(X_train_features),
            eval_times,
            sample_ids,
            params["time_col"],
            params["event_col"],
            y_train_df,
        )
        oof_df.to_csv(cv_dir / "oof_predictions.csv", index=False)

        calibration_dir = ensure_dir(metrics_dir / "calibration")
        decision_dir = ensure_dir(metrics_dir / "decision_curves")

        bins = int(params.get("calibration", {}).get("bins", 10))
        cv_reliability = ipcw_reliability_curve(y_train_eval, oof_surv, eval_times, bins=bins)
        cv_reliability.assign(label="cv").to_csv(calibration_dir / f"reliability_{model_name}_cv.csv", index=False)
        plot_reliability(cv_reliability.assign(label="cv"), calibration_dir / f"reliability_{model_name}_cv.png")

        cv_decision = decision_curve_ipcw(y_train_eval, oof_surv, eval_times, decision_thresholds)
        cv_decision.assign(label="cv").to_csv(decision_dir / f"net_benefit_{model_name}_cv.csv", index=False)
        plot_decision(cv_decision.assign(label="cv"), decision_dir / f"net_benefit_{model_name}_cv.png")

        oof_metrics = compute_metrics(
            y_train_eval,
            oof_risk,
            oof_surv,
            eval_times,
            bootstrap=bootstrap_reps,
            seed=seed_value,
        )
        _metrics_to_json(oof_metrics, metrics_dir / f"metrics_oof_{model_name}.json")

        leaderboard_rows.append(
            {
                "model": model_name,
                "concordance": oof_metrics["concordance"].estimate,
                "ibs": oof_metrics["ibs"].estimate,
            }
        )

        eval_label = "external" if external_present else "holdout"
        eval_result: EvaluationResult = evaluate_model(
            result.estimator,
            X_train_features,
            y_train_eval,
            X_eval_features,
            y_eval_eval,
            eval_times,
            metrics_dir,
            label=eval_label,
            thresholds=decision_thresholds,
            bins=int(params.get("calibration", {}).get("bins", 10)),
        )
        eval_metrics = {name: interval.estimate for name, interval in eval_result.metrics.items()}
        external_rows.append({"model": model_name, **eval_metrics})
        reliability_path = metrics_dir / f"calibration_{model_name}_{eval_label}.csv"
        decision_path = metrics_dir / f"decision_{model_name}_{eval_label}.csv"
        eval_result.reliability.to_csv(reliability_path, index=False)
        eval_result.decision.to_csv(decision_path, index=False)

    leaderboard_path = metrics_dir / "leaderboard.csv"
    pd.DataFrame(leaderboard_rows).to_csv(leaderboard_path, index=False)
    pd.DataFrame(external_rows).to_csv(metrics_dir / "external_summary.csv", index=False)
    save_json(metadata, artifacts_dir / "dataset_metadata.json")

    best_model_name = (
        pd.DataFrame(leaderboard_rows)
        .sort_values(by=["concordance", "ibs"], ascending=[False, True])
        .iloc[0]["model"]
    )
    save_json({"best_model": best_model_name}, metrics_dir / "best_model.json")

    final_label = params.get("external", {}).get("label", "holdout")
    final_dir = ensure_dir(models_root / final_label)
    joblib.dump(trained_models[best_model_name].pipeline, final_dir / "pipeline.joblib")

    typer.echo(f"Training complete. Leaderboard saved to {leaderboard_path}")


@app.command()
def evaluate_command(
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
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
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    model_name: str = typer.Option("coxph"),  # noqa: B008
) -> None:
    train_split, _, _, params, feature_spec = _prepare_data(config)
    X_train_raw, y_train_df = train_split
    X_features, _ = prepare_features(X_train_raw, feature_spec)
    X_features = X_features.reset_index(drop=True)
    y_struct = Surv.from_dataframe(params["event_col"], params["time_col"], y_train_df.reset_index(drop=True))

    model_path = Path(params["paths"]["outdir"]) / "artifacts" / "models" / model_name / "pipeline.joblib"
    if not model_path.exists():  # pragma: no cover - user error
        raise FileNotFoundError(model_path)
    pipeline = joblib.load(model_path)
    transformer = pipeline.named_steps["pre"]
    estimator = pipeline.named_steps["est"]
    X_transformed = transformer.transform(X_features)

    explain_dir = ensure_dir(Path(params["paths"]["outdir"]) / "artifacts" / "explain" / model_name)
    explain_paths = explain_model(
        estimator,
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
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    out: Path = typer.Option(Path("results/report.html")),  # noqa: B008
) -> None:
    params = load_yaml(config)
    outdir = Path(params["paths"]["outdir"])
    artifacts_dir = outdir / "artifacts"
    metrics_dir = artifacts_dir / "metrics"
    leaderboard_path = metrics_dir / "leaderboard.csv"
    dataset_meta_path = artifacts_dir / "dataset_metadata.json"
    dataset_meta = json.loads(dataset_meta_path.read_text()) if dataset_meta_path.exists() else {}
    best_model = load_best_model(metrics_dir)

    calibration_figs: dict[str, Path | None] = {}
    decision_figs: dict[str, Path | None] = {}
    if best_model:
        calibration_dir = metrics_dir / "calibration"
        decision_dir = metrics_dir / "decision_curves"
        calibration_figs["oof"] = calibration_dir / f"reliability_{best_model}_cv.png"
        decision_figs["oof"] = decision_dir / f"net_benefit_{best_model}_cv.png"
        external_label = "external" if (calibration_dir / f"reliability_{best_model}_external.png").exists() else "holdout"
        calibration_candidate = calibration_dir / f"reliability_{best_model}_{external_label}.png"
        decision_candidate = decision_dir / f"net_benefit_{best_model}_{external_label}.png"
        if calibration_candidate.exists():
            calibration_figs[external_label] = calibration_candidate
        if decision_candidate.exists():
            decision_figs[external_label] = decision_candidate

    shap_figs: list[Path] = []
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
    config: Path = typer.Option(Path("configs/params.yaml"), exists=True),  # noqa: B008
    grid: Path = typer.Option(Path("configs/model_grid.yaml"), exists=True),  # noqa: B008
) -> None:
    train(config=config, grid=grid)
    params = load_yaml(config)
    report_path = Path(params["paths"]["outdir"]) / "report.html"
    report(config=config, out=report_path)
    typer.echo(f"Pipeline completed -> {report_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
