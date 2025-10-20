"""CLI command implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.pipeline import Pipeline
from sksurv.util import Surv

from clinical_survival import __version__
from clinical_survival.config_validation import (
    print_validation_errors,
    validate_all_configs,
)
from clinical_survival.eval import EvaluationResult, compute_metrics, evaluate_model
from clinical_survival.metrics import (
    decision_curve_ipcw,
    ipcw_reliability_curve,
    plot_calibration_curve,
    plot_decision_curve,
    calibration_summary,
)
from clinical_survival.explain import explain_model
from clinical_survival.io import load_dataset
from clinical_survival.logging_utils import (
    format_success_message,
    log_error_with_context,
    log_function_call,
    setup_logging,
)
from clinical_survival.models import PipelineModel, make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.monitoring import ModelMonitor, PerformanceTracker
from clinical_survival.report import build_report, load_best_model
from clinical_survival.tuning import NestedCVResult, nested_cv
from clinical_survival.incremental_learning import (
    IncrementalLearningManager,
    IncrementalUpdateConfig,
    create_incremental_learner,
    load_incremental_learning_config,
)
from clinical_survival.distributed import (
    DistributedBenchmarker,
    DistributedClient,
    DistributedConfig,
    DistributedDataset,
    DistributedEvaluator,
    DistributedMetrics,
    DistributedTrainer,
    create_distributed_config,
    load_distributed_config,
)
from clinical_survival.clinical_interpretability import (
    ClinicalDecisionSupport,
    create_clinical_interpretability_config,
    save_clinical_report,
)
from clinical_survival.mlops import (
    ABTestManager,
    AutomatedRetrainer,
    DeploymentEnvironment,
    DeploymentManager,
    ModelRegistry,
    ModelVersion,
    RetrainingTrigger,
    create_mlops_config,
    initialize_mlops_system,
    load_mlops_config,
)
from clinical_survival.data_quality import (
    DataCleansingPipeline,
    DataQualityMonitor,
    DataQualityProfiler,
    DataQualityReport,
    DataQualityMetrics,
    DataValidator,
    ValidationRule,
    create_data_quality_config,
    create_validation_rules,
    save_data_quality_report,
)
from clinical_survival.utils import (
    ensure_dir,
    load_json,
    load_yaml,
    prepare_features,
    save_json,
    set_global_seed,
    stratified_event_split,
)


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
    try:
        params = load_yaml(config_path)
        log_function_call("load_yaml", {"path": str(config_path)})
    except Exception as e:
        log_error_with_context(e, f"loading config {config_path}")
        raise typer.Exit(f"❌ Failed to load configuration file {config_path}: {e}") from e

    paths = params.setdefault("paths", {})

    # Validate required paths exist
    for required_path in ["data_csv", "metadata", "outdir"]:
        if required_path not in paths:
            raise typer.Exit(f"❌ Required path '{required_path}' not found in configuration")

    if seed_override is not None:
        params["seed"] = seed_override
    if horizons_override is not None:
        params.setdefault("calibration", {})["times_days"] = horizons_override
        params.setdefault("decision_curve", {}).setdefault("times_days", horizons_override)
    if thresholds_override is not None:
        params.setdefault("decision_curve", {})["thresholds"] = thresholds_override

    feature_spec_path = features_path_override or Path(
        paths.get("features", "configs/features.yaml")
    )
    feature_spec = _load_feature_spec(Path(feature_spec_path))

    external_cfg = params.get("external", {}).copy()
    external_csv = paths.get("external_csv")
    if external_csv:
        external_cfg["csv"] = external_csv

    try:
        train_split, external_split, metadata = load_dataset(
            paths["data_csv"],
            paths["metadata"],
            time_col=params["time_col"],
            event_col=params["event_col"],
            external_config=external_cfg,
        )
        log_function_call(
            "load_dataset",
            {
                "data_csv": paths["data_csv"],
                "metadata": paths["metadata"],
                "time_col": params["time_col"],
                "event_col": params["event_col"],
            },
        )
    except Exception as e:
        log_error_with_context(e, "loading dataset")
        raise typer.Exit(f"❌ Failed to load dataset: {e}") from e

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
        fold_df = pd.DataFrame(
            {
                "sample_index": fold.test_indices,
                "risk": fold.risk,
            }
        )
        for t_idx, horizon in enumerate(times):
            fold_df[f"surv@{int(horizon)}"] = fold.survival[:, t_idx]
        fold_df.to_csv(cv_dir / f"fold_{fold.fold}.csv", index=False)


def _metrics_to_json(metrics: dict[str, Any], path: Path) -> None:
    payload = {name: interval.as_dict() for name, interval in metrics.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def setup_main_callback(
    version: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """Setup the main CLI callback with logging configuration."""
    # Setup logging based on verbosity options
    setup_logging(verbose=verbose, debug=debug)

    if version:
        typer.echo(__version__)
        raise typer.Exit()


def run_load_command(
    data: Path,
    meta: Path,
    time_col: str,
    event_col: str,
) -> None:
    """Run the load command."""
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


def run_train_command(
    config: Path,
    grid: Path,
    features_yaml: Path | None,
    seed: int | None,
    horizons: list[int] | None,
    thresholds: list[float] | None,
) -> None:
    """Run the train command."""
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
        y_train_df = train_df[
            [
                params["time_col"],
                params["event_col"],
                *([params.get("id_col")] if params.get("id_col") in train_df.columns else []),
            ]
        ]
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

        # Handle ensemble model names specially for logging
        display_name = model_name
        model_params = grid_config.get(model_name, {})
        if model_name == "stacking" and "base_models" in model_params:
            base_models = (
                model_params["base_models"][0] if model_params["base_models"] else ["coxph", "rsf"]
            )
            display_name = f"{model_name}({'+'.join(base_models)})"
        elif model_name == "bagging" and "base_model" in model_params:
            base_model = model_params["base_model"][0] if model_params["base_model"] else "rsf"
            display_name = f"{model_name}({base_model})"
        elif model_name == "dynamic" and "base_models" in model_params:
            base_models = (
                model_params["base_models"][0] if model_params["base_models"] else ["coxph", "rsf"]
            )
            display_name = f"{model_name}({'+'.join(base_models)})"

        log_function_call(
            "nested_cv",
            {
                "model": model_name,
                "display_name": display_name,
                "n_samples": len(X_train_features),
                "n_splits": params.get("n_splits", 3),
                "inner_splits": params.get("inner_splits", 2),
            },
        )

        group_series = (
            y_train_df[params["id_col"]]
            if params.get("id_col") and params["id_col"] in y_train_df
            else None
        )

        try:
            pipeline_factory = _build_pipeline_factory(
                model_name, feature_spec, params.get("missing", {}), seed_value
            )
            result = nested_cv(
                model_name,
                X_train_features,
                y_train_df[params["time_col"]],
                y_train_df[params["event_col"]],
                params.get("n_splits", 3),
                params.get("inner_splits", 2),
                model_params,
                eval_times,
                random_state=seed_value,
                group_ids=group_series,
                pipeline_builder=pipeline_factory,
            )
        except Exception as e:
            log_error_with_context(e, f"training {model_name}")
            typer.echo(f"❌ Failed to train {model_name}: {e}")
            # For ensemble models, provide more specific error guidance
            if model_name in ["stacking", "bagging", "dynamic"]:
                typer.echo(
                    "💡 Tip: Check that base models are properly configured in model_grid.yaml"
                )
                typer.echo(f"   Current config: {model_params}")
            continue

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
        cv_reliability.assign(label="cv").to_csv(
            calibration_dir / f"reliability_{model_name}_cv.csv", index=False
        )
        plot_calibration_curve(
            cv_reliability.assign(label="cv"),
            calibration_dir / f"reliability_{model_name}_cv.png",
        )

        cv_decision = decision_curve_ipcw(y_train_eval, oof_surv, eval_times, decision_thresholds)
        cv_decision.assign(label="cv").to_csv(
            decision_dir / f"net_benefit_{model_name}_cv.csv", index=False
        )
        plot_decision_curve(
            cv_decision.assign(label="cv"),
            decision_dir / f"net_benefit_{model_name}_cv.png",
        )

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
            bootstrap=bootstrap_reps,
            seed=seed_value,
        )
        eval_metrics = {name: interval.estimate for name, interval in eval_result.metrics.items()}
        external_rows.append({"model": model_name, **eval_metrics})
        reliability_path = metrics_dir / f"calibration_{model_name}_{eval_label}.csv"
        decision_path = metrics_dir / f"decision_{model_name}_{eval_label}.csv"
        eval_result.reliability.to_csv(reliability_path, index=False)
        eval_result.decision.to_csv(decision_path, index=False)
        _metrics_to_json(
            eval_result.metrics,
            metrics_dir / f"metrics_{eval_label}_{model_name}.json",
        )

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

    typer.echo(
        format_success_message(f"Training complete. Leaderboard saved to {leaderboard_path}")
    )
    typer.echo(f"Best model: {best_model_name}")
    typer.echo(f"Results available in: {outdir}")


def run_evaluate_command(
    config: Path,
    *,
    report: Path | None = None,
    competing_risks: str = "none",
) -> None:
    """Run the evaluate command."""
    if competing_risks.lower() not in {"none", "finegray"}:
        raise typer.Exit("❌ Invalid value for --competing-risks. Choose from none, finegray")
    if competing_risks.lower() == "finegray":
        raise typer.Exit(
            "❌ Fine–Gray competing risks is not yet available. Pass --competing-risks none."
        )
    params = load_yaml(config)
    metrics_dir = Path(params["paths"]["outdir"]) / "artifacts" / "metrics"
    leaderboard_path = metrics_dir / "leaderboard.csv"
    external_path = metrics_dir / "external_summary.csv"
    if leaderboard_path.exists():
        typer.echo("Leaderboard:\n" + leaderboard_path.read_text())
    if external_path.exists():
        typer.echo("\nExternal validation:\n" + external_path.read_text())
    if report is not None:
        run_report_command(config=config, out=report)


def run_explain_command(config: Path, model_name: str) -> None:
    """Run the explain command."""
    train_split, _, _, params, feature_spec = _prepare_data(config)
    X_train_raw, y_train_df = train_split
    X_features, _ = prepare_features(X_train_raw, feature_spec)
    X_features = X_features.reset_index(drop=True)
    y_struct = Surv.from_dataframe(
        params["event_col"], params["time_col"], y_train_df.reset_index(drop=True)
    )

    model_path = (
        Path(params["paths"]["outdir"]) / "artifacts" / "models" / model_name / "pipeline.joblib"
    )
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
    typer.echo(
        json.dumps({k: [str(p) for p in v if p] for k, v in explain_paths.items()}, indent=2)
    )


def run_report_command(config: Path, out: Path) -> None:
    """Run the report command."""
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
        calibration_figs["cv"] = calibration_dir / f"reliability_{best_model}_cv.png"
        decision_figs["cv"] = decision_dir / f"net_benefit_{best_model}_cv.png"
        external_label = (
            "external"
            if (calibration_dir / f"reliability_{best_model}_external.png").exists()
            else "holdout"
        )
        calibration_candidate = calibration_dir / f"reliability_{best_model}_{external_label}.png"
        decision_candidate = decision_dir / f"net_benefit_{best_model}_{external_label}.png"
        if calibration_candidate.exists():
            calibration_figs[external_label] = calibration_candidate
        if decision_candidate.exists():
            decision_figs[external_label] = decision_candidate

    metrics_payload: dict[str, Any] = {}
    if best_model:
        oof_metrics_path = metrics_dir / f"metrics_oof_{best_model}.json"
        if oof_metrics_path.exists():
            metrics_payload["oof"] = load_json(oof_metrics_path)
        eval_label = params.get("external", {}).get("label", "holdout")
        holdout_metrics_path = metrics_dir / f"metrics_{eval_label}_{best_model}.json"
        if holdout_metrics_path.exists():
            metrics_payload[eval_label] = load_json(holdout_metrics_path)

    calibration_summary_df = pd.DataFrame()
    decision_curve_records: list[dict[str, Any]] = []
    if best_model:
        eval_label = params.get("external", {}).get("label", "holdout")
        calibration_csv = metrics_dir / f"calibration_{best_model}_{eval_label}.csv"
        decision_csv = metrics_dir / f"decision_{best_model}_{eval_label}.csv"
        if calibration_csv.exists():
            calibration_summary_df = calibration_summary(pd.read_csv(calibration_csv))
        if decision_csv.exists():
            decision_curve_records = pd.read_csv(decision_csv).to_dict(orient="records")

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
        extra_context={
            "metrics": metrics_payload,
            "calibration_summary": calibration_summary_df.to_dict(orient="records")
            if not calibration_summary_df.empty
            else [],
            "decision_curve_points": decision_curve_records,
        },
    )
    typer.echo(f"Report written to {out}")


def run_validate_config_command(config: Path, grid: Path, features: Path) -> None:
    """Run the validate-config command."""
    typer.echo("🔍 Validating configuration files...")

    errors = validate_all_configs(config, grid, features)

    if any(file_errors for file_errors in errors.values()):
        print_validation_errors(errors)
        typer.echo("\n💡 Tip: Check the configuration documentation for correct parameter values")
        raise typer.Exit(1)
    else:
        typer.echo(format_success_message("All configuration files are valid!"))


def run_run_command(config: Path, grid: Path) -> None:
    """Run the run command (train + report)."""
    run_train_command(
        config=config, grid=grid, features_yaml=None, seed=None, horizons=None, thresholds=None
    )
    params = load_yaml(config)
    report_path = Path(params["paths"]["outdir"]) / "report.html"
    run_report_command(config=config, out=report_path)
    typer.echo(f"Pipeline completed -> {report_path}")


def run_monitor_command(
    config: Path,
    data: Path,
    meta: Path,
    model_name: str | None = None,
    batch_size: int = 100,
    save_monitoring: bool = True,
) -> None:
    """Monitor model predictions for drift and performance."""
    typer.echo("🔍 Monitoring model predictions for drift detection...")

    # Load configuration and data
    params = load_yaml(config)
    train_split, _, _, _, feature_spec = _prepare_data(config)

    X_train_raw, y_train_df = train_split
    X_features, _ = prepare_features(X_train_raw, feature_spec)

    # Load additional data for monitoring
    if data != Path("data/toy/toy_survival.csv"):  # If not using default toy data
        monitor_data = pd.read_csv(data)
        if meta != Path("data/toy/metadata.yaml"):
            # Apply metadata transformations if needed
            pass
        monitor_features, _ = prepare_features(monitor_data.drop(columns=["time", "event"]), feature_spec)
        monitor_targets = monitor_data[["time", "event"]]
    else:
        # Use training data for demonstration
        monitor_features = X_features
        monitor_targets = y_train_df[["time", "event"]]

    # Initialize monitor
    models_dir = Path(params["paths"]["outdir"]) / "artifacts" / "models"
    monitor = ModelMonitor(models_dir)

    # Load existing monitoring data
    monitor.load_monitoring_data()

    # Determine which model to monitor
    if model_name is None:
        # Find best model from leaderboard
        leaderboard_path = Path(params["paths"]["outdir"]) / "artifacts" / "metrics" / "leaderboard.csv"
        if leaderboard_path.exists():
            leaderboard = pd.read_csv(leaderboard_path)
            model_name = leaderboard.loc[leaderboard["concordance"].idxmax(), "model"]
        else:
            model_name = "coxph"  # Default fallback

    typer.echo(f"📊 Monitoring model: {model_name}")

    # Process data in batches for monitoring
    n_samples = len(monitor_features)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        batch_features = monitor_features.iloc[start_idx:end_idx]
        batch_targets = monitor_targets.iloc[start_idx:end_idx]

        typer.echo(f"  Processing batch {i + 1}/{n_batches} ({len(batch_features)} samples)")

        try:
            # Load model for prediction
            model_path = models_dir / model_name / "pipeline.joblib"
            if not model_path.exists():
                typer.echo(f"❌ Model {model_name} not found at {model_path}")
                continue

            import joblib
            pipeline = joblib.load(model_path)
            model = pipeline.named_steps["est"]

            # Make predictions
            risk_pred = model.predict_risk(batch_features)

            # For survival predictions, we'd need time horizons from config
            eval_times = params.get("calibration", {}).get("times_days", [365])
            survival_pred = model.predict_survival_function(batch_features, eval_times)

            # Record monitoring metrics
            metrics = monitor.record_metrics(
                model_name=model_name,
                X=batch_features,
                y_true=batch_targets,
                y_pred=risk_pred,
                survival_pred=survival_pred,
                eval_times=eval_times,
            )

            # Show current metrics
            typer.echo(f"    Concordance: {metrics.concordance:.3f}")
            typer.echo(f"    Brier Score: {metrics.brier_score:.3f}")

            if metrics.drift_scores:
                max_drift = max(metrics.drift_scores.values())
                typer.echo(f"    Max Drift Score: {max_drift:.3f}")

        except Exception as e:
            typer.echo(f"❌ Error monitoring batch {i + 1}: {e}")
            continue

    # Save monitoring data if requested
    if save_monitoring:
        monitor.save_monitoring_data()
        typer.echo("💾 Monitoring data saved")

    # Show recent alerts
    recent_alerts = monitor.get_recent_alerts(model_name, days=1)
    if recent_alerts:
        typer.echo(f"\n🚨 Recent alerts for {model_name}:")
        for alert in recent_alerts:
            typer.echo(f"  [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")
    else:
        typer.echo(f"\n✅ No recent alerts for {model_name}")


def run_drift_command(
    config: Path,
    model_name: str | None = None,
    days: int = 7,
    show_details: bool = False,
) -> None:
    """Check for model drift and performance degradation."""
    typer.echo("🔍 Checking for model drift and performance issues...")

    params = load_yaml(config)
    models_dir = Path(params["paths"]["outdir"]) / "artifacts" / "models"

    # Initialize monitor
    monitor = ModelMonitor(models_dir)
    monitor.load_monitoring_data()

    # Determine which model to check
    if model_name is None:
        # Check all models with monitoring data
        models_to_check = [model for model in monitor.metrics_history.keys()]
    else:
        models_to_check = [model_name] if model_name in monitor.metrics_history else []

    if not models_to_check:
        typer.echo("❌ No monitoring data found. Run monitoring first.")
        return

    # Check each model for drift and performance issues
    for model in models_to_check:
        typer.echo(f"\n📊 Model: {model}")

        # Get performance summary
        summary = monitor.get_performance_summary(model, days=days)

        if "error" in summary:
            typer.echo(f"  ❌ {summary['error']}")
            continue

        # Show key metrics
        typer.echo(f"  Observations: {summary['n_observations']}")
        typer.echo(f"  Samples: {summary['total_samples']}")
        typer.echo(f"  Concordance: {summary['concordance']['mean']".3f"} ± {summary['concordance']['std']:".3f"")

        if "brier_score" in summary:
            typer.echo(f"  Brier Score: {summary['brier_score']['mean']".3f"} ± {summary['brier_score']['std']:".3f"")

        # Show trend
        concordance_trend = summary["concordance"]["trend"]
        trend_emoji = {"improving": "📈", "degrading": "📉", "stable": "➡️"}.get(concordance_trend, "❓")
        typer.echo(f"  Trend: {trend_emoji} {concordance_trend}")

        # Show drift scores
        if "latest_drift_scores" in summary:
            drift_scores = summary["latest_drift_scores"]
            if drift_scores:
                max_drift = max(drift_scores.values())
                typer.echo(f"  Max Drift Score: {max_drift:.3f}")

                if show_details:
                    for feature, score in sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                        typer.echo(f"    {feature}: {score:.3f}")

        # Show alerts
        alert_count = summary.get("latest_alerts", 0)
        if alert_count > 0:
            typer.echo(f"  🚨 Active alerts: {alert_count}")

        # Recommendations
        if concordance_trend == "degrading" or (summary.get("brier_score", {}).get("trend") == "degrading"):
            typer.echo("  💡 Recommendation: Consider model retraining")
        elif max_drift > 0.2 if "latest_drift_scores" in summary else False:
            typer.echo("  💡 Recommendation: Review data collection process")
        else:
            typer.echo("  ✅ Model performance appears stable")


def run_monitoring_status_command(config: Path) -> None:
    """Show overall monitoring status for all models."""
    typer.echo("📊 Monitoring Status Dashboard")
    typer.echo("=" * 40)

    params = load_yaml(config)
    models_dir = Path(params["paths"]["outdir"]) / "artifacts" / "models"

    # Initialize monitor
    monitor = ModelMonitor(models_dir)
    monitor.load_monitoring_data()

    if not monitor.metrics_history:
        typer.echo("❌ No monitoring data available")
        typer.echo("💡 Run 'clinical-ml monitor' to start monitoring model predictions")
        return

    # Show summary for each model
    for model_name in monitor.metrics_history.keys():
        summary = monitor.get_performance_summary(model_name, days=7)

        if "error" not in summary:
            status_emoji = "✅"
            if summary["concordance"]["trend"] == "degrading":
                status_emoji = "⚠️"
            elif summary.get("brier_score", {}).get("trend") == "degrading":
                status_emoji = "⚠️"

            typer.echo(f"\n{status_emoji} {model_name}")
            typer.echo(f"  Concordance: {summary['concordance']['mean']:.3f}")
            typer.echo(f"  Trend: {summary['concordance']['trend']}")
            typer.echo(f"  Observations: {summary['n_observations']}")

            # Show recent alerts
            recent_alerts = monitor.get_recent_alerts(model_name, days=1)
            if recent_alerts:
                typer.echo(f"  Alerts: {len(recent_alerts)}")

    # Show overall alerts
    all_alerts = monitor.get_recent_alerts(days=7)
    if all_alerts:
        typer.echo(f"\n🚨 Total active alerts: {len(all_alerts)}")

        # Group by severity
        severity_counts = {}
        for alert in all_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        for severity, count in severity_counts.items():
            emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}.get(severity, "❓")
            typer.echo(f"  {emoji} {severity.capitalize()}: {count}")


def run_reset_monitoring_command(
    config: Path,
    model_name: str | None = None,
    confirm: bool = False,
) -> None:
    """Reset monitoring baselines for a model or all models."""
    if not confirm:
        typer.echo("⚠️  This will reset all monitoring baselines and historical data.")
        typer.echo("💡 Use --confirm to proceed with the reset.")
        return

    typer.echo("🔄 Resetting monitoring baselines...")

    params = load_yaml(config)
    models_dir = Path(params["paths"]["outdir"]) / "artifacts" / "models"

    # Initialize monitor
    monitor = ModelMonitor(models_dir)
    monitor.load_monitoring_data()

    # Reset baselines
    monitor.reset_baseline(model_name)

    # Clear monitoring data if resetting all models
    if model_name is None:
        monitor.metrics_history.clear()
        monitor.alerts.clear()

    # Save changes
    monitor.save_monitoring_data()

    typer.echo("✅ Monitoring baselines reset successfully")

    if model_name:
        typer.echo(f"💡 Start fresh monitoring for {model_name} with new baseline data")
    else:
        typer.echo("💡 All monitoring data cleared. Start fresh monitoring for all models")


@log_function_call
def run_automl_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    time_limit: int,
    model_types: list[str],
    metric: str,
    output_dir: Path,
) -> None:
    """Run automated model selection and hyperparameter optimization."""
    try:
        import optuna
        from clinical_survival.automl import AutoSurvivalML
    except ImportError as e:
        typer.echo(f"❌ AutoML dependencies not available: {e}")
        typer.echo("💡 Install with: pip install clinical-survival-ml[automl]")
        raise typer.Exit(1) from e

    typer.echo("🚀 Starting AutoML optimization...")
    typer.echo(f"⏱️  Time limit: {time_limit} seconds")
    typer.echo(f"🎯 Metric: {metric}")
    typer.echo(f"🤖 Models: {', '.join(model_types)}")
    typer.echo(f"📁 Output: {output_dir}")

    # Load configuration
    params = load_yaml(config_path)
    features_config = load_yaml(meta_path)

    # Load and prepare data
    typer.echo("📊 Loading dataset...")
    X, y, features = load_dataset(data_path, meta_path)

    # Prepare features
    feature_spec = prepare_features(features_config, features)
    X_processed = build_preprocessor(
        feature_spec,
        params["missing"],
        random_state=params["seed"]
    ).fit_transform(X)

    # Convert to DataFrame for AutoML
    X_df = pd.DataFrame(X_processed, columns=feature_spec.get("numeric", []) + feature_spec.get("categorical", []))

    # Initialize AutoML
    automl = AutoSurvivalML(
        time_limit=time_limit,
        metric=metric,
        random_state=params["seed"],
        cv_folds=params["n_splits"]
    )

    # Run optimization
    typer.echo("🔬 Running Bayesian optimization...")
    automl.fit(X_df, y, model_types=model_types)

    # Get best model and results
    best_params = automl.get_best_params()
    best_score = automl.best_score_
    best_model_type = automl.best_model_type_

    typer.echo("✅ Optimization completed!")
    typer.echo(f"🏆 Best model: {best_model_type}")
    typer.echo(f"📈 Best score ({metric}): {best_score:.4f}")
    typer.echo(f"⚙️  Best parameters: {json.dumps(best_params, indent=2)}")

    # Save results
    ensure_dir(output_dir)
    results_path = output_dir / "automl_results.json"
    study_path = output_dir / "optuna_study.pkl"

    # Save optimization results
    results = {
        "best_model_type": best_model_type,
        "best_score": best_score,
        "best_params": best_params,
        "optimization_time": time_limit,
        "metric": metric,
        "model_types": model_types,
        "cv_folds": params["n_splits"]
    }

    save_json(results, results_path)
    joblib.dump(automl.study_, study_path)

    typer.echo(f"💾 Results saved to {results_path}")
    typer.echo(f"💾 Study saved to {study_path}")

    # Train best model on full dataset and save
    typer.echo("🎓 Training best model on full dataset...")
    best_model = automl.get_best_model(X_df, y)
    model_path = output_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    typer.echo(f"💾 Best model saved to {model_path}")
    typer.echo("🎉 AutoML optimization completed successfully!")


@log_function_call
def run_benchmark_hardware_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    model_type: str,
    use_gpu: bool,
    gpu_id: int,
) -> None:
    """Benchmark hardware performance and check GPU availability."""
    try:
        from clinical_survival.gpu_utils import create_gpu_accelerator
        from clinical_survival.models import make_model
    except ImportError as e:
        typer.echo(f"❌ GPU utilities not available: {e}")
        raise typer.Exit(1) from e

    typer.echo("🔧 Hardware Benchmark & GPU Detection")
    typer.echo("=" * 50)

    # Load configuration and data
    params = load_yaml(config_path)
    features_config = load_yaml(meta_path)

    # Load and prepare data
    typer.echo("📊 Loading dataset for benchmarking...")
    X, y, features = load_dataset(data_path, meta_path)

    # Prepare features for benchmarking
    feature_spec = prepare_features(features_config, features)
    X_processed = build_preprocessor(
        feature_spec,
        params["missing"],
        random_state=params["seed"]
    ).fit_transform(X)

    # Convert to DataFrame for models
    X_df = pd.DataFrame(X_processed, columns=feature_spec.get("numeric", []) + feature_spec.get("categorical", []))

    # Create GPU accelerator
    typer.echo("🔍 Detecting hardware capabilities...")
    gpu_accelerator = create_gpu_accelerator(use_gpu=use_gpu, gpu_id=gpu_id)

    # Display hardware info
    optimal_config = gpu_accelerator.get_optimal_config()
    typer.echo("\n🖥️  Hardware Configuration:")
    typer.echo(f"   • CPU cores available: {optimal_config['n_jobs']}")
    typer.echo(f"   • CUDA available: {optimal_config['cuda_available']}")
    typer.echo(f"   • GPU available: {optimal_config['gpu_available']}")

    if optimal_config['cuda_available']:
        typer.echo(f"   • GPU count: {optimal_config.get('gpu_count', 0)}")
        if optimal_config.get('gpu_name'):
            typer.echo(f"   • GPU name: {optimal_config['gpu_name']}")
        if optimal_config.get('gpu_memory_gb'):
            typer.echo(f"   • GPU memory: {optimal_config['gpu_memory_gb']:.".1f"GB")

    typer.echo(f"   • XGBoost GPU support: {optimal_config['xgb_gpu_available']}")

    # Benchmark model training
    typer.echo("
🏃 Running performance benchmarks..."    typer.echo(f"   Model: {model_type}")

    def create_benchmark_model():
        return make_model(
            model_type,
            random_state=params["seed"],
            use_gpu=use_gpu,
            gpu_id=gpu_id
        )

    # Run benchmark
    try:
        benchmark_results = gpu_accelerator.benchmark_hardware(
            create_benchmark_model, X_df, y
        )

        typer.echo("
⏱️  Benchmark Results:"        typer.echo(f"   • CPU training time: {benchmark_results['cpu_time']:.".3f" seconds")

        if "gpu_time" in benchmark_results:
            typer.echo(f"   • GPU training time: {benchmark_results['gpu_time']:.".3f" seconds")
            typer.echo(f"   • Speedup: {benchmark_results['speedup']:.".2f"x")
        elif "gpu_error" in benchmark_results:
            typer.echo(f"   • GPU benchmark failed: {benchmark_results['gpu_error']}")

    except Exception as e:
        typer.echo(f"   • Benchmark failed: {e}")

    # Recommendations
    typer.echo("
💡 Recommendations:"    if optimal_config['xgb_gpu_available'] and use_gpu:
        typer.echo("   • ✅ GPU acceleration is recommended for XGBoost models")
    elif optimal_config['cuda_available']:
        typer.echo("   • ⚠️  GPU available but XGBoost may not have GPU support")
        typer.echo("   • 💡 Consider installing XGBoost with GPU support")
    else:
        typer.echo("   • ✅ CPU-only mode is optimal for this hardware")

    typer.echo(f"   • 💡 Use --gpu-id {optimal_config.get('recommended_gpu_id', 0)} for best performance")

    typer.echo("
🎯 Hardware detection completed!"


@log_function_call
def run_counterfactual_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    model_name: str,
    target_risk: float | None,
    target_time: float | None,
    n_counterfactuals: int,
    method: str,
    output_dir: Path,
) -> None:
    """Generate counterfactual explanations for model predictions."""
    try:
        from clinical_survival.counterfactual import create_counterfactual_explainer
        from clinical_survival.models import make_model
    except ImportError as e:
        typer.echo(f"❌ Counterfactual utilities not available: {e}")
        raise typer.Exit(1) from e

    typer.echo("🔍 Generating Counterfactual Explanations")
    typer.echo("=" * 50)
    typer.echo(f"🎯 Model: {model_name}")
    typer.echo(f"🔢 Counterfactuals: {n_counterfactuals}")
    typer.echo(f"⚙️  Method: {method}")

    if target_risk is not None:
        typer.echo(f"📊 Target risk: {target_risk}")
    if target_time is not None:
        typer.echo(f"⏰ Target survival time: {target_time} days")

    # Load configuration and data
    params = load_yaml(config_path)
    features_config = load_yaml(meta_path)

    # Load and prepare data
    typer.echo("📊 Loading dataset...")
    X, y, features = load_dataset(data_path, meta_path)

    # Prepare features
    feature_spec = prepare_features(features_config, features)
    X_processed = build_preprocessor(
        feature_spec,
        params["missing"],
        random_state=params["seed"]
    ).fit_transform(X)

    # Convert to DataFrame
    X_df = pd.DataFrame(X_processed, columns=feature_spec.get("numeric", []) + feature_spec.get("categorical", []))

    # Load the best model (assuming it was trained previously)
    typer.echo("🤖 Loading trained model...")

    # Try to load from results directory
    models_dir = Path(params["paths"]["outdir"]) / "artifacts" / "models"
    model_path = models_dir / f"{model_name}.pkl"

    if model_path.exists():
        model = joblib.load(model_path)
        typer.echo(f"✅ Loaded model from {model_path}")
    else:
        typer.echo(f"⚠️  Model not found at {model_path}, training new model...")
        # Train a simple model for demonstration
        model = make_model(
            model_name,
            random_state=params["seed"],
            use_gpu=params.get("use_gpu", True),
            gpu_id=params.get("gpu_id", 0)
        )
        model = model.fit(X_df, y)
        typer.echo("✅ Model trained successfully")

    # Create counterfactual explainer
    typer.echo("🔬 Setting up counterfactual explainer...")

    # Define feature ranges (simplified for demo)
    feature_ranges = {}
    for feature in X_df.columns:
        if X_df[feature].dtype in ['int64', 'float64']:
            feature_ranges[feature] = (X_df[feature].min(), X_df[feature].max())

    explainer = create_counterfactual_explainer(
        model=model,
        feature_names=list(X_df.columns),
        feature_ranges=feature_ranges,
        method=method,
        random_state=params["seed"]
    )

    # Generate counterfactual explanations
    typer.echo("🎯 Generating counterfactuals...")

    if target_risk is not None:
        explanation = explainer.explain_prediction(
            X_df.iloc[:1],  # First sample
            target_risk=target_risk,
            n_counterfactuals=n_counterfactuals
        )
    elif target_time is not None:
        explanation = explainer.explain_prediction(
            X_df.iloc[:1],  # First sample
            time_targets=[target_time],
            n_counterfactuals=n_counterfactuals
        )
    else:
        # Generate default explanation with multiple risk targets
        explanation = explainer.explain_prediction(
            X_df.iloc[:1],  # First sample
            n_counterfactuals=n_counterfactuals
        )

    # Display results
    typer.echo("✅ Counterfactual generation completed!")
    typer.echo(f"📈 Original risk: {explanation['original_prediction']['risk']".3f"}")

    if explanation["counterfactuals"]:
        typer.echo(f"\n🎯 Generated {len(explanation['counterfactuals'])} counterfactuals:")

        for i, cf in enumerate(explanation["counterfactuals"], 1):
            cf_type = cf.get("type", "unknown")
            if "target_risk" in cf:
                typer.echo(f"\n  {i}. Target risk: {cf['target_risk']".3f"}")
            elif "target_time" in cf:
                typer.echo(f"\n  {i}. Target time: {cf['target_time']} days")

            typer.echo(f"     Distance: {cf['distance']".3f"}")
            typer.echo("     Key changes:")

            # Show top 3 most significant changes
            changes = cf["changes"]
            sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

            for feature, change in sorted_changes:
                if abs(change) > 0.01:  # Only show meaningful changes
                    direction = "↑" if change > 0 else "↓"
                    typer.echo(f"       {feature}: {direction} {abs(change)".3f"}")

        # Summary statistics
        if "summary" in explanation:
            summary = explanation["summary"]
            typer.echo("
📊 Summary:"            typer.echo(f"   Total counterfactuals: {summary['total_counterfactuals']}")
            typer.echo(f"   Average distance: {summary['avg_distance']".3f"}")
            typer.echo(f"   Min/Max distance: {summary['min_distance']".3f"} / {summary['max_distance']".3f"}")
    else:
        typer.echo("❌ No valid counterfactuals found within constraints")

    # Save results
    ensure_dir(output_dir)
    results_path = output_dir / "counterfactual_results.json"

    # Convert explanation to serializable format
    serializable_explanation = {}
    for key, value in explanation.items():
        if key == "counterfactuals":
            serializable_explanation[key] = []
            for cf in value:
                cf_copy = cf.copy()
                # Convert DataFrame to dict if present
                if "counterfactual" in cf_copy and hasattr(cf_copy["counterfactual"], "to_dict"):
                    cf_copy["counterfactual"] = cf_copy["counterfactual"].to_dict()
                serializable_explanation[key].append(cf_copy)
        else:
            serializable_explanation[key] = value

    save_json(serializable_explanation, results_path)

    typer.echo(f"\n💾 Results saved to {results_path}")
    typer.echo("🎉 Counterfactual explanation generation completed!")


def run_update_models_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    models_dir: Path,
    incremental_config_path: Path | None = None,
    model_names: list[str] | None = None,
    force_update: bool = False,
) -> None:
    """Update trained models with new data using incremental learning."""
    log_function_call("run_update_models_command", {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "meta_path": str(meta_path),
        "models_dir": str(models_dir),
        "incremental_config_path": str(incremental_config_path) if incremental_config_path else None,
        "model_names": model_names,
        "force_update": force_update
    })

    typer.echo("🔄 Starting incremental model updates...")

    # Load configuration
    params = load_yaml(config_path)

    # Load incremental learning configuration
    if incremental_config_path:
        incremental_config = load_incremental_learning_config(incremental_config_path)
    else:
        # Use default configuration
        incremental_config = IncrementalUpdateConfig()

    # Load new data
    typer.echo("📊 Loading new data...")
    try:
        X, y = load_dataset(data_path, meta_path, params["time_col"], params["event_col"])
        typer.echo(f"✅ Loaded {len(X)} samples")
    except Exception as e:
        log_error_with_context(e, f"loading dataset {data_path}")
        raise typer.Exit(f"❌ Failed to load dataset: {e}") from e

    # Initialize incremental learning manager
    manager = IncrementalLearningManager(models_dir, incremental_config)

    # Update specified models or all available models
    if model_names:
        models_to_update = model_names
    else:
        # Find all available models
        models_to_update = []
        for model_file in models_dir.glob("*.pkl"):
            model_name = model_file.stem
            if model_name not in models_to_update:
                models_to_update.append(model_name)

    typer.echo(f"🔧 Updating models: {models_to_update}")

    updated_models = []
    for model_name in models_to_update:
        typer.echo(f"\n📈 Processing model: {model_name}")

        try:
            # Check if model exists
            model_path = models_dir / f"{model_name}.pkl"
            if not model_path.exists():
                typer.echo(f"⚠️  Model {model_name} not found at {model_path}")
                continue

            # Load the model
            model = joblib.load(model_path)

            # Add to incremental learning manager if not already there
            if model_name not in manager.learners:
                manager.add_model_for_incremental_learning(model_name, model)

            # Process new data for this model
            success = manager.process_new_data(model_name, X, y)

            if success:
                typer.echo(f"✅ Successfully updated model {model_name}")
                updated_models.append(model_name)
            else:
                typer.echo(f"⚠️  No update needed for model {model_name}")

        except Exception as e:
            log_error_with_context(e, f"updating model {model_name}")
            typer.echo(f"❌ Failed to update model {model_name}: {e}")
            continue

    # Save all learners
    manager.save_all_learners()

    typer.echo(f"\n📊 Update summary: {len(updated_models)}/{len(models_to_update)} models updated")
    typer.echo("🎉 Incremental model updates completed!")


def run_incremental_status_command(
    models_dir: Path,
    model_names: list[str] | None = None,
) -> None:
    """Show status of incremental learning for models."""
    log_function_call("run_incremental_status_command", {
        "models_dir": str(models_dir),
        "model_names": model_names
    })

    typer.echo("📊 Checking incremental learning status...")

    # Use default incremental learning configuration
    incremental_config = IncrementalUpdateConfig()

    # Initialize manager
    manager = IncrementalLearningManager(models_dir, incremental_config)

    # Get status for specified models or all models
    if model_names:
        models_to_check = model_names
    else:
        models_to_check = list(manager.learners.keys())

    if not models_to_check:
        typer.echo("ℹ️  No models found for incremental learning")
        return

    typer.echo(f"\n🔍 Status for {len(models_to_check)} models:")

    for model_name in models_to_check:
        status = manager.get_model_update_status(model_name)

        typer.echo(f"\n📈 Model: {model_name}")
        typer.echo(f"   Status: {status.get('status', 'unknown')}")
        typer.echo(f"   Buffer size: {status.get('buffer_size', 0)} samples")
        typer.echo(f"   Total updates: {status.get('total_updates', 0)}")

        last_update = status.get('last_update')
        if last_update:
            typer.echo(f"   Last update: {last_update}")

        config = status.get('config', {})
        if config:
            typer.echo(f"   Min samples for update: {config.get('min_samples_for_update', 'N/A')}")
            typer.echo(f"   Max samples in memory: {config.get('max_samples_in_memory', 'N/A')}")
            typer.echo(f"   Update strategy: {config.get('update_strategy', 'N/A')}")

    typer.echo("\n✅ Incremental learning status check completed!")


def run_configure_incremental_command(
    config_path: Path,
    update_frequency_days: int = 7,
    min_samples_for_update: int = 50,
    max_samples_in_memory: int = 1000,
    update_strategy: str = "online",
    drift_detection_enabled: bool = True,
    create_backup_before_update: bool = True,
) -> None:
    """Configure incremental learning settings."""
    log_function_call("run_configure_incremental_command", {
        "config_path": str(config_path),
        "update_frequency_days": update_frequency_days,
        "min_samples_for_update": min_samples_for_update,
        "max_samples_in_memory": max_samples_in_memory,
        "update_strategy": update_strategy,
        "drift_detection_enabled": drift_detection_enabled,
        "create_backup_before_update": create_backup_before_update
    })

    typer.echo("⚙️  Configuring incremental learning settings...")

    # Create configuration
    config = {
        "update_frequency_days": update_frequency_days,
        "min_samples_for_update": min_samples_for_update,
        "max_samples_in_memory": max_samples_in_memory,
        "update_strategy": update_strategy,
        "window_size_days": 365,
        "performance_threshold": 0.02,
        "max_updates_per_model": 10,
        "drift_detection_enabled": drift_detection_enabled,
        "drift_threshold": 0.1,
        "create_backup_before_update": create_backup_before_update,
        "backup_retention_days": 30
    }

    # Save configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    typer.echo(f"✅ Incremental learning configuration saved to {config_path}")
    typer.echo("\n📋 Configuration summary:"    typer.echo(f"   Update frequency: {update_frequency_days} days")
    typer.echo(f"   Min samples for update: {min_samples_for_update}")
    typer.echo(f"   Max samples in memory: {max_samples_in_memory}")
    typer.echo(f"   Update strategy: {update_strategy}")
    typer.echo(f"   Drift detection: {'enabled' if drift_detection_enabled else 'disabled'}")
    typer.echo(f"   Backup before update: {'enabled' if create_backup_before_update else 'disabled'}")

    typer.echo("\n🎉 Incremental learning configuration completed!")


def run_distributed_benchmark_command(
    config_path: Path,
    cluster_type: str = "local",
    n_workers: int = 4,
    dataset_sizes: list[int] = [1000, 5000, 10000],
    model_type: str = "coxph",
    output_dir: Path = Path("results/distributed_benchmark"),
) -> None:
    """Benchmark distributed computing performance across different dataset sizes."""
    log_function_call("run_distributed_benchmark_command", {
        "config_path": str(config_path),
        "cluster_type": cluster_type,
        "n_workers": n_workers,
        "dataset_sizes": dataset_sizes,
        "model_type": model_type,
        "output_dir": str(output_dir)
    })

    typer.echo("🚀 Starting distributed computing benchmark...")

    # Load configuration
    params = load_yaml(config_path)

    # Create distributed configuration
    dist_config = create_distributed_config(
        cluster_type=cluster_type,
        n_workers=n_workers
    )

    # Initialize distributed client
    client = DistributedClient(dist_config)
    if not client.initialize():
        typer.echo("❌ Failed to initialize distributed client")
        return

    try:
        # Create benchmarker
        benchmarker = DistributedBenchmarker(client, dist_config)

        # Run benchmark
        typer.echo(f"📊 Benchmarking {len(dataset_sizes)} dataset sizes...")
        typer.echo(f"🔧 Cluster: {cluster_type} with {n_workers} workers")

        results = benchmarker.benchmark_scaling(
            dataset_sizes=dataset_sizes,
            model_factory=lambda **kwargs: make_model(model_type, **kwargs),
            model_params={"random_state": params["seed"]}
        )

        if not results:
            typer.echo("❌ No benchmark results generated")
            return

        # Analyze scaling efficiency
        analysis = benchmarker.analyze_scaling_efficiency(results)

        # Save results
        ensure_dir(output_dir)
        results_file = output_dir / "benchmark_results.json"

        # Convert metrics to serializable format
        serializable_results = {}
        for size, metrics in results.items():
            serializable_results[size] = {
                "total_time": metrics.total_time,
                "computation_time": metrics.computation_time,
                "communication_time": metrics.communication_time,
                "n_tasks_completed": metrics.n_tasks_completed,
                "n_tasks_failed": metrics.n_tasks_failed,
                "speedup_factor": metrics.speedup_factor
            }

        output_data = {
            "benchmark_results": serializable_results,
            "scaling_analysis": analysis,
            "configuration": {
                "cluster_type": cluster_type,
                "n_workers": n_workers,
                "dataset_sizes": dataset_sizes,
                "model_type": model_type
            }
        }

        save_json(output_data, results_file)

        # Display results
        typer.echo(f"\n📈 Benchmark Results (saved to {results_file}):")

        for size in sorted(results.keys()):
            metrics = results[size]
            typer.echo(f"   Dataset size {size","}:")
            typer.echo(f"     Total time: {metrics.total_time".2f"}s")
            typer.echo(f"     Speedup: {metrics.speedup_factor".2f"}x")
            typer.echo(f"     Tasks: {metrics.n_tasks_completed} completed, {metrics.n_tasks_failed} failed")

        # Display scaling analysis
        typer.echo("
📊 Scaling Analysis:"        if "r_squared" in analysis:
            typer.echo(f"   R²: {analysis['r_squared']".3f"}")
            typer.echo(f"   Scaling coefficient: {analysis['scaling_coefficient']".3f"}")
            typer.echo(f"   Trend: {analysis.get('ideal_scaling', 'unknown')}")

        typer.echo(f"   Efficiency trend: {analysis.get('efficiency_trend', 'unknown')}")

        typer.echo("\n🎉 Distributed benchmark completed!")

    finally:
        client.shutdown()


def run_distributed_train_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    cluster_type: str = "local",
    n_workers: int = 4,
    n_partitions: int = 10,
    model_type: str = "coxph",
    output_dir: Path = Path("results/distributed_training"),
) -> None:
    """Train model using distributed computing."""
    log_function_call("run_distributed_train_command", {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "meta_path": str(meta_path),
        "cluster_type": cluster_type,
        "n_workers": n_workers,
        "n_partitions": n_partitions,
        "model_type": model_type,
        "output_dir": str(output_dir)
    })

    typer.echo("🚀 Starting distributed training...")

    # Load configuration
    params = load_yaml(config_path)

    # Create distributed configuration
    dist_config = create_distributed_config(
        cluster_type=cluster_type,
        n_workers=n_workers,
        n_partitions=n_partitions
    )

    # Initialize distributed client
    client = DistributedClient(dist_config)
    if not client.initialize():
        typer.echo("❌ Failed to initialize distributed client")
        return

    try:
        # Load and partition data
        typer.echo("📊 Loading and partitioning data...")
        dataset = DistributedDataset(client, data_path, dist_config)

        if not dataset.load_and_partition():
            typer.echo("❌ Failed to load and partition data")
            return

        typer.echo(f"✅ Data loaded and partitioned into {dataset.get_n_partitions()} partitions")

        # Create trainer
        trainer = DistributedTrainer(client, dist_config)

        # Train model
        typer.echo(f"🔧 Training {model_type} model on {dataset.get_n_partitions()} partitions...")

        model, metrics = trainer.train_distributed(
            dataset,
            model_factory=lambda **kwargs: make_model(model_type, **kwargs),
            model_params={"random_state": params["seed"]}
        )

        typer.echo("✅ Training completed!"        typer.echo(f"   Total time: {metrics.total_time".2f"}s")
        typer.echo(f"   Computation time: {metrics.computation_time".2f"}s")
        typer.echo(f"   Communication time: {metrics.communication_time".2f"}s")
        typer.echo(f"   Tasks completed: {metrics.n_tasks_completed}")
        typer.echo(f"   Tasks failed: {metrics.n_tasks_failed}")
        typer.echo(f"   Speedup factor: {metrics.speedup_factor".2f"}x")

        # Save model and metrics
        ensure_dir(output_dir)

        # Save model
        model_file = output_dir / f"distributed_{model_type}_model.pkl"
        import joblib
        joblib.dump(model, model_file)
        typer.echo(f"💾 Model saved to {model_file}")

        # Save metrics
        metrics_file = output_dir / "distributed_metrics.json"
        metrics_data = {
            "total_time": metrics.total_time,
            "computation_time": metrics.computation_time,
            "communication_time": metrics.communication_time,
            "memory_usage_peak": metrics.memory_usage_peak,
            "cpu_usage_avg": metrics.cpu_usage_avg,
            "n_tasks_completed": metrics.n_tasks_completed,
            "n_tasks_failed": metrics.n_tasks_failed,
            "data_transfer_volume": metrics.data_transfer_volume,
            "speedup_factor": metrics.speedup_factor
        }
        save_json(metrics_data, metrics_file)
        typer.echo(f"📊 Metrics saved to {metrics_file}")

        typer.echo("\n🎉 Distributed training completed!")

    finally:
        client.shutdown()


def run_distributed_evaluate_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    model_path: Path,
    cluster_type: str = "local",
    n_workers: int = 4,
    n_partitions: int = 10,
    metrics: list[str] = ["concordance", "ibs"],
) -> None:
    """Evaluate model using distributed computing."""
    log_function_call("run_distributed_evaluate_command", {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "meta_path": str(meta_path),
        "model_path": str(model_path),
        "cluster_type": cluster_type,
        "n_workers": n_workers,
        "n_partitions": n_partitions,
        "metrics": metrics
    })

    typer.echo("🚀 Starting distributed evaluation...")

    # Load configuration
    params = load_yaml(config_path)

    # Create distributed configuration
    dist_config = create_distributed_config(
        cluster_type=cluster_type,
        n_workers=n_workers,
        n_partitions=n_partitions
    )

    # Initialize distributed client
    client = DistributedClient(dist_config)
    if not client.initialize():
        typer.echo("❌ Failed to initialize distributed client")
        return

    try:
        # Load model
        typer.echo(f"📥 Loading model from {model_path}...")
        import joblib
        model = joblib.load(model_path)
        typer.echo("✅ Model loaded")

        # Load and partition data
        typer.echo("📊 Loading and partitioning data...")
        dataset = DistributedDataset(client, data_path, dist_config)

        if not dataset.load_and_partition():
            typer.echo("❌ Failed to load and partition data")
            return

        typer.echo(f"✅ Data loaded and partitioned into {dataset.get_n_partitions()} partitions")

        # Create evaluator
        evaluator = DistributedEvaluator(client, dist_config)

        # Evaluate model
        typer.echo(f"🔧 Evaluating model on {dataset.get_n_partitions()} partitions...")
        typer.echo(f"📊 Metrics: {', '.join(metrics)}")

        results = evaluator.evaluate_distributed(dataset, model, metrics)

        # Display results
        typer.echo("\n📈 Evaluation Results:"        for metric, value in results.items():
            typer.echo(f"   {metric}: {value".4f"}")

        typer.echo("\n🎉 Distributed evaluation completed!")

    finally:
        client.shutdown()


def run_configure_distributed_command(
    config_path: Path,
    cluster_type: str = "local",
    n_workers: int = 4,
    threads_per_worker: int = 2,
    memory_per_worker: str = "2GB",
    partition_strategy: str = "balanced",
    n_partitions: int = 10,
) -> None:
    """Configure distributed computing settings."""
    log_function_call("run_configure_distributed_command", {
        "config_path": str(config_path),
        "cluster_type": cluster_type,
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_per_worker": memory_per_worker,
        "partition_strategy": partition_strategy,
        "n_partitions": n_partitions
    })

    typer.echo("⚙️  Configuring distributed computing settings...")

    # Create configuration
    config = {
        "cluster_type": cluster_type,
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_per_worker": memory_per_worker,
        "partition_strategy": partition_strategy,
        "n_partitions": n_partitions,
        "scheduler_address": "127.0.0.1:8786",
        "dashboard_address": "127.0.0.1:8787",
        "chunk_size": 1000,
        "optimize_memory": True,
        "use_gpu_if_available": True,
        "retry_failed_tasks": True,
        "max_retries": 3,
        "timeout_minutes": 60,
        "resource_allocation_strategy": "balanced"
    }

    # Save configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    typer.echo(f"✅ Distributed computing configuration saved to {config_path}")
    typer.echo("\n📋 Configuration summary:"    typer.echo(f"   Cluster type: {cluster_type}")
    typer.echo(f"   Workers: {n_workers}")
    typer.echo(f"   Threads per worker: {threads_per_worker}")
    typer.echo(f"   Memory per worker: {memory_per_worker}")
    typer.echo(f"   Partition strategy: {partition_strategy}")
    typer.echo(f"   Number of partitions: {n_partitions}")

    typer.echo("\n🎉 Distributed computing configuration completed!")


def run_clinical_interpret_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    model_path: Path,
    output_dir: Path = Path("results/clinical_interpretability"),
    patient_ids: list[str] | None = None,
    risk_thresholds: dict[str, float] | None = None,
    include_detailed_analysis: bool = True,
    output_format: str = "html",
) -> None:
    """Generate comprehensive clinical interpretability report."""
    log_function_call("run_clinical_interpret_command", {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "meta_path": str(meta_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "patient_ids": patient_ids,
        "risk_thresholds": risk_thresholds,
        "include_detailed_analysis": include_detailed_analysis,
        "output_format": output_format
    })

    typer.echo("🩺 Starting clinical interpretability analysis...")

    # Load configuration
    params = load_yaml(config_path)

    # Load model
    typer.echo(f"📥 Loading model from {model_path}...")
    import joblib
    model = joblib.load(model_path)
    typer.echo("✅ Model loaded")

    # Load data for interpretation
    typer.echo("📊 Loading data for interpretation...")
    try:
        X, y = load_dataset(data_path, meta_path, params["time_col"], params["event_col"])
        typer.echo(f"✅ Loaded {len(X)} samples for interpretation")
    except Exception as e:
        log_error_with_context(e, f"loading dataset {data_path}")
        raise typer.Exit(f"❌ Failed to load dataset: {e}") from e

    # Filter to specific patients if requested
    if patient_ids:
        X = X[X.index.isin(patient_ids)]
        typer.echo(f"🔍 Filtered to {len(X)} specified patients")

    if len(X) == 0:
        typer.echo("❌ No patients to analyze")
        return

    # Create clinical context
    clinical_context = {
        "feature_domains": {
            "age": "demographics",
            "sex": "demographics",
            "sofa": "vitals",
            "stage": "comorbidities",
            "creatinine": "labs",
            "bilirubin": "labs"
        },
        "risk_categories": {
            "protective": ["age", "sex"],
            "risk_factor": ["sofa", "stage", "creatinine", "bilirubin"],
            "neutral": []
        }
    }

    # Initialize clinical decision support system
    typer.echo("🔬 Initializing clinical decision support system...")
    cds = ClinicalDecisionSupport(model, clinical_context)

    # Use subset of data as background for SHAP
    X_background = X.sample(n=min(100, len(X)), random_state=42) if len(X) > 100 else X

    # Initialize explainers
    cds.initialize_explainers(X_background, list(X.columns))

    # Generate comprehensive clinical report
    typer.echo("🎯 Generating clinical interpretability report...")
    report = cds.generate_clinical_report(
        X,
        patient_ids=patient_ids,
        include_detailed_analysis=include_detailed_analysis
    )

    # Save report
    ensure_dir(output_dir)

    if output_format.lower() == "html":
        report_file = output_dir / "clinical_interpretability_report.html"
        save_clinical_report(report, report_file, format="html")
        typer.echo(f"💾 HTML report saved to {report_file}")
    else:
        report_file = output_dir / "clinical_interpretability_report.json"
        save_clinical_report(report, report_file, format="json")
        typer.echo(f"💾 JSON report saved to {report_file}")

    # Display summary
    summary = report["summary"]
    typer.echo("
📊 Report Summary:"    typer.echo(f"   Total patients analyzed: {summary['total_patients']}")
    typer.echo(f"   Average risk score: {summary['average_risk_score']".3f"}")
    typer.echo(f"   Risk distribution: Low={summary['risk_distribution']['low']}, "
               f"Moderate={summary['risk_distribution']['moderate']}, "
               f"High={summary['risk_distribution']['high']}, "
               f"Very High={summary['risk_distribution']['very_high']}")
    typer.echo(f"   Critical urgency patients: {summary['critical_urgency_patients']}")

    # Display population insights
    if report["population_insights"]:
        insights = report["population_insights"]
        typer.echo("
🔍 Population Insights:"        typer.echo(f"   Top risk features: {', '.join(insights.get('top_risk_features', [])[:3])}")

        if "risk_factor_prevalence" in insights:
            for feature, prevalence in list(insights["risk_factor_prevalence"].items())[:3]:
                typer.echo(f"   {feature}: {prevalence".1%"} of patients")

    typer.echo("
🎉 Clinical interpretability analysis completed!"
def run_risk_stratification_command(
    config_path: Path,
    data_path: Path,
    meta_path: Path,
    model_path: Path,
    output_dir: Path = Path("results/risk_stratification"),
    risk_thresholds: dict[str, float] | None = None,
) -> None:
    """Generate risk stratification report."""
    log_function_call("run_risk_stratification_command", {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "meta_path": str(meta_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "risk_thresholds": risk_thresholds
    })

    typer.echo("🏥 Starting risk stratification analysis...")

    # Load configuration
    params = load_yaml(config_path)

    # Load model
    typer.echo(f"📥 Loading model from {model_path}...")
    import joblib
    model = joblib.load(model_path)
    typer.echo("✅ Model loaded")

    # Load data
    typer.echo("📊 Loading data for stratification...")
    try:
        X, y = load_dataset(data_path, meta_path, params["time_col"], params["event_col"])
        typer.echo(f"✅ Loaded {len(X)} samples for stratification")
    except Exception as e:
        log_error_with_context(e, f"loading dataset {data_path}")
        raise typer.Exit(f"❌ Failed to load dataset: {e}") from e

    # Initialize risk stratifier
    from clinical_survival.clinical_interpretability import RiskStratifier

    risk_stratifier = RiskStratifier(
        model,
        risk_thresholds=risk_thresholds,
        clinical_context={}
    )

    # Generate stratifications
    typer.echo("📈 Stratifying patients by risk...")
    stratifications = risk_stratifier.stratify_patients(X)

    # Save results
    ensure_dir(output_dir)

    # Convert to serializable format
    serializable_stratifications = []
    for stratification in stratifications:
        serializable_stratifications.append({
            "patient_id": stratification.patient_id,
            "risk_score": stratification.risk_score,
            "risk_category": stratification.risk_category,
            "confidence_interval": stratification.confidence_interval,
            "primary_risk_factors": stratification.primary_risk_factors,
            "protective_factors": stratification.protective_factors,
            "clinical_recommendations": stratification.clinical_recommendations,
            "urgency_level": stratification.urgency_level
        })

    results_file = output_dir / "risk_stratification_results.json"
    save_json(serializable_stratifications, results_file)

    # Display summary
    risk_categories = [s.risk_category for s in stratifications]
    urgency_levels = [s.urgency_level for s in stratifications]

    typer.echo("
📊 Risk Stratification Summary:"    typer.echo(f"   Total patients: {len(stratifications)}")
    typer.echo(f"   Risk categories: Low={risk_categories.count('low')}, "
               f"Moderate={risk_categories.count('moderate')}, "
               f"High={risk_categories.count('high')}, "
               f"Very High={risk_categories.count('very_high')}")
    typer.echo(f"   Urgency levels: Routine={urgency_levels.count('routine')}, "
               f"Urgent={urgency_levels.count('urgent')}, "
               f"Critical={urgency_levels.count('critical')}")

    # Display recommendations
    all_recommendations = []
    for stratification in stratifications:
        all_recommendations.extend(stratification.clinical_recommendations)

    if all_recommendations:
        from collections import Counter
        common_recommendations = Counter(all_recommendations).most_common(3)
        typer.echo("
💡 Top Clinical Recommendations:"        for rec, count in common_recommendations:
            typer.echo(f"   • {rec} ({count} patients)")

    typer.echo(f"\n💾 Results saved to {results_file}")
    typer.echo("🎉 Risk stratification analysis completed!"


def run_data_quality_profile_command(
    data_path: Path,
    meta_path: Path | None = None,
    output_dir: Path = Path("results/data_quality"),
    include_anomaly_detection: bool = True,
    clinical_context: dict[str, Any] | None = None,
    output_format: str = "html",
) -> None:
    """Generate comprehensive data quality profile."""
    log_function_call("run_data_quality_profile_command", {
        "data_path": str(data_path),
        "meta_path": str(meta_path) if meta_path else None,
        "output_dir": str(output_dir),
        "include_anomaly_detection": include_anomaly_detection,
        "clinical_context": clinical_context,
        "output_format": output_format
    })

    typer.echo("🔍 Starting data quality profiling...")

    # Load data
    try:
        if meta_path:
            X, y = load_dataset(data_path, meta_path, "time", "event")
            df = X  # Use features for profiling
        else:
            df = pd.read_csv(data_path)
            # Remove time/event columns if present
            df = df.drop(columns=["time", "event"], errors="ignore")

        typer.echo(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        log_error_with_context(e, f"loading dataset {data_path}")
        raise typer.Exit(f"❌ Failed to load dataset: {e}") from e

    # Create profiler
    profiler = DataQualityProfiler()
    clinical_context = clinical_context or {}

    # Generate profile
    typer.echo("📊 Generating data quality profile...")
    report = profiler.profile_dataset(
        df,
        dataset_name=data_path.stem,
        include_anomaly_detection=include_anomaly_detection,
        clinical_context=clinical_context
    )

    # Save report
    ensure_dir(output_dir)

    if output_format.lower() == "html":
        report_file = output_dir / "data_quality_report.html"
        save_data_quality_report(report, report_file, format="html")
        typer.echo(f"💾 HTML report saved to {report_file}")
    else:
        report_file = output_dir / "data_quality_report.json"
        save_data_quality_report(report, report_file, format="json")
        typer.echo(f"💾 JSON report saved to {report_file}")

    # Display summary
    metrics = report.quality_metrics
    typer.echo("
📊 Quality Summary:"    typer.echo(f"   Overall Score: {metrics.overall_quality_score".1f"} ({metrics.quality_grade})")
    typer.echo(f"   Missing Values: {metrics.missing_values_percentage".1f"}%")
    typer.echo(f"   Duplicates: {metrics.duplicate_rows_percentage".1f"}%")
    typer.echo(f"   Issues Found: {len(report.issues_found)}")

    if report.recommendations:
        typer.echo("
💡 Recommendations:"        for rec in report.recommendations[:3]:  # Show top 3
            typer.echo(f"   • {rec}")

    typer.echo("
🎉 Data quality profiling completed!"
def run_data_validation_command(
    data_path: Path,
    meta_path: Path | None = None,
    validation_rules: list[str] | None = None,
    output_dir: Path = Path("results/data_validation"),
    strict_mode: bool = False,
    fail_on_first_error: bool = False,
) -> None:
    """Validate dataset against configured rules."""
    log_function_call("run_data_validation_command", {
        "data_path": str(data_path),
        "meta_path": str(meta_path) if meta_path else None,
        "validation_rules": validation_rules,
        "output_dir": str(output_dir),
        "strict_mode": strict_mode,
        "fail_on_first_error": fail_on_first_error
    })

    typer.echo("✅ Starting data validation...")

    # Load data
    try:
        if meta_path:
            X, y = load_dataset(data_path, meta_path, "time", "event")
            df = X  # Use features for validation
        else:
            df = pd.read_csv(data_path)
            # Remove time/event columns if present
            df = df.drop(columns=["time", "event"], errors="ignore")

        typer.echo(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        log_error_with_context(e, f"loading dataset {data_path}")
        raise typer.Exit(f"❌ Failed to load dataset: {e}") from e

    # Create validation rules
    if validation_rules == "default" or validation_rules is None:
        rules = create_validation_rules()
    else:
        # Custom rules would be loaded here
        rules = create_validation_rules()  # Placeholder

    typer.echo(f"📋 Validating against {len(rules)} rules...")

    # Create validator
    validator = DataValidator(rules)

    # Validate dataset
    result = validator.validate_dataset(df, dataset_name=data_path.stem)

    # Save validation results
    ensure_dir(output_dir)
    results_file = output_dir / "validation_results.json"
    save_json(result, results_file)

    # Display results
    typer.echo("
📊 Validation Results:"    typer.echo(f"   Status: {result['overall_status']}")
    typer.echo(f"   Total Rules: {result['total_rules']}")
    typer.echo(f"   Passed: {result['passed_rules']}")
    typer.echo(f"   Failed: {result['failed_rules']}")

    if result['errors']:
        typer.echo("
❌ Errors:"        for error in result['errors']:
            typer.echo(f"   • {error}")

    if result['warnings']:
        typer.echo("
⚠️  Warnings:"        for warning in result['warnings']:
            typer.echo(f"   • {warning}")

    if result['overall_status'] == 'failed':
        if fail_on_first_error:
            raise typer.Exit("❌ Data validation failed")
        else:
            typer.echo("
⚠️  Validation completed with errors"
    else:
        typer.echo("
✅ Validation passed!"
    typer.echo(f"💾 Results saved to {results_file}")
    typer.echo("🎉 Data validation completed!"


def run_data_cleansing_command(
    data_path: Path,
    meta_path: Path | None = None,
    output_dir: Path = Path("results/data_cleansing"),
    remove_duplicates: bool = True,
    handle_outliers: bool = True,
    preserve_original: bool = True,
) -> None:
    """Cleanse dataset based on quality assessment."""
    log_function_call("run_data_cleansing_command", {
        "data_path": str(data_path),
        "meta_path": str(meta_path) if meta_path else None,
        "output_dir": str(output_dir),
        "remove_duplicates": remove_duplicates,
        "handle_outliers": handle_outliers,
        "preserve_original": preserve_original
    })

    typer.echo("🧹 Starting data cleansing...")

    # Load data
    try:
        if meta_path:
            X, y = load_dataset(data_path, meta_path, "time", "event")
            df = X  # Use features for cleansing
        else:
            df = pd.read_csv(data_path)
            # Remove time/event columns if present
            df = df.drop(columns=["time", "event"], errors="ignore")

        typer.echo(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        log_error_with_context(e, f"loading dataset {data_path}")
        raise typer.Exit(f"❌ Failed to load dataset: {e}") from e

    # First, profile the data to understand quality issues
    typer.echo("📊 Profiling data quality...")
    profiler = DataQualityProfiler()
    quality_report = profiler.profile_dataset(df, dataset_name=data_path.stem)

    # Create cleansing pipeline
    cleansing_config = {
        "remove_duplicates": remove_duplicates,
        "missing_values_strategy": "auto",
        "handle_outliers": handle_outliers,
        "preserve_original": preserve_original
    }

    pipeline = DataCleansingPipeline(cleansing_config)

    # Cleanse data
    typer.echo("🔧 Applying data cleansing...")
    df_clean, cleansing_summary = pipeline.cleanse_dataset(df, quality_report, preserve_original)

    # Save results
    ensure_dir(output_dir)

    # Save cleaned data
    cleaned_data_path = output_dir / f"cleaned_{data_path.name}"
    df_clean.to_csv(cleaned_data_path, index=False)
    typer.echo(f"💾 Cleaned data saved to {cleaned_data_path}")

    # Save cleansing summary
    summary_file = output_dir / "cleansing_summary.json"
    save_json(cleansing_summary, summary_file)
    typer.echo(f"📊 Cleansing summary saved to {summary_file}")

    # Display results
    typer.echo("
📊 Cleansing Results:"    typer.echo(f"   Original shape: {cleansing_summary['original_shape']}")
    typer.echo(f"   Final shape: {cleansing_summary['final_shape']}")
    typer.echo(f"   Rows removed: {cleansing_summary['rows_removed']}")
    typer.echo(f"   Columns removed: {cleansing_summary['columns_removed']}")

    if cleansing_summary['cleansing_steps']:
        typer.echo("
🔧 Applied cleansing steps:"        for step in cleansing_summary['cleansing_steps']:
            typer.echo(f"   • {step}")

    typer.echo("
🎉 Data cleansing completed!"


def run_register_model_command(
    model_path: Path,
    model_name: str,
    version_number: str,
    description: str = "",
    tags: list[str] = None,
    registry_path: Path = Path("results/mlops"),
    created_by: str = "system",
) -> None:
    """Register a trained model in the MLOps registry."""
    log_function_call("run_register_model_command", {
        "model_path": str(model_path),
        "model_name": model_name,
        "version_number": version_number,
        "description": description,
        "tags": tags,
        "registry_path": str(registry_path),
        "created_by": created_by
    })

    typer.echo("📦 Registering model in MLOps registry...")

    # Load model
    typer.echo(f"📥 Loading model from {model_path}...")
    import joblib
    model = joblib.load(model_path)
    typer.echo("✅ Model loaded")

    # Initialize registry
    registry = ModelRegistry(registry_path)

    # Register model
    version = registry.register_model(
        model=model,
        model_name=model_name,
        version_number=version_number,
        created_by=created_by,
        description=description,
        tags=tags or []
    )

    typer.echo(f"✅ Model registered successfully!"    typer.echo(f"   Model: {model_name}")
    typer.echo(f"   Version: {version_number}")
    typer.echo(f"   Version ID: {version.version_id}")
    typer.echo(f"   Status: {version.status}")
    typer.echo(f"   Model type: {version.model_type}")

    typer.echo("\n🎉 Model registration completed!")


def run_mlops_status_command(
    registry_path: Path = Path("results/mlops"),
    model_name: str | None = None,
) -> None:
    """Show MLOps registry status and model versions."""
    log_function_call("run_mlops_status_command", {
        "registry_path": str(registry_path),
        "model_name": model_name
    })

    typer.echo("📊 Checking MLOps registry status...")

    # Initialize registry
    registry = ModelRegistry(registry_path)

    # Get all models or specific model
    if model_name:
        models_to_check = [model_name]
    else:
        models_to_check = list(registry._models.keys())

    if not models_to_check:
        typer.echo("ℹ️  No models found in registry")
        return

    typer.echo(f"\n🔍 Registry Status for {len(models_to_check)} models:")

    for current_model_name in models_to_check:
        typer.echo(f"\n📈 Model: {current_model_name}")

        # Get all versions
        versions = registry.get_model_versions(current_model_name)

        if not versions:
            typer.echo("   No versions found")
            continue

        typer.echo(f"   Total versions: {len(versions)}")

        # Show latest version
        latest_version = registry.get_latest_version(current_model_name)
        if latest_version:
            typer.echo(f"   Latest version: {latest_version.version_number}")
            typer.echo(f"   Status: {latest_version.status}")
            typer.echo(f"   Created: {latest_version.created_at.strftime('%Y-%m-%d %H:%M')}")

            if latest_version.performance_metrics:
                typer.echo("   Performance metrics:"                for metric, value in latest_version.performance_metrics.items():
                    typer.echo(f"     {metric}: {value".4f"}")

        # Show all versions
        typer.echo("   All versions:"        for version in versions:
            status_icon = "✅" if version.status == "production" else "🔄" if version.status == "staging" else "📋"
            typer.echo(f"     {status_icon} {version.version_number} ({version.created_at.strftime('%Y-%m-%d')}) - {version.status}")

    typer.echo("\n✅ MLOps status check completed!")


def run_deploy_model_command(
    version_id: str,
    environment_name: str,
    registry_path: Path = Path("results/mlops"),
    traffic_percentage: float = 100.0,
    approved_by: str = "system",
) -> None:
    """Deploy a model version to an environment."""
    log_function_call("run_deploy_model_command", {
        "version_id": version_id,
        "environment_name": environment_name,
        "registry_path": str(registry_path),
        "traffic_percentage": traffic_percentage,
        "approved_by": approved_by
    })

    typer.echo(f"🚀 Deploying model version {version_id} to {environment_name}...")

    # Initialize MLOps system
    config = create_mlops_config(registry_path)
    registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

    # Deploy model
    success = deployment_manager.deploy_model(
        version_id=version_id,
        environment_name=environment_name,
        traffic_percentage=traffic_percentage,
        approved_by=approved_by
    )

    if success:
        typer.echo("✅ Model deployed successfully!"        typer.echo(f"   Version ID: {version_id}")
        typer.echo(f"   Environment: {environment_name}")
        typer.echo(f"   Traffic percentage: {traffic_percentage}%")

        # Show deployment status
        status = deployment_manager.get_deployment_status(environment_name)
        if "current_version" in status:
            current = status["current_version"]
            typer.echo(f"   Deployed at: {current['deployed_at'].strftime('%Y-%m-%d %H:%M')}")
            typer.echo(f"   Version: {current['version_number']}")

    else:
        typer.echo("❌ Model deployment failed")
        raise typer.Exit(1)

    typer.echo("\n🎉 Model deployment completed!")


def run_rollback_deployment_command(
    environment_name: str,
    target_version_id: str,
    registry_path: Path = Path("results/mlops"),
    reason: str = "manual_rollback",
) -> None:
    """Rollback a deployment to a previous version."""
    log_function_call("run_rollback_deployment_command", {
        "environment_name": environment_name,
        "target_version_id": target_version_id,
        "registry_path": str(registry_path),
        "reason": reason
    })

    typer.echo(f"🔄 Rolling back deployment in {environment_name} to version {target_version_id}...")

    # Initialize MLOps system
    config = create_mlops_config(registry_path)
    registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

    # Rollback deployment
    success = deployment_manager.rollback_deployment(
        environment_name=environment_name,
        target_version_id=target_version_id,
        reason=reason
    )

    if success:
        typer.echo("✅ Deployment rollback completed successfully!"        typer.echo(f"   Environment: {environment_name}")
        typer.echo(f"   Target version: {target_version_id}")
        typer.echo(f"   Reason: {reason}")

        # Show updated deployment status
        status = deployment_manager.get_deployment_status(environment_name)
        if "current_version" in status:
            current = status["current_version"]
            typer.echo(f"   New active version: {current['version_number']}")

    else:
        typer.echo("❌ Deployment rollback failed")
        raise typer.Exit(1)

    typer.echo("\n🎉 Deployment rollback completed!")


def run_create_ab_test_command(
    test_name: str,
    model_versions: list[str],
    traffic_split: dict[str, float],
    registry_path: Path = Path("results/mlops"),
    test_duration_days: int = 14,
    success_metrics: list[str] = None,
) -> None:
    """Create an A/B test for model versions."""
    log_function_call("run_create_ab_test_command", {
        "test_name": test_name,
        "model_versions": model_versions,
        "traffic_split": traffic_split,
        "registry_path": str(registry_path),
        "test_duration_days": test_duration_days,
        "success_metrics": success_metrics
    })

    typer.echo(f"🧪 Creating A/B test '{test_name}'...")

    # Initialize MLOps system
    config = create_mlops_config(registry_path)
    registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

    # Create A/B test
    try:
        test_id = ab_test_manager.create_ab_test(
            test_name=test_name,
            model_versions=model_versions,
            traffic_split=traffic_split,
            test_duration_days=test_duration_days,
            success_metrics=success_metrics or ["concordance", "ibs"]
        )

        typer.echo("✅ A/B test created successfully!"        typer.echo(f"   Test ID: {test_id}")
        typer.echo(f"   Test name: {test_name}")
        typer.echo(f"   Model versions: {', '.join(model_versions)}")
        typer.echo(f"   Duration: {test_duration_days} days")
        typer.echo(f"   Traffic split: {traffic_split}")

        # Show traffic allocation
        typer.echo("   Traffic allocation:"        for version_id, percentage in traffic_split.items():
            version = registry.get_version(version_id)
            if version:
                typer.echo(f"     {version.version_number}: {percentage*100".1f"}%")

    except ValueError as e:
        typer.echo(f"❌ Failed to create A/B test: {e}")
        raise typer.Exit(1)

    typer.echo("\n🎉 A/B test creation completed!")


def run_ab_test_results_command(
    test_id: str,
    registry_path: Path = Path("results/mlops"),
) -> None:
    """Get results for an A/B test."""
    log_function_call("run_ab_test_results_command", {
        "test_id": test_id,
        "registry_path": str(registry_path)
    })

    typer.echo(f"📊 Getting results for A/B test {test_id}...")

    # Initialize MLOps system
    config = create_mlops_config(registry_path)
    registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

    # Get test results
    results = ab_test_manager.get_test_results(test_id)

    if "error" in results:
        typer.echo(f"❌ A/B test not found: {results['error']}")
        return

    typer.echo("✅ A/B test results:"    typer.echo(f"   Test: {results['test_name']}")
    typer.echo(f"   Status: {results['status']}")
    typer.echo(f"   Duration: {results['duration_days']} days")

    # Show version results
    if "version_results" in results:
        typer.echo("   Version performance:"        for version_id, version_data in results["version_results"].items():
            version = registry.get_version(version_id)
            if version:
                typer.echo(f"     {version.version_number}:"                typer.echo(f"       Traffic: {version.traffic_percentage*100".1f"}%")
                if version_data["performance_metrics"]:
                    for metric, value in version_data["performance_metrics"].items():
                        typer.echo(f"       {metric}: {value".4f"}")

    # Show statistical significance
    if "statistical_significance" in results:
        sig = results["statistical_significance"]
        typer.echo("   Statistical analysis:"        for metric, p_value in sig.items():
            if metric.endswith("_p_value"):
                metric_name = metric.replace("_p_value", "")
                significance = "significant" if p_value < 0.05 else "not significant"
                typer.echo(f"     {metric_name}: p-value = {p_value".4f"} ({significance})")

    typer.echo("\n🎉 A/B test results retrieved!")


def run_check_retraining_triggers_command(
    registry_path: Path = Path("results/mlops"),
    model_name: str | None = None,
) -> None:
    """Check if any retraining triggers should fire."""
    log_function_call("run_check_retraining_triggers_command", {
        "registry_path": str(registry_path),
        "model_name": model_name
    })

    typer.echo("🔄 Checking retraining triggers...")

    # Initialize MLOps system
    config = create_mlops_config(registry_path)
    registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

    # Check triggers
    triggered_models = auto_retrainer.check_triggers()

    if model_name:
        triggered_models = [m for m in triggered_models if m == model_name]

    if not triggered_models:
        typer.echo("ℹ️  No retraining triggers activated")
        return

    typer.echo(f"🚨 Retraining triggers activated for {len(triggered_models)} models:")
    for model in triggered_models:
        typer.echo(f"   • {model}")

        # Show which triggers fired
        for trigger in auto_retrainer.triggers:
            if trigger.enabled and model in auto_retrainer._get_models_for_trigger(trigger):
                typer.echo(f"     - {trigger.trigger_name} ({trigger.trigger_type})")

    typer.echo("
💡 Recommendation: Run model retraining for the triggered models"
    typer.echo("\n🎉 Trigger check completed!"


@log_function_call
def run_synthetic_data_command(
    scenario: str = typer.Option("icu", help="Type of synthetic data to generate"),
    n_samples: int = typer.Option(1000, help="Number of samples to generate"),
    output_dir: Path = typer.Option("data/synthetic", help="Output directory for generated data"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility")
) -> None:
    """Generate synthetic clinical datasets for testing."""
    from clinical_survival.testing import SyntheticDatasetGenerator

    typer.echo("🔬 Generating synthetic clinical dataset...")

    generator = SyntheticDatasetGenerator(random_state=random_state)

    if scenario == "icu":
        features, outcomes = generator.generate_icu_dataset(n_samples=n_samples)
    elif scenario == "cancer":
        features, outcomes = generator.generate_cancer_dataset(n_samples=n_samples)
    elif scenario == "cardiovascular":
        features, outcomes = generator.generate_cardiovascular_dataset(n_samples=n_samples)
    else:
        typer.echo(f"❌ Unknown scenario: {scenario}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    features_file = output_dir / f"{scenario}_features.csv"
    outcomes_file = output_dir / f"{scenario}_outcomes.csv"

    features.to_csv(features_file, index=False)
    outcomes.to_csv(outcomes_file, index=False)

    typer.echo(f"✅ Generated {n_samples} samples for {scenario} scenario")
    typer.echo(f"📁 Features saved to: {features_file}")
    typer.echo(f"📁 Outcomes saved to: {outcomes_file}")

    # Show summary statistics
    typer.echo("
📊 Dataset Summary:"    typer.echo(f"   • Features: {features.shape[1]}")
    typer.echo(f"   • Samples: {features.shape[0]}")
    typer.echo(f"   • Event rate: {outcomes['event'].mean()".2%"}")
    typer.echo(f"   • Median survival time: {outcomes['time'].median()".1f"} days")

    typer.echo("\n🎉 Synthetic data generation completed!"


@log_function_call
def run_performance_regression_command(
    config: Path = typer.Option(..., help="Configuration file"),
    baseline_file: Path = typer.Option("tests/baseline_performance.json", help="Baseline performance file"),
    tolerance: float = typer.Option(0.05, help="Performance tolerance for regression detection"),
    output_dir: Path = typer.Option("tests/performance_regression", help="Output directory for results")
) -> None:
    """Run automated performance regression testing."""
    from clinical_survival.testing import PerformanceRegressionTester
    from clinical_survival.config_validation import load_config
    from clinical_survival.models import make_model

    typer.echo("🧪 Running performance regression tests...")

    # Load configuration
    config_dict = load_config(config)

    # Load data
    data_csv = config_dict["paths"]["data_csv"]
    metadata = config_dict["paths"]["metadata"]

    X, y, metadata_dict = load_dataset(data_csv, metadata)
    typer.echo(f"📊 Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Initialize tester
    tester = PerformanceRegressionTester(baseline_file=str(baseline_file))

    # Test different models
    models_to_test = config_dict.get("models", ["coxph", "rsf", "xgb_cox"])

    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models_to_test:
        typer.echo(f"\n🔬 Testing {model_name}...")

        try:
            # Create model constructor
            def model_constructor():
                return make_model(model_name, **config_dict.get("model_params", {}).get(model_name, {}))

            # Run regression test
            result = tester.run_regression_test(
                model_constructor=model_constructor,
                X=X,
                y=y,
                test_name=f"{model_name}_regression",
                tolerance=tolerance
            )

            results[model_name] = result

            if result.passed:
                typer.echo(f"✅ {model_name}: PASSED (concordance: {result.value".3f"})")
            else:
                typer.echo(f"❌ {model_name}: FAILED (concordance: {result.value".3f"}, threshold: {result.threshold".3f"})")
                if result.error_message:
                    typer.echo(f"   Error: {result.error_message}")

        except Exception as e:
            typer.echo(f"❌ {model_name}: ERROR - {str(e)}")
            results[model_name] = TestResult(
                test_name=f"{model_name}_error",
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=0.0,
                error_message=str(e)
            )

    # Save results
    results_file = output_dir / "regression_results.json"
    results_dict = {
        model_name: {
            "passed": result.passed,
            "value": result.value,
            "threshold": result.threshold,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
        for model_name, result in results.items()
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Summary
    passed_tests = sum(1 for r in results.values() if r.passed)
    total_tests = len(results)

    typer.echo("
📋 Regression Test Summary:"    typer.echo(f"   • Passed: {passed_tests}/{total_tests}")
    typer.echo(f"   • Results saved to: {results_file}")

    if passed_tests == total_tests:
        typer.echo("✅ All regression tests PASSED!")
    else:
        typer.echo("❌ Some regression tests FAILED!")
        typer.echo("💡 Recommendation: Investigate performance changes and update baselines if necessary")

    typer.echo("\n🎉 Performance regression testing completed!"


@log_function_call
def run_cv_integrity_command(
    config: Path = typer.Option(..., help="Configuration file"),
    cv_folds: int = typer.Option(5, help="Number of CV folds to test"),
    output_dir: Path = typer.Option("tests/cv_integrity", help="Output directory for results")
) -> None:
    """Check cross-validation integrity and detect data leakage."""
    from clinical_survival.testing import CrossValidationIntegrityChecker
    from clinical_survival.config_validation import load_config
    from clinical_survival.models import make_model

    typer.echo("🔍 Checking cross-validation integrity...")

    # Load configuration
    config_dict = load_config(config)

    # Load data
    data_csv = config_dict["paths"]["data_csv"]
    metadata = config_dict["paths"]["metadata"]

    X, y, metadata_dict = load_dataset(data_csv, metadata)
    typer.echo(f"📊 Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Initialize checker
    checker = CrossValidationIntegrityChecker()

    # Test different models
    models_to_test = config_dict.get("models", ["coxph", "rsf", "xgb_cox"])

    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models_to_test:
        typer.echo(f"\n🔬 Checking {model_name}...")

        try:
            # Create model constructor
            def model_constructor():
                return make_model(model_name, **config_dict.get("model_params", {}).get(model_name, {}))

            # Run integrity check
            result = checker.check_cv_integrity(
                model_constructor=model_constructor,
                X=X,
                y=y,
                cv_folds=cv_folds,
                test_name=f"{model_name}_integrity"
            )

            results[model_name] = result

            if result.passed:
                typer.echo(f"✅ {model_name}: PASSED (concordance: {result.value".3f"})")
            else:
                typer.echo(f"❌ {model_name}: FAILED (concordance: {result.value".3f"})")
                if result.error_message:
                    typer.echo(f"   Issue: {result.error_message}")

        except Exception as e:
            typer.echo(f"❌ {model_name}: ERROR - {str(e)}")
            results[model_name] = TestResult(
                test_name=f"{model_name}_error",
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=0.0,
                error_message=str(e)
            )

    # Save results
    results_file = output_dir / "integrity_results.json"
    results_dict = {
        model_name: {
            "passed": result.passed,
            "value": result.value,
            "threshold": result.threshold,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
        for model_name, result in results.items()
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Summary
    passed_tests = sum(1 for r in results.values() if r.passed)
    total_tests = len(results)

    typer.echo("
📋 CV Integrity Check Summary:"    typer.echo(f"   • Passed: {passed_tests}/{total_tests}")
    typer.echo(f"   • Results saved to: {results_file}")

    if passed_tests == total_tests:
        typer.echo("✅ All CV integrity checks PASSED!")
    else:
        typer.echo("❌ Some CV integrity checks FAILED!")
        typer.echo("💡 Recommendation: Review cross-validation setup for potential data leakage")

    typer.echo("\n🎉 CV integrity check completed!"


@log_function_call
def run_benchmark_suite_command(
    config: Path = typer.Option(..., help="Configuration file"),
    output_dir: Path = typer.Option("tests/benchmark_results", help="Output directory for results"),
    include_sksurv: bool = typer.Option(True, help="Include scikit-survival benchmarks"),
    include_lifelines: bool = typer.Option(True, help="Include lifelines benchmarks")
) -> None:
    """Run comprehensive benchmark against other survival libraries."""
    from clinical_survival.testing import SurvivalBenchmarkSuite
    from clinical_survival.config_validation import load_config
    from clinical_survival.models import make_model

    typer.echo("🏆 Running survival analysis benchmark suite...")

    # Load configuration
    config_dict = load_config(config)

    # Load data
    data_csv = config_dict["paths"]["data_csv"]
    metadata = config_dict["paths"]["metadata"]

    X, y, metadata_dict = load_dataset(data_csv, metadata)
    typer.echo(f"📊 Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Initialize benchmark suite
    benchmarker = SurvivalBenchmarkSuite()

    # Prepare our models
    our_models = {}
    models_to_test = config_dict.get("models", ["coxph", "rsf", "xgb_cox"])

    for model_name in models_to_test:
        try:
            def model_constructor():
                return make_model(model_name, **config_dict.get("model_params", {}).get(model_name, {}))
            our_models[model_name] = model_constructor
        except Exception as e:
            typer.echo(f"⚠️  Could not create {model_name}: {e}")

    # Run benchmarks
    results = benchmarker.run_full_benchmark(X, y, our_models)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "benchmark_results.json"

    results_dict = {
        test_name: {
            "passed": result.passed,
            "value": result.value,
            "threshold": result.threshold,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
        for test_name, result in results.items()
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Display results
    typer.echo("
📊 Benchmark Results:"    for test_name, result in results.items():
        status = "✅" if result.passed else "❌"
        typer.echo(f"   {status} {test_name}: {result.value".3f"}")

    typer.echo(f"\n📁 Results saved to: {results_file}")
    typer.echo("\n🎉 Benchmark suite completed!"
