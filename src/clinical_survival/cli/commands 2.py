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
from clinical_survival.utils import (
    ensure_dir,
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
        plot_reliability(
            cv_reliability.assign(label="cv"), calibration_dir / f"reliability_{model_name}_cv.png"
        )

        cv_decision = decision_curve_ipcw(y_train_eval, oof_surv, eval_times, decision_thresholds)
        cv_decision.assign(label="cv").to_csv(
            decision_dir / f"net_benefit_{model_name}_cv.csv", index=False
        )
        plot_decision(
            cv_decision.assign(label="cv"), decision_dir / f"net_benefit_{model_name}_cv.png"
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

    typer.echo(
        format_success_message(f"Training complete. Leaderboard saved to {leaderboard_path}")
    )
    typer.echo(f"Best model: {best_model_name}")
    typer.echo(f"Results available in: {outdir}")


def run_evaluate_command(config: Path) -> None:
    """Run the evaluate command."""
    params = load_yaml(config)
    metrics_dir = Path(params["paths"]["outdir"]) / "artifacts" / "metrics"
    leaderboard_path = metrics_dir / "leaderboard.csv"
    external_path = metrics_dir / "external_summary.csv"
    if leaderboard_path.exists():
        typer.echo("Leaderboard:\n" + leaderboard_path.read_text())
    if external_path.exists():
        typer.echo("\nExternal validation:\n" + external_path.read_text())


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
        calibration_figs["oof"] = calibration_dir / f"reliability_{best_model}_cv.png"
        decision_figs["oof"] = decision_dir / f"net_benefit_{best_model}_cv.png"
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
        typer.echo(f"  Concordance: {summary['concordance']['mean']:.3f} ± {summary['concordance']['std']:.3f}")

        if "brier_score" in summary:
            typer.echo(f"  Brier Score: {summary['brier_score']['mean']:.3f} ± {summary['brier_score']['std']:.3f}")

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
