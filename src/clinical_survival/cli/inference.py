"""CLI commands for model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from clinical_survival.logging_config import get_logger

app = typer.Typer(
    name="inference",
    help="Run inference using trained models.",
)
console = Console()
logger = get_logger(__name__)


@app.command("predict")
def predict(
    model_path: Path = typer.Argument(
        ...,
        help="Path to the trained model (.joblib file)",
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to CSV file with patient data for prediction",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save predictions (CSV or JSON based on extension)",
    ),
    times: Optional[str] = typer.Option(
        "365,730,1095",
        "--times", "-t",
        help="Comma-separated time points for survival probabilities (in days)",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Column name for patient IDs",
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, csv, or json",
    ),
) -> None:
    """
    Run predictions on new patient data.
    
    Examples:
    
        # Basic prediction
        clinical-ml inference predict models/xgb_cox.joblib new_patients.csv
        
        # Save predictions to file
        clinical-ml inference predict models/xgb_cox.joblib data.csv -o predictions.csv
        
        # Custom time points
        clinical-ml inference predict models/xgb_cox.joblib data.csv --times 90,180,365
    """
    import pandas as pd
    from clinical_survival.inference import ModelInference
    
    # Parse time points
    time_points = [int(t.strip()) for t in times.split(",")] if times else None
    
    # Load model
    console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
    inference = ModelInference.load(model_path)
    console.print(f"[green]✓[/green] Model loaded: {inference.model_name}")
    
    # Load data
    console.print(f"[cyan]Loading data from {data_path}...[/cyan]")
    df = pd.read_csv(data_path)
    console.print(f"[green]✓[/green] Loaded {len(df)} samples")
    
    # Get patient IDs
    patient_ids = None
    if id_column and id_column in df.columns:
        patient_ids = df[id_column].astype(str).tolist()
        df = df.drop(columns=[id_column])
    
    # Run predictions
    console.print("[cyan]Running predictions...[/cyan]")
    results = inference.predict_batch(df, patient_ids=patient_ids, times=time_points)
    console.print(f"[green]✓[/green] Generated {len(results.predictions)} predictions")
    
    # Output results
    if format == "table" or (format == "table" and output_path is None):
        _print_predictions_table(results, time_points)
    
    if output_path:
        output_path = Path(output_path)
        
        if output_path.suffix.lower() == ".json":
            import json
            with open(output_path, "w") as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            console.print(f"[green]✓[/green] Saved predictions to {output_path}")
        else:
            results_df = results.to_dataframe()
            results_df.to_csv(output_path, index=False)
            console.print(f"[green]✓[/green] Saved predictions to {output_path}")


def _print_predictions_table(results, time_points) -> None:
    """Print predictions as a Rich table."""
    table = Table(title="Predictions")
    
    table.add_column("Patient", style="cyan")
    table.add_column("Risk Score", justify="right")
    table.add_column("Risk %ile", justify="right")
    table.add_column("Category", style="bold")
    
    if time_points:
        for t in time_points[:3]:  # Limit to first 3 time points
            table.add_column(f"Surv@{t}d", justify="right")
    
    for pred in results.predictions[:20]:  # Limit to first 20
        row = [
            pred.patient_id or "N/A",
            f"{pred.risk_score:.4f}",
            f"{pred.risk_percentile:.1f}%" if pred.risk_percentile else "N/A",
            _style_category(pred.risk_category),
        ]
        
        if time_points:
            for t in time_points[:3]:
                prob = pred.survival_probabilities.get(t)
                row.append(f"{prob:.1%}" if prob else "N/A")
        
        table.add_row(*row)
    
    if len(results.predictions) > 20:
        table.add_row("...", "...", "...", "...", *["..."] * min(3, len(time_points or [])))
    
    console.print(table)


def _style_category(category: str) -> str:
    """Apply color styling to risk category."""
    styles = {
        "low": "[green]Low[/green]",
        "moderate": "[yellow]Moderate[/yellow]",
        "high": "[orange3]High[/orange3]",
        "very_high": "[red]Very High[/red]",
    }
    return styles.get(category, category)


@app.command("single")
def predict_single(
    model_path: Path = typer.Argument(
        ...,
        help="Path to the trained model",
    ),
    features: str = typer.Argument(
        ...,
        help="Feature values as JSON string, e.g., '{\"age\": 65, \"stage\": 2}'",
    ),
    times: Optional[str] = typer.Option(
        "365,730,1095",
        "--times", "-t",
        help="Comma-separated time points for survival probabilities",
    ),
) -> None:
    """
    Run prediction for a single patient.
    
    Examples:
    
        clinical-ml inference single models/xgb_cox.joblib '{"age": 65, "stage": 2}'
    """
    import json
    from clinical_survival.inference import ModelInference
    
    # Parse features
    try:
        feature_dict = json.loads(features)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing features JSON:[/red] {e}")
        raise typer.Exit(1)
    
    # Parse time points
    time_points = [int(t.strip()) for t in times.split(",")] if times else None
    
    # Load model and predict
    inference = ModelInference.load(model_path)
    result = inference.predict(feature_dict, times=time_points)
    
    # Display result
    console.print()
    console.print("[bold cyan]Prediction Result[/bold cyan]")
    console.print(f"  Risk Score: [bold]{result.risk_score:.4f}[/bold]")
    
    if result.risk_percentile is not None:
        console.print(f"  Percentile: {result.risk_percentile:.1f}%")
    
    console.print(f"  Category: {_style_category(result.risk_category)}")
    
    if result.survival_probabilities:
        console.print()
        console.print("[bold]Survival Probabilities:[/bold]")
        for time, prob in sorted(result.survival_probabilities.items()):
            years = time / 365
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            console.print(f"  {time:4d}d ({years:.1f}y): {bar} {prob:.1%}")
    
    if result.median_survival_time:
        console.print(f"\n  Median Survival: {result.median_survival_time:.0f} days")
    
    console.print()


@app.command("explain")
def explain_prediction(
    model_path: Path = typer.Argument(
        ...,
        help="Path to the trained model",
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to CSV with patient data",
    ),
    sample_index: int = typer.Option(
        0,
        "--index", "-i",
        help="Index of sample to explain (0-based)",
    ),
    n_features: int = typer.Option(
        10,
        "--n-features", "-n",
        help="Number of top features to show",
    ),
) -> None:
    """
    Explain a prediction using feature contributions.
    
    Shows which features contributed most to the prediction.
    """
    import pandas as pd
    from clinical_survival.inference import ModelInference
    
    # Load model and data
    inference = ModelInference.load(model_path)
    df = pd.read_csv(data_path)
    
    if sample_index >= len(df):
        console.print(f"[red]Sample index {sample_index} out of range (max: {len(df)-1})[/red]")
        raise typer.Exit(1)
    
    sample = df.iloc[sample_index]
    
    # Get prediction
    result = inference.predict(sample)
    
    console.print()
    console.print(f"[bold cyan]Prediction Explanation (Sample {sample_index})[/bold cyan]")
    console.print(f"  Risk Score: [bold]{result.risk_score:.4f}[/bold]")
    console.print(f"  Category: {_style_category(result.risk_category)}")
    console.print()
    
    # Try SHAP explanation
    try:
        import shap
        
        console.print("[cyan]Computing SHAP values...[/cyan]")
        
        # Get underlying model
        model = inference.model
        if hasattr(model, "named_steps") and "est" in model.named_steps:
            estimator = model.named_steps["est"]
        else:
            estimator = model
        
        # Create explainer
        if hasattr(estimator, "predict_risk"):
            predict_fn = lambda x: estimator.predict_risk(pd.DataFrame(x, columns=sample.index))
        else:
            predict_fn = lambda x: estimator.predict(pd.DataFrame(x, columns=sample.index))
        
        explainer = shap.Explainer(predict_fn, sample.values.reshape(1, -1))
        shap_values = explainer(sample.values.reshape(1, -1))
        
        # Display top features
        feature_importance = list(zip(sample.index, shap_values.values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        table = Table(title="Feature Contributions")
        table.add_column("Feature", style="cyan")
        table.add_column("Value")
        table.add_column("Contribution", justify="right")
        table.add_column("Direction")
        
        for feature, contribution in feature_importance[:n_features]:
            value = sample[feature]
            direction = "[red]↑ Risk[/red]" if contribution > 0 else "[green]↓ Risk[/green]"
            table.add_row(
                str(feature),
                f"{value:.3f}" if isinstance(value, float) else str(value),
                f"{contribution:+.4f}",
                direction,
            )
        
        console.print(table)
        
    except ImportError:
        console.print("[yellow]SHAP not installed. Install with: pip install shap[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Could not compute SHAP values: {e}[/yellow]")
    
    console.print()


@app.command("compare")
def compare_models(
    model_paths: str = typer.Argument(
        ...,
        help="Comma-separated paths to model files",
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to test data CSV",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save comparison results",
    ),
) -> None:
    """
    Compare predictions from multiple models.
    """
    import pandas as pd
    import numpy as np
    from clinical_survival.inference import ModelInference
    
    paths = [Path(p.strip()) for p in model_paths.split(",")]
    
    # Load data
    df = pd.read_csv(data_path)
    console.print(f"[cyan]Loaded {len(df)} samples[/cyan]")
    
    # Load models and get predictions
    all_predictions = {}
    
    for path in paths:
        console.print(f"[cyan]Loading {path.name}...[/cyan]")
        inference = ModelInference.load(path)
        risks = inference.predict_risk_batch(df)
        all_predictions[inference.model_name] = risks
    
    # Create comparison table
    table = Table(title="Model Comparison")
    table.add_column("Metric")
    
    for name in all_predictions.keys():
        table.add_column(name, justify="right")
    
    # Statistics
    stats = ["Mean Risk", "Std Risk", "Min Risk", "Max Risk"]
    
    for stat in stats:
        row = [stat]
        for name, risks in all_predictions.items():
            if stat == "Mean Risk":
                row.append(f"{np.mean(risks):.4f}")
            elif stat == "Std Risk":
                row.append(f"{np.std(risks):.4f}")
            elif stat == "Min Risk":
                row.append(f"{np.min(risks):.4f}")
            elif stat == "Max Risk":
                row.append(f"{np.max(risks):.4f}")
        table.add_row(*row)
    
    # Correlation between models
    if len(all_predictions) > 1:
        console.print()
        corr_table = Table(title="Risk Score Correlations")
        corr_table.add_column("")
        
        names = list(all_predictions.keys())
        for name in names:
            corr_table.add_column(name, justify="right")
        
        for name1 in names:
            row = [name1]
            for name2 in names:
                corr = np.corrcoef(all_predictions[name1], all_predictions[name2])[0, 1]
                row.append(f"{corr:.3f}")
            corr_table.add_row(*row)
        
        console.print(corr_table)
    
    console.print(table)
    
    if output_path:
        comparison_df = pd.DataFrame(all_predictions)
        comparison_df.to_csv(output_path, index=False)
        console.print(f"[green]✓[/green] Saved comparison to {output_path}")





