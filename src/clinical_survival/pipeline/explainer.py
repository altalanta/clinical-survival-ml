from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from rich.console import Console

from clinical_survival.tracking import MLflowTracker
from clinical_survival.utils import ensure_dir
from clinical_survival.explainability.shap_explainer import ShapExplainer

console = Console()


def generate_explanations(
    pipeline: Pipeline,
    X: pd.DataFrame,
    model_name: str,
    explain_dir: Path,
    tracker: MLflowTracker,
) -> None:
    """Generates and logs SHAP explanations."""
    console.print(f"--- Generating SHAP explanations for {model_name} ---")
    try:
        explainer = ShapExplainer(pipeline, X)

        summary_plot_path = explain_dir / "shap_summary.png"
        explainer.save_summary_plot(summary_plot_path)
        tracker.log_artifact(str(summary_plot_path), artifact_path=f"explainability/{model_name}")

        dependence_plots_dir = ensure_dir(explain_dir / "dependence_plots")
        explainer.save_top_dependence_plots(dependence_plots_dir)
        tracker.log_artifacts(
            str(dependence_plots_dir), artifact_path=f"explainability/{model_name}/dependence_plots"
        )
    except Exception as e:
        console.print(
            f"[bold red]Failed to generate SHAP explanations for {model_name}: {e}[/bold red]"
        )

