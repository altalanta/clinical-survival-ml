from pathlib import Path
import pandas as pd
from rich.console import Console

from clinical_survival.config import ParamsConfig
from clinical_survival.counterfactual import generate_cf_explanations
from clinical_survival.utils import ensure_dir

console = Console()


def generate_all_counterfactuals(
    X: pd.DataFrame,
    y_surv: pd.DataFrame,
    params_config: ParamsConfig,
    outdir: Path,
    **kwargs,
):
    """
    Orchestrates the generation of counterfactual explanations for all trained models.
    """
    if not params_config.counterfactuals.enabled:
        console.print("Counterfactual explanations are disabled. Skipping.", style="yellow")
        return

    cf_config = params_config.counterfactuals

    # This pipeline step relies on the 'final_pipelines' dictionary
    # being present in the context, which is an output of the training_loop.
    final_pipelines = kwargs.get("final_pipelines")
    if not final_pipelines:
        console.print("Could not find trained models. Skipping counterfactuals.", style="red")
        return

    cf_dir = ensure_dir(outdir / "artifacts" / "counterfactuals")

    for model_name, pipeline in final_pipelines.items():
        console.print(f"--- Generating counterfactuals for {model_name} ---")
        output_path = cf_dir / f"{model_name}_counterfactuals.{cf_config.output_format}"

        try:
            generate_cf_explanations(
                pipeline=pipeline,
                X=X,
                y_surv=y_surv,
                features_to_vary=cf_config.features_to_vary,
                n_examples=cf_config.n_examples,
                sample_size=cf_config.sample_size,
                output_path=output_path,
            )
        except Exception as e:
            console.print(
                f"[bold red]Failed to generate counterfactuals for {model_name}: {e}[/bold red]"
            )




