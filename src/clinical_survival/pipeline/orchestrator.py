import importlib
from typing import Dict, Any

from joblib import Memory
from rich.console import Console

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.tracking import MLflowTracker
from clinical_survival.utils import set_global_seed, ensure_dir

console = Console()


def run_pipeline(
    params_config: ParamsConfig, features_config: FeaturesConfig, grid_config: Dict[str, Any]
) -> None:
    """
    Orchestrates the execution of the modular training pipeline based on the config.
    """
    # 1. Setup
    set_global_seed(params_config.seed)
    tracker = MLflowTracker(params_config.mlflow_tracking.model_dump())
    outdir = ensure_dir(params_config.paths.outdir)

    _memory = None
    if params_config.caching.enabled:
        cache_dir = ensure_dir(params_config.caching.dir)
        _memory = Memory(cache_dir, verbose=0)

    # Pipeline context to pass data between steps
    context: Dict[str, Any] = {
        "params_config": params_config,
        "features_config": features_config,
        "grid_config": grid_config,
        "tracker": tracker,
        "outdir": outdir,
    }

    # 2. Execute pipeline steps from config
    with tracker.start_run("main_run"):
        for step in params_config.pipeline:
            module_name, func_name = step.rsplit(".", 1)
            try:
                module = importlib.import_module(f"clinical_survival.pipeline.{module_name}")
                func = getattr(module, func_name)

                # Call the pipeline function and update the context with its output
                result = func(**context)
                if result:
                    context.update(result)

            except (ImportError, AttributeError) as e:
                console.print(f"[bold red]Error loading pipeline step '{step}': {e}[/bold red]")
                raise
    console.print("✅ Pipeline finished.", style="bold green")
