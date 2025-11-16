from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt, confloat


class PathsConfig(BaseModel):
    data_csv: Path
    metadata: Path
    external_csv: Path | None = None
    outdir: Path
    features: Path


class ScoringConfig(BaseModel):
    primary: str
    secondary: list[str] = Field(default_factory=list)


class CalibrationConfig(BaseModel):
    times_days: list[PositiveInt]
    bins: int = Field(ge=5, le=20)


class DecisionCurveConfig(BaseModel):
    times_days: list[PositiveInt]
    thresholds: list[confloat(ge=0, le=1)]  # type: ignore


class MissingConfig(BaseModel):
    strategy: Literal["iterative", "simple"]
    max_iter: PositiveInt = Field(ge=1, le=50)
    initial_strategy: Literal["mean", "median", "most_frequent", "constant"]


class EvaluationConfig(BaseModel):
    bootstrap: PositiveInt = Field(ge=10, le=1000)


class ExternalConfig(BaseModel):
    label: str | None = None
    group_column: str | None = None
    train_value: str | None = None
    external_value: str | None = None


class ExplainConfig(BaseModel):
    shap_samples: PositiveInt = Field(ge=10, le=1000)
    pdp_features: list[str] = Field(default_factory=list)


class ParamsConfig(BaseModel):
    seed: PositiveInt = Field(ge=0)
    n_splits: int = Field(ge=2, le=10)
    inner_splits: int = Field(ge=2, le=5)
    test_split: confloat(ge=0.1, le=0.5)  # type: ignore
    time_col: str
    event_col: str
    id_col: str | None = None
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    decision_curve: DecisionCurveConfig = Field(default_factory=DecisionCurveConfig)
    missing: MissingConfig = Field(default_factory=MissingConfig)
    models: list[Literal["coxph", "rsf", "xgb_cox", "xgb_aft"]]
    paths: PathsConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    external: ExternalConfig = Field(default_factory=ExternalConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)


class ModelGridConfig(BaseModel):
    # Using a dict to allow dynamic keys for model names
    # The actual validation for model names will happen in runtime if needed,
    # or through a custom validator if strict type checking is required for keys.
    __root__: dict[str, Any] = Field(default_factory=dict)

    # A validator to ensure keys are valid model names if desired
    # @validator('__root__', pre=True)
    # def check_model_names(cls, v):
    #     valid_models = ["coxph", "rsf", "xgb_cox", "xgb_aft"]
    #     for model_name in v.keys():
    #         if not re.match(r"^(coxph|rsf|xgb_cox|xgb_aft)$", model_name):
    #             raise ValueError(f"Invalid model name in model_grid: {model_name}")
    #     return v


class FeaturesConfig(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    drop: list[str] = Field(default_factory=list)







