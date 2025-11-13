from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ScoringConfig(BaseModel):
    primary: str
    secondary: List[str]

class CalibrationConfig(BaseModel):
    times_days: List[int]
    bins: int

class DecisionCurveConfig(BaseModel):
    times_days: List[int]
    thresholds: List[float]

class MissingConfig(BaseModel):
    strategy: str
    max_iter: int
    initial_strategy: str

class ExplainConfig(BaseModel):
    shap_samples: int
    pdp_features: List[str]

class MonitoringAlertsConfig(BaseModel):
    concordance_drop: float
    brier_increase: float
    feature_drift: float
    concept_drift: float

class MonitoringConfig(BaseModel):
    alert_thresholds: MonitoringAlertsConfig
    baseline_window_days: int
    max_history_records: int

class PathsConfig(BaseModel):
    data_csv: str
    metadata: str
    external_csv: Optional[str]
    outdir: str
    features: str

class EvaluationConfig(BaseModel):
    bootstrap: int

class ExternalConfig(BaseModel):
    label: str
    group_column: str
    train_value: str
    external_value: str

class IncrementalLearningConfig(BaseModel):
    enabled: bool
    update_frequency_days: int
    min_samples_for_update: int
    max_samples_in_memory: int
    update_strategy: str
    drift_detection_enabled: bool
    create_backup_before_update: bool
    backup_retention_days: int

class DistributedComputingConfig(BaseModel):
    enabled: bool
    cluster_type: str
    n_workers: int
    threads_per_worker: int
    memory_per_worker: str
    partition_strategy: str
    n_partitions: int
    scheduler_address: str
    dashboard_address: str

class ClinicalContextConfig(BaseModel):
    feature_domains: Dict[str, str]
    risk_categories: Dict[str, List[str]]

class ExplanationFeaturesConfig(BaseModel):
    include_shap_values: bool
    include_clinical_interpretation: bool
    include_risk_stratification: bool
    include_recommendations: bool
    include_confidence_scores: bool

class ClinicalInterpretabilityConfig(BaseModel):
    enabled: bool
    risk_thresholds: Dict[str, float]
    clinical_context: ClinicalContextConfig
    explanation_features: ExplanationFeaturesConfig

class MLOpsEnvironmentConfig(BaseModel):
    type: str
    auto_rollback: bool
    rollback_threshold: float

class MLOpsTriggerConfig(BaseModel):
    enabled: bool
    trigger_type: str
    schedule_cron: Optional[str] = None
    performance_threshold: Optional[float] = None
    drift_threshold: Optional[float] = None
    auto_retrain: Optional[bool] = None
    require_approval: bool

class MLOpsDeploymentSettingsConfig(BaseModel):
    require_approval_for_production: bool
    auto_rollback_on_failure: bool
    max_concurrent_deployments: int
    deployment_timeout_minutes: int

class MLOpsConfig(BaseModel):
    enabled: bool
    registry_path: str
    environments: Dict[str, MLOpsEnvironmentConfig]
    triggers: Dict[str, MLOpsTriggerConfig]
    deployment_settings: MLOpsDeploymentSettingsConfig

class MLflowTrackingConfig(BaseModel):
    enabled: bool
    experiment_name: str
    tracking_uri: str

class CachingConfig(BaseModel):
    enabled: bool
    dir: str

class ParamsConfig(BaseModel):
    seed: int
    n_splits: int
    inner_splits: int
    test_split: float
    time_col: str
    event_col: str
    id_col: str
    scoring: ScoringConfig
    calibration: CalibrationConfig
    decision_curve: DecisionCurveConfig = Field(alias="decision_curve")
    missing: MissingConfig
    models: List[str]
    explain: ExplainConfig
    monitoring: MonitoringConfig
    paths: PathsConfig
    evaluation: EvaluationConfig
    external: ExternalConfig
    incremental_learning: IncrementalLearningConfig = Field(alias="incremental_learning")
    distributed_computing: DistributedComputingConfig = Field(alias="distributed_computing")
    clinical_interpretability: ClinicalInterpretabilityConfig = Field(alias="clinical_interpretability")
    mlops: MLOpsConfig
    mlflow_tracking: MLflowTrackingConfig = Field(alias="mlflow_tracking")
    caching: CachingConfig
    pipeline: List[str]

class FeaturesConfig(BaseModel):
    numeric: List[str]
    categorical: List[str]
    drop: List[str]

