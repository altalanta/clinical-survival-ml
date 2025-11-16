"""Advanced testing and quality assurance framework for clinical survival models."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    import scikit_survival
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False

try:
    import lifelines
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    passed: bool
    value: float
    threshold: float
    execution_time: float
    error_message: Optional[str] = None


class SyntheticDatasetGenerator:
    """Generate synthetic clinical datasets for testing and validation."""

    def __init__(self, random_state: int = 42):
        """Initialize the dataset generator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_icu_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        survival_time_range: Tuple[float, float] = (1, 365),
        censoring_rate: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic ICU dataset with realistic clinical features.

        Args:
            n_samples: Number of patients
            n_features: Number of clinical features
            survival_time_range: Range for survival times in days
            censoring_rate: Fraction of censored observations

        Returns:
            Tuple of (features DataFrame, outcomes DataFrame)
        """
        # Generate demographic features
        features = pd.DataFrame({
            'age': np.random.normal(65, 15, n_samples).clip(18, 100),
            'sex': np.random.choice(['male', 'female'], n_samples),
            'bmi': np.random.normal(28, 6, n_samples).clip(15, 50),
        })

        # Generate clinical features
        clinical_features = [
            'sofa_score', 'apache_score', 'gcs_score', 'creatinine',
            'bilirubin', 'platelet_count', 'wbc_count', 'temperature',
            'heart_rate', 'respiratory_rate', 'systolic_bp', 'diastolic_bp',
            'oxygen_saturation', 'ph_level', 'lactate', 'hemoglobin',
            'sodium', 'potassium', 'chloride', 'glucose'
        ]

        for feature in clinical_features[:n_features-3]:  # -3 for demographics already added
            if 'score' in feature:
                # Score features (discrete)
                features[feature] = np.random.poisson(5, n_samples) + 1
            elif 'count' in feature:
                # Count features (discrete)
                features[feature] = np.random.poisson(200, n_samples) + 50
            else:
                # Continuous clinical measurements
                features[feature] = np.random.normal(100, 20, n_samples)

        # Generate survival times based on risk factors
        risk_factors = (
            features['age'] / 100 +  # Age effect
            features['sofa_score'] / 20 +  # SOFA score effect
            features['creatinine'] / 10 +  # Kidney function
            features['lactate'] / 5 +  # Tissue perfusion
            np.random.normal(0, 0.5, n_samples)  # Random variation
        )

        # Convert risk to survival time (exponential distribution)
        mean_survival_time = survival_time_range[1] / (1 + np.exp(risk_factors))
        survival_times = np.random.exponential(mean_survival_time)

        # Apply censoring
        censored = np.random.random(n_samples) < censoring_rate
        observed_times = np.where(censored, survival_times * 0.7, survival_times)
        events = ~censored

        outcomes = pd.DataFrame({
            'time': observed_times,
            'event': events.astype(int),
            'true_risk_score': risk_factors
        })

        return features, outcomes

    def generate_cancer_dataset(
        self,
        n_samples: int = 1000,
        cancer_types: List[str] = None,
        survival_time_range: Tuple[float, float] = (30, 1825),  # 1 month to 5 years
        censoring_rate: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic cancer survival dataset.

        Args:
            n_samples: Number of patients
            cancer_types: List of cancer types to include
            survival_time_range: Range for survival times in days
            censoring_rate: Fraction of censored observations

        Returns:
            Tuple of (features DataFrame, outcomes DataFrame)
        """
        if cancer_types is None:
            cancer_types = ['breast', 'lung', 'prostate', 'colorectal', 'lymphoma']

        # Generate patient characteristics
        features = pd.DataFrame({
            'age': np.random.normal(62, 12, n_samples).clip(20, 90),
            'sex': np.random.choice(['male', 'female'], n_samples),
            'cancer_type': np.random.choice(cancer_types, n_samples),
            'stage': np.random.choice([1, 2, 3, 4], n_samples),
            'grade': np.random.choice([1, 2, 3], n_samples),
            'tumor_size': np.random.exponential(3, n_samples).clip(0.1, 10),
            'lymph_nodes': np.random.poisson(2, n_samples),
            'metastasis': np.random.choice([0, 1], n_samples),
            'performance_status': np.random.choice([0, 1, 2, 3, 4], n_samples),
        })

        # Add treatment features
        features['surgery'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        features['chemotherapy'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        features['radiation'] = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        features['targeted_therapy'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

        # Generate comorbidities
        comorbidities = ['diabetes', 'hypertension', 'heart_disease', 'copd', 'kidney_disease']
        for comorbidity in comorbidities:
            features[comorbidity] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

        # Calculate risk score based on cancer characteristics
        risk_factors = (
            features['age'] / 100 +  # Age effect
            features['stage'] / 4 +  # Cancer stage effect
            features['tumor_size'] / 5 +  # Tumor size effect
            features['lymph_nodes'] / 10 +  # Lymph node involvement
            features['metastasis'] * 2 +  # Metastasis effect
            np.random.normal(0, 0.3, n_samples)  # Random variation
        )

        # Convert to survival time
        mean_survival_time = survival_time_range[1] / (1 + np.exp(risk_factors))
        survival_times = np.random.exponential(mean_survival_time)

        # Apply censoring (less censoring for cancer data)
        censored = np.random.random(n_samples) < censoring_rate
        observed_times = np.where(censored, survival_times * 0.8, survival_times)
        events = ~censored

        outcomes = pd.DataFrame({
            'time': observed_times,
            'event': events.astype(int),
            'true_risk_score': risk_factors
        })

        return features, outcomes

    def generate_cardiovascular_dataset(
        self,
        n_samples: int = 1000,
        follow_up_years: int = 5,
        event_rate: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic cardiovascular disease dataset.

        Args:
            n_samples: Number of patients
            follow_up_years: Maximum follow-up time in years
            event_rate: Expected event rate over follow-up period

        Returns:
            Tuple of (features DataFrame, outcomes DataFrame)
        """
        # Generate baseline characteristics
        features = pd.DataFrame({
            'age': np.random.normal(55, 10, n_samples).clip(30, 80),
            'sex': np.random.choice(['male', 'female'], n_samples),
            'systolic_bp': np.random.normal(135, 20, n_samples).clip(90, 200),
            'diastolic_bp': np.random.normal(85, 12, n_samples).clip(50, 120),
            'total_cholesterol': np.random.normal(220, 40, n_samples).clip(120, 400),
            'hdl_cholesterol': np.random.normal(50, 15, n_samples).clip(20, 100),
            'ldl_cholesterol': np.random.normal(140, 35, n_samples).clip(50, 300),
            'triglycerides': np.random.exponential(150, n_samples).clip(50, 500),
            'glucose': np.random.normal(100, 25, n_samples).clip(60, 300),
            'bmi': np.random.normal(28, 5, n_samples).clip(15, 50),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        })

        # Calculate cardiovascular risk score
        risk_factors = (
            (features['age'] - 55) / 20 +  # Age effect
            (features['systolic_bp'] - 120) / 40 +  # Blood pressure effect
            (features['total_cholesterol'] - 200) / 100 +  # Cholesterol effect
            features['smoking'] * 0.5 +  # Smoking effect
            features['diabetes'] * 0.8 +  # Diabetes effect
            features['bmi'] / 50 +  # BMI effect
            np.random.normal(0, 0.2, n_samples)  # Random variation
        )

        # Convert risk to time-to-event
        mean_survival_time = follow_up_years * 365 / (1 + np.exp(risk_factors))
        survival_times = np.random.exponential(mean_survival_time)

        # Apply censoring based on follow-up time
        max_follow_up = follow_up_years * 365
        censored = survival_times > max_follow_up
        observed_times = np.minimum(survival_times, max_follow_up)
        events = ~censored

        outcomes = pd.DataFrame({
            'time': observed_times,
            'event': events.astype(int),
            'true_risk_score': risk_factors
        })

        return features, outcomes


class PerformanceRegressionTester:
    """Automated performance regression testing for survival models."""

    def __init__(self, baseline_file: Optional[str] = None):
        """Initialize the regression tester.

        Args:
            baseline_file: Path to baseline performance file (optional)
        """
        self.baseline_file = baseline_file or "tests/baseline_performance.json"
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> Dict[str, Dict[str, float]]:
        """Load baseline performance metrics."""
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_baselines(self) -> None:
        """Save current performance as baselines."""
        Path(self.baseline_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)

    def run_regression_test(
        self,
        model_constructor: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_name: str,
        metric: str = "concordance",
        tolerance: float = 0.05
    ) -> TestResult:
        """Run a performance regression test.

        Args:
            model_constructor: Function that creates a model instance
            X: Feature matrix
            y: Survival outcomes (time, event pairs)
            test_name: Name of the test
            metric: Performance metric to test
            tolerance: Acceptable performance degradation (fraction)

        Returns:
            TestResult object
        """
        start_time = time.time()

        try:
            # Prepare data for sklearn-style cross-validation
            times = y['time'].values
            events = y['event'].values

            # Create stratified folds based on event status
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Run cross-validation
            scores = []
            for train_idx, test_idx in skf.split(X, events):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = list(zip(events[train_idx], times[train_idx]))
                y_test = list(zip(events[test_idx], times[test_idx]))

                model = model_constructor()
                model.fit(X_train, y_train)

                # Calculate concordance for this fold
                try:
                    risk_scores = model.predict_risk(X_test)
                    concordance = self._calculate_concordance(times[test_idx], events[test_idx], risk_scores)
                    scores.append(concordance)
                except Exception as e:
                    warnings.warn(f"Failed to calculate concordance for fold: {e}")
                    scores.append(0.5)  # Neutral score

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Check against baseline
            baseline_score = self.baselines.get(test_name, {}).get(metric, mean_score)
            threshold = baseline_score * (1 - tolerance)

            passed = mean_score >= threshold

            # Update baseline if test passes
            if passed and test_name not in self.baselines:
                self.baselines[test_name] = {metric: mean_score}
                self._save_baselines()

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_name,
                passed=passed,
                value=mean_score,
                threshold=threshold,
                execution_time=execution_time,
                error_message=None
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _calculate_concordance(self, times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
        """Calculate concordance index for survival data."""
        # Simple concordance calculation for testing purposes
        n_pairs = 0
        n_correct = 0

        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if events[i] == 1 and events[j] == 1:  # Both experienced events
                    if times[i] < times[j]:  # i experienced event first
                        n_pairs += 1
                        if risk_scores[i] > risk_scores[j]:  # Higher risk for earlier event
                            n_correct += 1
                    elif times[j] < times[i]:  # j experienced event first
                        n_pairs += 1
                        if risk_scores[j] > risk_scores[i]:  # Higher risk for earlier event
                            n_correct += 1

        return n_correct / max(n_pairs, 1)


class CrossValidationIntegrityChecker:
    """Verify cross-validation integrity and detect data leakage."""

    def __init__(self):
        """Initialize the integrity checker."""
        pass

    def check_cv_integrity(
        self,
        model_constructor: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        cv_folds: int = 5,
        test_name: str = "cv_integrity"
    ) -> TestResult:
        """Check for data leakage in cross-validation.

        Args:
            model_constructor: Function that creates a model instance
            X: Feature matrix
            y: Survival outcomes
            cv_folds: Number of CV folds to test
            test_name: Name of the test

        Returns:
            TestResult object
        """
        start_time = time.time()

        try:
            times = y['time'].values
            events = y['event'].values

            # Test for suspicious performance patterns that might indicate leakage
            scores = []

            for fold in range(cv_folds):
                # Create train/test split
                test_start = int(fold * len(X) / cv_folds)
                test_end = int((fold + 1) * len(X) / cv_folds)

                test_mask = np.zeros(len(X), dtype=bool)
                test_mask[test_start:test_end] = True

                X_train, X_test = X[~test_mask], X[test_mask]
                y_train = list(zip(events[~test_mask], times[~test_mask]))
                y_test = list(zip(events[test_mask], times[test_mask]))

                # Train and evaluate
                model = model_constructor()
                model.fit(X_train, y_train)

                try:
                    risk_scores = model.predict_risk(X_test)
                    concordance = self._calculate_concordance(times[test_mask], events[test_mask], risk_scores)
                    scores.append(concordance)
                except Exception:
                    scores.append(0.5)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Check for suspicious patterns
            # 1. Very low variance across folds (might indicate leakage)
            # 2. Perfect or near-perfect scores (might indicate overfitting/leakage)
            suspicious_variance = std_score < 0.01  # Very low variance
            suspicious_performance = mean_score > 0.95  # Unrealistically high performance

            passed = not (suspicious_variance or suspicious_performance)

            execution_time = time.time() - start_time

            error_msg = None
            if suspicious_variance:
                error_msg = f"Low variance across CV folds (std={std_score".4f"}) suggests possible data leakage"
            elif suspicious_performance:
                error_msg = f"Unrealistically high performance (mean={mean_score".4f"}) suggests possible overfitting"

            return TestResult(
                test_name=test_name,
                passed=passed,
                value=mean_score,
                threshold=0.8,  # Expected realistic performance threshold
                execution_time=execution_time,
                error_message=error_msg
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _calculate_concordance(self, times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
        """Calculate concordance index."""
        n_pairs = 0
        n_correct = 0

        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if events[i] == 1 and events[j] == 1:
                    if times[i] < times[j]:
                        n_pairs += 1
                        if risk_scores[i] > risk_scores[j]:
                            n_correct += 1
                    elif times[j] < times[i]:
                        n_pairs += 1
                        if risk_scores[j] > risk_scores[i]:
                            n_correct += 1

        return n_correct / max(n_pairs, 1)


class SurvivalBenchmarkSuite:
    """Benchmark against other survival analysis libraries."""

    def __init__(self):
        """Initialize the benchmark suite."""
        self.results = {}

    def benchmark_vs_sksurv(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_name: str = "vs_sksurv"
    ) -> Dict[str, TestResult]:
        """Benchmark against scikit-survival.

        Args:
            X: Feature matrix
            y: Survival outcomes
            test_name: Name of the benchmark test

        Returns:
            Dictionary of test results
        """
        if not SKSURV_AVAILABLE:
            return {"sksurv_unavailable": TestResult(
                test_name="sksurv_benchmark",
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=0.0,
                error_message="scikit-survival not available"
            )}

        results = {}

        try:
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.linear_model import CoxPHSurvivalAnalysis
            from sksurv.metrics import concordance_index_censored

            # Prepare data for scikit-survival format
            times = y['time'].values
            events = y['event'].astype(bool).values
            y_structured = np.array([(event, time) for event, time in zip(events, times)],
                                   dtype=[('event', bool), ('time', float)])

            # Test Cox PH model
            cox_model = CoxPHSurvivalAnalysis()
            cox_model.fit(X.values, y_structured)

            # Calculate concordance
            risk_scores = cox_model.predict(X.values)
            concordance = concordance_index_censored(events, times, risk_scores)[0]

            results["coxph_concordance"] = TestResult(
                test_name=f"{test_name}_coxph",
                passed=True,
                value=concordance,
                threshold=0.6,
                execution_time=0.0,
                error_message=None
            )

            # Test Random Survival Forest
            rsf_model = RandomSurvivalForest(n_estimators=100, random_state=42)
            rsf_model.fit(X.values, y_structured)

            risk_scores_rsf = rsf_model.predict(X.values)
            concordance_rsf = concordance_index_censored(events, times, risk_scores_rsf)[0]

            results["rsf_concordance"] = TestResult(
                test_name=f"{test_name}_rsf",
                passed=True,
                value=concordance_rsf,
                threshold=0.6,
                execution_time=0.0,
                error_message=None
            )

        except Exception as e:
            results["sksurv_error"] = TestResult(
                test_name=f"{test_name}_error",
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=0.0,
                error_message=str(e)
            )

        return results

    def benchmark_vs_lifelines(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_name: str = "vs_lifelines"
    ) -> Dict[str, TestResult]:
        """Benchmark against lifelines.

        Args:
            X: Feature matrix
            y: Survival outcomes
            test_name: Name of the benchmark test

        Returns:
            Dictionary of test results
        """
        if not LIFELINES_AVAILABLE:
            return {"lifelines_unavailable": TestResult(
                test_name="lifelines_benchmark",
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=0.0,
                error_message="lifelines not available"
            )}

        results = {}

        try:
            from lifelines import CoxPHFitter
            from lifelines.utils import concordance_index

            # Prepare data for lifelines
            times = y['time'].values
            events = y['event'].astype(bool).values

            # Test Cox PH model
            cox_model = CoxPHFitter()
            df_lifelines = X.copy()
            df_lifelines['time'] = times
            df_lifelines['event'] = events

            cox_model.fit(df_lifelines, duration_col='time', event_col='event')

            # Calculate concordance
            partial_hazards = cox_model.predict_partial_hazards(X)
            concordance = concordance_index(times, partial_hazards.values.flatten(), events)

            results["coxph_concordance"] = TestResult(
                test_name=f"{test_name}_coxph",
                passed=True,
                value=concordance,
                threshold=0.6,
                execution_time=0.0,
                error_message=None
            )

        except Exception as e:
            results["lifelines_error"] = TestResult(
                test_name=f"{test_name}_error",
                passed=False,
                value=0.0,
                threshold=0.0,
                execution_time=0.0,
                error_message=str(e)
            )

        return results

    def run_full_benchmark(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        our_models: Dict[str, Any],
        test_name: str = "full_benchmark"
    ) -> Dict[str, TestResult]:
        """Run comprehensive benchmark against other libraries.

        Args:
            X: Feature matrix
            y: Survival outcomes
            our_models: Dictionary of our model constructors
            test_name: Name of the benchmark test

        Returns:
            Dictionary of all benchmark results
        """
        all_results = {}

        # Benchmark against scikit-survival
        sksurv_results = self.benchmark_vs_sksurv(X, y, f"{test_name}_sksurv")
        all_results.update(sksurv_results)

        # Benchmark against lifelines
        lifelines_results = self.benchmark_vs_lifelines(X, y, f"{test_name}_lifelines")
        all_results.update(lifelines_results)

        # Run our models for comparison
        for model_name, model_constructor in our_models.items():
            try:
                times = y['time'].values
                events = y['event'].values
                y_pairs = list(zip(events, times))

                model = model_constructor()
                model.fit(X, y_pairs)

                risk_scores = model.predict_risk(X)
                concordance = self._calculate_concordance(times, events, risk_scores)

                all_results[f"{model_name}_concordance"] = TestResult(
                    test_name=f"{test_name}_{model_name}",
                    passed=True,
                    value=concordance,
                    threshold=0.6,
                    execution_time=0.0,
                    error_message=None
                )

            except Exception as e:
                all_results[f"{model_name}_error"] = TestResult(
                    test_name=f"{test_name}_{model_name}_error",
                    passed=False,
                    value=0.0,
                    threshold=0.0,
                    execution_time=0.0,
                    error_message=str(e)
                )

        return all_results

    def _calculate_concordance(self, times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
        """Calculate concordance index."""
        n_pairs = 0
        n_correct = 0

        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if events[i] == 1 and events[j] == 1:
                    if times[i] < times[j]:
                        n_pairs += 1
                        if risk_scores[i] > risk_scores[j]:
                            n_correct += 1
                    elif times[j] < times[i]:
                        n_pairs += 1
                        if risk_scores[j] > risk_scores[i]:
                            n_correct += 1

        return n_correct / max(n_pairs, 1)













