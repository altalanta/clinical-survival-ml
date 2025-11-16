"""Counterfactual explanations and causal inference for clinical survival models."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import json
from pathlib import Path

import dice_ml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from rich.console import Console

console = Console()


try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class CounterfactualExplainer:
    """Generate counterfactual explanations for survival model predictions."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        feature_ranges: dict[str, tuple[float, float]] | None = None,
        method: str = "gradient",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        step_size: float = 0.1,
        random_state: int | None = None,
    ):
        """Initialize counterfactual explainer.

        Args:
            model: Trained survival model with predict_risk method
            feature_names: List of feature names in the same order as model expects
            feature_ranges: Dict mapping feature names to (min, max) ranges
            method: Counterfactual generation method ('gradient', 'genetic', 'random')
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
            step_size: Step size for gradient-based methods
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_size = step_size
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize feature bounds
        self._initialize_bounds()

    def _initialize_bounds(self) -> None:
        """Initialize feature bounds from ranges or data statistics."""
        self.bounds = {}

        for feature in self.feature_names:
            if feature in self.feature_ranges:
                self.bounds[feature] = self.feature_ranges[feature]
            else:
                # Default bounds: will be set when first counterfactual is generated
                self.bounds[feature] = (None, None)

    def generate_counterfactual(
        self,
        X: pd.DataFrame | np.ndarray,
        target_risk: float | None = None,
        target_survival_time: float | None = None,
        n_counterfactuals: int = 1,
        distance_metric: str = "euclidean",
        max_distance: float = np.inf,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate counterfactual explanations.

        Args:
            X: Input features for which to generate counterfactuals
            target_risk: Target risk value to achieve
            target_survival_time: Target survival time to achieve
            n_counterfactuals: Number of counterfactuals to generate
            distance_metric: Distance metric for optimization ('euclidean', 'manhattan')
            max_distance: Maximum allowed distance from original instance
            **kwargs: Additional method-specific parameters

        Returns:
            Dictionary containing counterfactuals and metadata
        """
        X = self._ensure_dataframe(X)

        if target_risk is None and target_survival_time is None:
            raise ValueError("Must specify either target_risk or target_survival_time")

        # Set up optimization target
        if target_risk is not None:
            target_func = lambda x: abs(self._predict_risk(x) - target_risk)
        elif target_survival_time is not None:
            target_func = lambda x: abs(self._predict_survival_time(x) - target_survival_time)
        else:
            raise ValueError("Must specify either target_risk or target_survival_time")

        counterfactuals = []
        distances = []

        for _ in range(n_counterfactuals):
            # Generate counterfactual using specified method
            if self.method == "gradient":
                cf, dist = self._gradient_optimization(
                    X.values[0], target_func, distance_metric, max_distance, **kwargs
                )
            elif self.method == "genetic":
                cf, dist = self._genetic_algorithm(
                    X.values[0], target_func, distance_metric, max_distance, **kwargs
                )
            elif self.method == "random":
                cf, dist = self._random_search(
                    X.values[0], target_func, distance_metric, max_distance, **kwargs
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")

            if cf is not None:
                counterfactuals.append(cf)
                distances.append(dist)

        if not counterfactuals:
            return {
                "counterfactuals": [],
                "distances": [],
                "success": False,
                "message": "No valid counterfactuals found within constraints"
            }

        # Convert back to DataFrame
        cf_df = pd.DataFrame(counterfactuals, columns=self.feature_names)

        return {
            "counterfactuals": cf_df,
            "distances": distances,
            "original": X,
            "target_risk": target_risk,
            "target_survival_time": target_survival_time,
            "method": self.method,
            "success": True,
            "metadata": {
                "n_attempts": n_counterfactuals,
                "n_successful": len(counterfactuals),
                "distance_metric": distance_metric,
                "max_distance": max_distance
            }
        }

    def _gradient_optimization(
        self,
        x: np.ndarray,
        target_func: Callable,
        distance_metric: str,
        max_distance: float,
        **kwargs: Any,
    ) -> tuple[np.ndarray | None, float]:
        """Gradient-based counterfactual optimization."""
        x_current = x.copy()
        best_x = x.copy()
        best_score = target_func(x)

        for iteration in range(self.max_iterations):
            # Compute gradient using finite differences
            gradient = self._compute_gradient(x_current, target_func)

            if np.linalg.norm(gradient) < self.tolerance:
                break

            # Gradient descent step
            x_new = x_current - self.step_size * gradient

            # Clip to bounds
            x_new = self._clip_to_bounds(x_new)

            # Check distance constraint
            distance = self._compute_distance(x, x_new, distance_metric)
            if distance > max_distance:
                # Scale back the step
                direction = x_new - x_current
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                    x_new = x_current + direction * (max_distance * 0.9)

            score = target_func(x_new)

            if score < best_score:
                best_score = score
                best_x = x_new.copy()

            x_current = x_new

        final_distance = self._compute_distance(x, best_x, distance_metric)

        if final_distance <= max_distance:
            return best_x, final_distance
        else:
            return None, np.inf

    def _genetic_algorithm(
        self,
        x: np.ndarray,
        target_func: Callable,
        distance_metric: str,
        max_distance: float,
        population_size: int = 50,
        **kwargs: Any,
    ) -> tuple[np.ndarray | None, float]:
        """Genetic algorithm for counterfactual generation."""
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = x.copy()
            # Add random perturbations
            for i, feature in enumerate(self.feature_names):
                if feature in self.bounds:
                    min_val, max_val = self.bounds[feature]
                    if min_val is not None and max_val is not None:
                        individual[i] += np.random.normal(0, (max_val - min_val) * 0.1)
                        individual[i] = np.clip(individual[i], min_val, max_val)
            population.append(individual)

        best_individual = x.copy()
        best_score = target_func(x)

        for generation in range(self.max_iterations // population_size):
            # Evaluate population
            scores = [target_func(ind) for ind in population]

            # Find best individual
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < best_score:
                best_score = scores[min_score_idx]
                best_individual = population[min_score_idx].copy()

            # Selection
            elite_size = population_size // 5
            elite_indices = np.argsort(scores)[:elite_size]
            new_population = [population[i].copy() for i in elite_indices]

            # Crossover and mutation
            while len(new_population) < population_size:
                parent1_idx = np.random.choice(elite_indices)
                parent2_idx = np.random.choice(elite_indices)

                child = (population[parent1_idx] + population[parent2_idx]) / 2

                # Mutation
                mutation_rate = 0.1
                for i in range(len(child)):
                    if np.random.random() < mutation_rate:
                        feature = self.feature_names[i]
                        if feature in self.bounds:
                            min_val, max_val = self.bounds[feature]
                            if min_val is not None and max_val is not None:
                                child[i] += np.random.normal(0, (max_val - min_val) * 0.05)
                                child[i] = np.clip(child[i], min_val, max_val)

                new_population.append(child)

            population = new_population

        final_distance = self._compute_distance(x, best_individual, distance_metric)

        if final_distance <= max_distance:
            return best_individual, final_distance
        else:
            return None, np.inf

    def _random_search(
        self,
        x: np.ndarray,
        target_func: Callable,
        distance_metric: str,
        max_distance: float,
        n_samples: int = 1000,
        **kwargs: Any,
    ) -> tuple[np.ndarray | None, float]:
        """Random search for counterfactual generation."""
        best_x = None
        best_score = np.inf
        best_distance = np.inf

        for _ in range(n_samples):
            # Generate random perturbation
            x_candidate = x.copy()

            for i, feature in enumerate(self.feature_names):
                if feature in self.bounds:
                    min_val, max_val = self.bounds[feature]
                    if min_val is not None and max_val is not None:
                        # Add perturbation within bounds
                        perturbation = np.random.uniform(-0.2, 0.2) * (max_val - min_val)
                        x_candidate[i] += perturbation
                        x_candidate[i] = np.clip(x_candidate[i], min_val, max_val)

            distance = self._compute_distance(x, x_candidate, distance_metric)
            if distance > max_distance:
                continue

            score = target_func(x_candidate)

            if score < best_score:
                best_score = score
                best_x = x_candidate.copy()
                best_distance = distance

        return (best_x, best_distance) if best_x is not None else (None, np.inf)

    def _compute_gradient(self, x: np.ndarray, target_func: Callable) -> np.ndarray:
        """Compute gradient using finite differences."""
        gradient = np.zeros_like(x)
        h = 1e-6

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i] += h
            x_minus[i] -= h

            f_plus = target_func(x_plus)
            f_minus = target_func(x_minus)

            gradient[i] = (f_plus - f_minus) / (2 * h)

        return gradient

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip values to feature bounds."""
        x_clipped = x.copy()

        for i, feature in enumerate(self.feature_names):
            if feature in self.bounds:
                min_val, max_val = self.bounds[feature]
                if min_val is not None:
                    x_clipped[i] = max(x_clipped[i], min_val)
                if max_val is not None:
                    x_clipped[i] = min(x_clipped[i], max_val)

        return x_clipped

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray, metric: str) -> float:
        """Compute distance between two instances."""
        if metric == "euclidean":
            return np.linalg.norm(x1 - x2)
        elif metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def _predict_risk(self, x: np.ndarray) -> float:
        """Predict risk for a single instance."""
        x_df = pd.DataFrame([x], columns=self.feature_names)
        return self.model.predict_risk(x_df)[0]

    def _predict_survival_time(self, x: np.ndarray) -> float:
        """Predict median survival time for a single instance."""
        x_df = pd.DataFrame([x], columns=self.feature_names)

        # Use risk as a proxy for survival time (higher risk = shorter survival)
        # This is a simplification - in practice, you'd want to use the actual survival function
        risk = self._predict_risk(x)

        # Convert risk to approximate survival time (inverse relationship)
        # Higher risk means shorter expected survival time
        # This is a rough approximation for demonstration
        if risk > 0.5:
            # High risk -> shorter survival time
            survival_time = 365 * (1 - risk)  # Max 1 year for very high risk
        else:
            # Lower risk -> longer survival time
            survival_time = 365 * 2 * (1 - risk)  # Up to 2 years for low risk

        return max(1, survival_time)  # Minimum 1 day survival

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Ensure input is a DataFrame."""
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names)
        return X.copy()

    def explain_prediction(
        self,
        X: pd.DataFrame | np.ndarray,
        n_counterfactuals: int = 3,
        risk_targets: list[float] | None = None,
        time_targets: list[float] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate comprehensive explanation with multiple counterfactuals.

        Args:
            X: Input features to explain
            n_counterfactuals: Number of counterfactuals per target
            risk_targets: List of target risk values
            time_targets: List of target survival times
            **kwargs: Additional arguments for generate_counterfactual

        Returns:
            Comprehensive explanation dictionary
        """
        X = self._ensure_dataframe(X)
        original_risk = self._predict_risk(X.values[0])

        explanations = {
            "original_prediction": {
                "risk": original_risk,
                "features": X.iloc[0].to_dict()
            },
            "counterfactuals": [],
            "summary": {}
        }

        # Default targets if none provided
        if risk_targets is None and time_targets is None:
            # Generate targets around the original prediction
            risk_targets = [
                max(0.1, original_risk * 0.5),  # Much lower risk
                max(0.1, original_risk * 0.8),  # Moderately lower risk
                min(0.9, original_risk * 1.2),  # Moderately higher risk
                min(0.9, original_risk * 2.0),  # Much higher risk
            ]

        # Generate counterfactuals for risk targets
        if risk_targets:
            for target_risk in risk_targets:
                cf_result = self.generate_counterfactual(
                    X,
                    target_risk=target_risk,
                    n_counterfactuals=n_counterfactuals,
                    **kwargs
                )

                if cf_result["success"]:
                    explanations["counterfactuals"].extend([
                        {
                            "type": "risk_target",
                            "target_risk": target_risk,
                            "counterfactual": cf,
                            "distance": dist,
                            "changes": self._compute_feature_changes(X.iloc[0], cf)
                        }
                        for cf, dist in zip(cf_result["counterfactuals"], cf_result["distances"])
                    ])

        # Generate counterfactuals for time targets
        if time_targets:
            for target_time in time_targets:
                cf_result = self.generate_counterfactual(
                    X,
                    target_survival_time=target_time,
                    n_counterfactuals=n_counterfactuals,
                    **kwargs
                )

                if cf_result["success"]:
                    explanations["counterfactuals"].extend([
                        {
                            "type": "time_target",
                            "target_time": target_time,
                            "counterfactual": cf,
                            "distance": dist,
                            "changes": self._compute_feature_changes(X.iloc[0], cf)
                        }
                        for cf, dist in zip(cf_result["counterfactuals"], cf_result["distances"])
                    ])

        # Generate summary statistics
        if explanations["counterfactuals"]:
            distances = [cf["distance"] for cf in explanations["counterfactuals"]]
            explanations["summary"] = {
                "total_counterfactuals": len(explanations["counterfactuals"]),
                "avg_distance": np.mean(distances),
                "min_distance": np.min(distances),
                "max_distance": np.max(distances),
                "risk_targets_covered": len(set(cf["target_risk"] for cf in explanations["counterfactuals"] if "target_risk" in cf)),
                "time_targets_covered": len(set(cf["target_time"] for cf in explanations["counterfactuals"] if "target_time" in cf))
            }

        return explanations

    def _compute_feature_changes(self, original: pd.Series, counterfactual: pd.Series) -> dict[str, float]:
        """Compute the changes in each feature."""
        changes = {}

        for feature in self.feature_names:
            orig_val = original[feature]
            cf_val = counterfactual[feature]
            changes[feature] = cf_val - orig_val

        return changes


class CausalInference:
    """Causal inference methods for survival analysis."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        treatment_features: list[str] | None = None,
    ):
        """Initialize causal inference analyzer.

        Args:
            model: Trained survival model
            feature_names: List of all feature names
            treatment_features: Features that represent treatments/interventions
        """
        self.model = model
        self.feature_names = feature_names
        self.treatment_features = treatment_features or []

    def estimate_treatment_effect(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment_feature: str,
        treatment_values: list[float],
        outcome_type: str = "risk",
        n_bootstrap: int = 100,
    ) -> dict[str, Any]:
        """Estimate treatment effect for different treatment values.

        Args:
            X: Input features
            treatment_feature: Feature representing the treatment
            treatment_values: Values of treatment to compare
            outcome_type: Type of outcome ('risk', 'survival_time')
            n_bootstrap: Number of bootstrap samples for confidence intervals

        Returns:
            Dictionary with treatment effect estimates
        """
        X = pd.DataFrame(X, columns=self.feature_names) if isinstance(X, np.ndarray) else X.copy()

        results = {
            "treatment_feature": treatment_feature,
            "treatment_values": treatment_values,
            "outcomes": {},
            "effects": {},
            "bootstrap_ci": {}
        }

        # Compute outcomes for each treatment value
        base_outcomes = []

        for i, treatment_val in enumerate(treatment_values):
            X_treated = X.copy()
            X_treated[treatment_feature] = treatment_val

            if outcome_type == "risk":
                outcomes = self.model.predict_risk(X_treated)
            else:  # survival_time
                # Approximate survival time (this is a simplification)
                survival_funcs = self.model.predict_survival_function(X_treated, times=[np.inf])
                outcomes = [np.median([t for t in range(1, 365*10) if sf(t) > 0.5]) for sf in survival_funcs]

            results["outcomes"][treatment_val] = {
                "mean": np.mean(outcomes),
                "std": np.std(outcomes),
                "values": outcomes
            }

            if i == 0:
                base_outcomes = outcomes

        # Compute treatment effects (comparing to baseline)
        baseline = treatment_values[0]

        for treatment_val in treatment_values[1:]:
            effect = np.array(results["outcomes"][treatment_val]["values"]) - np.array(base_outcomes)
            results["effects"][treatment_val] = {
                "mean": np.mean(effect),
                "std": np.std(effect),
                "values": effect.tolist()
            }

        # Bootstrap confidence intervals
        if n_bootstrap > 0:
            bootstrap_effects = {tv: [] for tv in treatment_values[1:]}

            for _ in range(n_bootstrap):
                # Bootstrap sample indices
                indices = np.random.choice(len(X), size=len(X), replace=True)

                for treatment_val in treatment_values[1:]:
                    X_boot = X.iloc[indices].copy()
                    X_boot[treatment_feature] = treatment_val

                    if outcome_type == "risk":
                        outcomes_boot = self.model.predict_risk(X_boot)
                    else:
                        survival_funcs = self.model.predict_survival_function(X_boot, times=[np.inf])
                        outcomes_boot = [np.median([t for t in range(1, 365*10) if sf(t) > 0.5]) for sf in survival_funcs]

                    # Effect relative to baseline treatment
                    X_base_boot = X.iloc[indices].copy()
                    X_base_boot[treatment_feature] = baseline

                    if outcome_type == "risk":
                        base_outcomes_boot = self.model.predict_risk(X_base_boot)
                    else:
                        survival_funcs_base = self.model.predict_survival_function(X_base_boot, times=[np.inf])
                        base_outcomes_boot = [np.median([t for t in range(1, 365*10) if sf(t) > 0.5]) for sf in survival_funcs_base]

                    effect_boot = np.mean(outcomes_boot) - np.mean(base_outcomes_boot)
                    bootstrap_effects[treatment_val].append(effect_boot)

            # Compute confidence intervals
            for treatment_val in treatment_values[1:]:
                effects = np.array(bootstrap_effects[treatment_val])
                results["bootstrap_ci"][treatment_val] = {
                    "lower_95": np.percentile(effects, 2.5),
                    "upper_95": np.percentile(effects, 97.5),
                    "median": np.median(effects)
                }

        return results

    def feature_importance_causal(
        self,
        X: pd.DataFrame | np.ndarray,
        outcome_type: str = "risk",
        method: str = "permutation",
        n_permutations: int = 100,
    ) -> dict[str, float]:
        """Compute causal feature importance using permutation importance.

        Args:
            X: Input features
            outcome_type: Type of outcome to explain
            method: Importance method ('permutation', 'shap')
            n_permutations: Number of permutations for permutation importance

        Returns:
            Dictionary mapping feature names to importance scores
        """
        X = pd.DataFrame(X, columns=self.feature_names) if isinstance(X, np.ndarray) else X.copy()

        if method == "shap" and SHAP_AVAILABLE:
            return self._shap_importance(X, outcome_type)
        else:
            return self._permutation_importance(X, outcome_type, n_permutations)

    def _shap_importance(self, X: pd.DataFrame, outcome_type: str) -> dict[str, float]:
        """Compute SHAP-based feature importance."""
        # Create a wrapper function for SHAP
        def predict_fn(x):
            x_df = pd.DataFrame(x, columns=self.feature_names)
            if outcome_type == "risk":
                return self.model.predict_risk(x_df)
            else:
                # For survival time, use a proxy based on risk
                risks = self.model.predict_risk(x_df)
                return 1 / (1 + risks)  # Higher risk = lower survival time

        # Use SHAP explainer
        try:
            if hasattr(self.model, '_booster') and self.model._booster is not None:
                # XGBoost model
                explainer = shap.Explainer(self.model._booster)
                shap_values = explainer(X.values)
            else:
                # General model wrapper
                explainer = shap.Explainer(predict_fn, X.values)
                shap_values = explainer(X.values)

            # Compute mean absolute SHAP values
            importance = np.mean(np.abs(shap_values.values), axis=0)
            return dict(zip(self.feature_names, importance))

        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}. Falling back to permutation importance.")
            return self._permutation_importance(X, outcome_type, 100)

    def _permutation_importance(
        self,
        X: pd.DataFrame,
        outcome_type: str,
        n_permutations: int,
    ) -> dict[str, float]:
        """Compute permutation-based feature importance."""
        # Get baseline performance
        if outcome_type == "risk":
            baseline_scores = self.model.predict_risk(X)
            baseline_perf = np.mean(baseline_scores)
        else:
            # For survival time, use negative risk as proxy
            baseline_risks = self.model.predict_risk(X)
            baseline_perf = -np.mean(baseline_risks)

        importance_scores = {}

        for feature in self.feature_names:
            permuted_scores = []

            for _ in range(n_permutations):
                # Create permuted dataset
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)

                # Compute performance with permuted feature
                if outcome_type == "risk":
                    scores = self.model.predict_risk(X_permuted)
                    perf = np.mean(scores)
                else:
                    risks = self.model.predict_risk(X_permuted)
                    perf = -np.mean(risks)

                permuted_scores.append(perf)

            # Importance is the drop in performance
            mean_permuted_perf = np.mean(permuted_scores)
            importance = abs(baseline_perf - mean_permuted_perf)
            importance_scores[feature] = importance

        return importance_scores


class SurvivalModelWrapper:
    """Wrapper for a survival model pipeline to make it compatible with DiCE."""

    def __init__(self, pipeline: Pipeline, risk_threshold: float):
        self.pipeline = pipeline
        self.risk_threshold = risk_threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts high/low risk class."""
        risk_scores = self.pipeline.predict(X)
        return (risk_scores > self.risk_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probability of being in the high/low risk classes."""
        # DiCE requires probabilities for two classes [P(low_risk), P(high_risk)]
        risk_scores = self.pipeline.predict(X)
        is_high_risk = (risk_scores > self.risk_threshold).astype(int)
        
        # Create a (n_samples, 2) array
        probs = np.zeros((len(X), 2))
        probs[:, 1] = is_high_risk  # P(high_risk)
        probs[:, 0] = 1 - is_high_risk # P(low_risk)
        return probs


def generate_cf_explanations(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_surv: np.ndarray,
    features_to_vary: list[str],
    n_examples: int,
    sample_size: int,
    output_path: Path,
):
    """
    Generates and saves counterfactual explanations for high-risk instances.
    """
    console.print("🎲 Generating counterfactual explanations...")

    # 1. Prepare data and model for DiCE
    risk_scores = pipeline.predict(X)
    risk_threshold = np.median(risk_scores)

    # Add outcome to the dataframe for DiCE
    df_for_dice = X.copy()
    outcome_name = "is_high_risk"
    df_for_dice[outcome_name] = (risk_scores > risk_threshold).astype(int)

    # 2. Initialize DiCE explainer
    d = dice_ml.Data(dataframe=df_for_dice, continuous_features=list(X.columns), outcome_name=outcome_name)
    backend_model = SurvivalModelWrapper(pipeline, risk_threshold)
    m = dice_ml.Model(model=backend_model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    # 3. Select high-risk instances to explain
    high_risk_indices = np.where(risk_scores > risk_threshold)[0]
    query_indices = np.random.choice(high_risk_indices, size=min(sample_size, len(high_risk_indices)), replace=False)
    query_instances = X.iloc[query_indices]

    # 4. Generate counterfactuals
    counterfactuals = exp.generate_counterfactuals(
        query_instances,
        total_CFs=n_examples,
        desired_class="opposite",
        features_to_vary=features_to_vary,
    )
    
    # 5. Save the explanations
    cf_json = json.loads(counterfactuals.to_json())
    with open(output_path, "w") as f:
        json.dump(cf_json, f, indent=2)

    console.print(f"✅ Counterfactuals saved to [bold cyan]{output_path}[/bold cyan]")


def create_counterfactual_explainer(
    model,
    feature_names: list[str],
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    **kwargs: Any,
) -> CounterfactualExplainer:
    """Create a counterfactual explainer with sensible defaults.

    Args:
        model: Trained survival model
        feature_names: List of feature names
        feature_ranges: Feature value ranges for constraints
        **kwargs: Additional arguments for CounterfactualExplainer

    Returns:
        Configured CounterfactualExplainer instance
    """
    return CounterfactualExplainer(
        model=model,
        feature_names=feature_names,
        feature_ranges=feature_ranges,
        **kwargs
    )
