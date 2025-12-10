"""
Model comparison and selection module.

This module provides automated comparison of trained models and selection
of the best model based on configurable criteria:
- Statistical comparison of metrics
- Cross-validation stability analysis
- Model complexity consideration
- Ensemble generation from top models

Usage:
    from clinical_survival.model_selection import (
        ModelComparator,
        compare_models,
        select_best_model,
    )
    
    comparator = ModelComparator(metrics_dict, cv_results)
    comparison = comparator.compare()
    best_model = comparator.select_best(criterion="concordance")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from clinical_survival.logging_config import get_logger

# Module logger
logger = get_logger(__name__)

# Console for output
console = Console()


class SelectionCriterion(Enum):
    """Criteria for model selection."""
    BEST_METRIC = "best_metric"           # Highest primary metric
    BEST_AVERAGE = "best_average"         # Best average across metrics
    MOST_STABLE = "most_stable"           # Lowest CV variance
    PARETO_OPTIMAL = "pareto_optimal"     # Best trade-off
    ENSEMBLE_TOP_K = "ensemble_top_k"     # Combine top K models


@dataclass
class ModelScore:
    """Score for a single model."""
    model_name: str
    primary_metric: float
    metrics: Dict[str, float]
    cv_scores: Optional[List[float]] = None
    cv_std: Optional[float] = None
    rank: int = 0
    
    @property
    def stability(self) -> float:
        """Return stability score (inverse of CV std)."""
        if self.cv_std is None or self.cv_std == 0:
            return 1.0
        return 1.0 / (1.0 + self.cv_std)


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    model_scores: List[ModelScore]
    primary_metric_name: str
    best_model: str
    comparison_table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_model": self.best_model,
            "primary_metric": self.primary_metric_name,
            "rankings": [
                {
                    "rank": s.rank,
                    "model": s.model_name,
                    "score": s.primary_metric,
                    "stability": s.stability,
                }
                for s in sorted(self.model_scores, key=lambda x: x.rank)
            ],
            "metrics": self.comparison_table,
            "statistical_tests": self.statistical_tests,
        }


@dataclass
class SelectionResult:
    """Result of model selection."""
    selected_model: str
    selection_criterion: SelectionCriterion
    score: float
    reasoning: str
    alternatives: List[Tuple[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_model": self.selected_model,
            "criterion": self.selection_criterion.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "alternatives": [
                {"model": m, "score": s} for m, s in self.alternatives
            ],
        }


class ModelComparator:
    """
    Compare and select the best model from a set of trained models.
    
    Usage:
        comparator = ModelComparator(
            metrics={
                "coxph": {"concordance": 0.72, "ibs": 0.18},
                "rsf": {"concordance": 0.75, "ibs": 0.16},
            },
            cv_results={
                "coxph": [0.70, 0.73, 0.71, 0.74, 0.72],
                "rsf": [0.74, 0.76, 0.73, 0.77, 0.75],
            },
        )
        
        comparison = comparator.compare(primary_metric="concordance")
        best = comparator.select_best()
    """
    
    def __init__(
        self,
        metrics: Dict[str, Dict[str, float]],
        cv_results: Optional[Dict[str, List[float]]] = None,
        higher_is_better: Dict[str, bool] = None,
    ):
        """
        Initialize the comparator.
        
        Args:
            metrics: Dict mapping model names to their metric dictionaries
            cv_results: Optional dict mapping model names to CV fold scores
            higher_is_better: Dict mapping metric names to optimization direction
        """
        self.metrics = metrics
        self.cv_results = cv_results or {}
        self.higher_is_better = higher_is_better or {
            "concordance": True,
            "concordance_index_censored": True,
            "c_index": True,
            "ibs": False,
            "integrated_brier_score": False,
            "brier": False,
            "brier_score": False,
        }
        
        self._comparison_result: Optional[ComparisonResult] = None
    
    def compare(
        self,
        primary_metric: str = "concordance",
    ) -> ComparisonResult:
        """
        Compare all models and generate rankings.
        
        Args:
            primary_metric: Metric to use for primary ranking
            
        Returns:
            ComparisonResult with rankings and statistics
        """
        model_scores: List[ModelScore] = []
        
        # Calculate scores for each model
        for model_name, model_metrics in self.metrics.items():
            primary_value = model_metrics.get(primary_metric, 0.0)
            
            # Get CV statistics if available
            cv_scores = self.cv_results.get(model_name)
            cv_std = np.std(cv_scores) if cv_scores else None
            
            score = ModelScore(
                model_name=model_name,
                primary_metric=primary_value,
                metrics=model_metrics,
                cv_scores=cv_scores,
                cv_std=cv_std,
            )
            model_scores.append(score)
        
        # Sort and rank
        higher_better = self.higher_is_better.get(primary_metric, True)
        model_scores.sort(
            key=lambda x: x.primary_metric,
            reverse=higher_better,
        )
        
        for i, score in enumerate(model_scores):
            score.rank = i + 1
        
        # Build comparison table
        comparison_table = {}
        for score in model_scores:
            comparison_table[score.model_name] = {
                **score.metrics,
                "cv_std": score.cv_std or 0.0,
                "stability": score.stability,
            }
        
        # Perform statistical tests if CV results available
        statistical_tests = {}
        if len(self.cv_results) >= 2:
            statistical_tests = self._perform_statistical_tests()
        
        best_model = model_scores[0].model_name if model_scores else ""
        
        self._comparison_result = ComparisonResult(
            model_scores=model_scores,
            primary_metric_name=primary_metric,
            best_model=best_model,
            comparison_table=comparison_table,
            statistical_tests=statistical_tests,
        )
        
        logger.info(
            f"Model comparison complete. Best: {best_model}",
            extra={
                "best_model": best_model,
                "n_models": len(model_scores),
                "primary_metric": primary_metric,
            },
        )
        
        return self._comparison_result
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        tests = {}
        
        model_names = list(self.cv_results.keys())
        if len(model_names) < 2:
            return tests
        
        try:
            from scipy import stats
            
            # Friedman test for multiple comparisons
            if len(model_names) >= 3:
                cv_arrays = [self.cv_results[m] for m in model_names]
                # Ensure all arrays have same length
                min_len = min(len(a) for a in cv_arrays)
                cv_arrays = [a[:min_len] for a in cv_arrays]
                
                try:
                    stat, p_value = stats.friedmanchisquare(*cv_arrays)
                    tests["friedman"] = {
                        "statistic": stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                except Exception:
                    pass
            
            # Pairwise Wilcoxon tests
            pairwise = {}
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    scores1 = self.cv_results[m1]
                    scores2 = self.cv_results[m2]
                    
                    # Ensure same length
                    min_len = min(len(scores1), len(scores2))
                    scores1 = scores1[:min_len]
                    scores2 = scores2[:min_len]
                    
                    try:
                        stat, p_value = stats.wilcoxon(scores1, scores2)
                        pairwise[f"{m1}_vs_{m2}"] = {
                            "statistic": stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                        }
                    except Exception:
                        pass
            
            if pairwise:
                tests["pairwise_wilcoxon"] = pairwise
                
        except ImportError:
            logger.warning("scipy not available for statistical tests")
        
        return tests
    
    def select_best(
        self,
        criterion: SelectionCriterion = SelectionCriterion.BEST_METRIC,
        **kwargs,
    ) -> SelectionResult:
        """
        Select the best model based on the specified criterion.
        
        Args:
            criterion: Selection criterion to use
            **kwargs: Additional arguments for specific criteria
            
        Returns:
            SelectionResult with selected model and reasoning
        """
        if self._comparison_result is None:
            self.compare()
        
        scores = self._comparison_result.model_scores
        
        if criterion == SelectionCriterion.BEST_METRIC:
            return self._select_by_metric(scores)
        
        elif criterion == SelectionCriterion.BEST_AVERAGE:
            return self._select_by_average(scores)
        
        elif criterion == SelectionCriterion.MOST_STABLE:
            return self._select_by_stability(scores)
        
        elif criterion == SelectionCriterion.PARETO_OPTIMAL:
            return self._select_pareto(scores)
        
        elif criterion == SelectionCriterion.ENSEMBLE_TOP_K:
            k = kwargs.get("k", 3)
            return self._select_ensemble(scores, k)
        
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def _select_by_metric(self, scores: List[ModelScore]) -> SelectionResult:
        """Select model with best primary metric."""
        best = scores[0]
        alternatives = [(s.model_name, s.primary_metric) for s in scores[1:3]]
        
        return SelectionResult(
            selected_model=best.model_name,
            selection_criterion=SelectionCriterion.BEST_METRIC,
            score=best.primary_metric,
            reasoning=f"Highest {self._comparison_result.primary_metric_name}: {best.primary_metric:.4f}",
            alternatives=alternatives,
        )
    
    def _select_by_average(self, scores: List[ModelScore]) -> SelectionResult:
        """Select model with best average across all metrics."""
        # Normalize metrics and compute average
        avg_scores = []
        for score in scores:
            normalized = []
            for metric_name, value in score.metrics.items():
                higher_better = self.higher_is_better.get(metric_name, True)
                all_values = [s.metrics.get(metric_name, 0) for s in scores]
                min_val, max_val = min(all_values), max(all_values)
                
                if max_val > min_val:
                    norm = (value - min_val) / (max_val - min_val)
                    if not higher_better:
                        norm = 1 - norm
                    normalized.append(norm)
            
            avg = np.mean(normalized) if normalized else 0
            avg_scores.append((score.model_name, avg, score))
        
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_avg, best_score = avg_scores[0]
        
        return SelectionResult(
            selected_model=best_name,
            selection_criterion=SelectionCriterion.BEST_AVERAGE,
            score=best_avg,
            reasoning=f"Best normalized average across metrics: {best_avg:.4f}",
            alternatives=[(n, s) for n, s, _ in avg_scores[1:3]],
        )
    
    def _select_by_stability(self, scores: List[ModelScore]) -> SelectionResult:
        """Select most stable model (lowest CV variance)."""
        stable_scores = sorted(scores, key=lambda x: x.stability, reverse=True)
        best = stable_scores[0]
        
        return SelectionResult(
            selected_model=best.model_name,
            selection_criterion=SelectionCriterion.MOST_STABLE,
            score=best.stability,
            reasoning=f"Most stable (CV std: {best.cv_std:.4f})",
            alternatives=[(s.model_name, s.stability) for s in stable_scores[1:3]],
        )
    
    def _select_pareto(self, scores: List[ModelScore]) -> SelectionResult:
        """Select Pareto-optimal model (best trade-off between metric and stability)."""
        # Simple Pareto: combine normalized metric and stability
        combined = []
        for score in scores:
            norm_metric = score.rank / len(scores)  # Lower is better (rank 1 = 0)
            norm_stability = 1 - score.stability    # Lower is better
            
            # Combined score (lower is better)
            pareto_score = 0.7 * norm_metric + 0.3 * norm_stability
            combined.append((score.model_name, pareto_score, score))
        
        combined.sort(key=lambda x: x[1])
        best_name, best_pareto, best_score = combined[0]
        
        return SelectionResult(
            selected_model=best_name,
            selection_criterion=SelectionCriterion.PARETO_OPTIMAL,
            score=1 - best_pareto,  # Convert back to higher-is-better
            reasoning=f"Best trade-off: rank {best_score.rank}, stability {best_score.stability:.3f}",
            alternatives=[(n, 1-s) for n, s, _ in combined[1:3]],
        )
    
    def _select_ensemble(self, scores: List[ModelScore], k: int) -> SelectionResult:
        """Recommend ensemble of top-K models."""
        top_k = scores[:k]
        model_names = [s.model_name for s in top_k]
        avg_score = np.mean([s.primary_metric for s in top_k])
        
        return SelectionResult(
            selected_model=f"ensemble({', '.join(model_names)})",
            selection_criterion=SelectionCriterion.ENSEMBLE_TOP_K,
            score=avg_score,
            reasoning=f"Ensemble of top {k} models for improved robustness",
            alternatives=[(s.model_name, s.primary_metric) for s in top_k],
        )
    
    def print_comparison(self) -> None:
        """Print comparison results as a formatted table."""
        if self._comparison_result is None:
            self.compare()
        
        result = self._comparison_result
        
        # Create table
        table = Table(title="🏆 Model Comparison")
        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("Model", style="bold")
        table.add_column(result.primary_metric_name.title(), justify="right")
        table.add_column("CV Std", justify="right")
        table.add_column("Stability", justify="right")
        
        for score in result.model_scores:
            rank_str = "🥇" if score.rank == 1 else ("🥈" if score.rank == 2 else str(score.rank))
            
            table.add_row(
                rank_str,
                score.model_name,
                f"{score.primary_metric:.4f}",
                f"{score.cv_std:.4f}" if score.cv_std else "N/A",
                f"{score.stability:.3f}",
            )
        
        console.print()
        console.print(table)
        
        # Print statistical test results if available
        if result.statistical_tests:
            console.print()
            console.print("[bold]Statistical Tests:[/bold]")
            
            if "friedman" in result.statistical_tests:
                f_test = result.statistical_tests["friedman"]
                sig = "✓" if f_test["significant"] else "✗"
                console.print(
                    f"  Friedman test: p={f_test['p_value']:.4f} {sig}"
                )
        
        console.print()


# =============================================================================
# Convenience Functions
# =============================================================================


def compare_models(
    metrics: Dict[str, Dict[str, float]],
    cv_results: Optional[Dict[str, List[float]]] = None,
    primary_metric: str = "concordance",
    print_results: bool = True,
) -> ComparisonResult:
    """
    Convenience function to compare models.
    
    Args:
        metrics: Model metrics dictionary
        cv_results: Optional CV results
        primary_metric: Primary metric for ranking
        print_results: Whether to print comparison table
        
    Returns:
        ComparisonResult
    """
    comparator = ModelComparator(metrics, cv_results)
    result = comparator.compare(primary_metric)
    
    if print_results:
        comparator.print_comparison()
    
    return result


def select_best_model(
    metrics: Dict[str, Dict[str, float]],
    cv_results: Optional[Dict[str, List[float]]] = None,
    criterion: str = "best_metric",
    **kwargs,
) -> SelectionResult:
    """
    Convenience function to select the best model.
    
    Args:
        metrics: Model metrics dictionary
        cv_results: Optional CV results
        criterion: Selection criterion name
        **kwargs: Additional arguments for selection
        
    Returns:
        SelectionResult
    """
    comparator = ModelComparator(metrics, cv_results)
    comparator.compare()
    
    criterion_enum = SelectionCriterion(criterion)
    return comparator.select_best(criterion_enum, **kwargs)





