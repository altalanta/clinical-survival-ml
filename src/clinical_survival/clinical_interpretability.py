"""Advanced clinical interpretability and decision support for survival models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from clinical_survival.logging_utils import log_function_call
from clinical_survival.models import BaseSurvivalModel
from clinical_survival.utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class ClinicalFeatureImportance:
    """Clinical feature importance with medical context."""

    feature_name: str
    importance_score: float
    clinical_relevance: float  # 0-1 scale of clinical importance
    medical_domain: str  # e.g., "demographics", "vitals", "labs", "comorbidities"
    risk_category: str  # "protective", "risk_factor", "neutral"
    clinical_interpretation: str
    evidence_level: str  # "strong", "moderate", "weak", "unknown"


@dataclass
class RiskStratification:
    """Patient risk stratification with clinical categories."""

    patient_id: str
    risk_score: float  # 0-1 scale
    risk_category: str  # "low", "moderate", "high", "very_high"
    confidence_interval: Tuple[float, float]
    primary_risk_factors: List[str]
    protective_factors: List[str]
    clinical_recommendations: List[str]
    urgency_level: str  # "routine", "urgent", "critical"


@dataclass
class ClinicalExplanation:
    """Comprehensive clinical explanation for a prediction."""

    patient_id: str
    predicted_risk: float
    risk_category: str
    feature_contributions: Dict[str, float]
    clinical_interpretation: str
    key_findings: List[str]
    clinical_recommendations: List[str]
    confidence_score: float
    explanation_strength: str  # "strong", "moderate", "weak"


class EnhancedSHAPExplainer:
    """Enhanced SHAP explainer with clinical context."""

    def __init__(
        self,
        model: BaseSurvivalModel,
        feature_names: List[str],
        clinical_context: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.clinical_context = clinical_context or {}
        self._explainer = None
        self._background_data = None

    def fit(self, X_background: pd.DataFrame, **kwargs) -> EnhancedSHAPExplainer:
        """Fit the SHAP explainer with background data."""
        try:
            import shap

            # Use a subset for background if dataset is large
            if len(X_background) > 1000:
                X_background = X_background.sample(n=1000, random_state=42)

            self._background_data = X_background.values

            # Create appropriate explainer based on model type
            if hasattr(self.model, '_booster'):
                # XGBoost model
                self._explainer = shap.Explainer(self.model._booster, self._background_data)
            else:
                # Other models - use TreeExplainer if possible, otherwise KernelExplainer
                try:
                    self._explainer = shap.TreeExplainer(self.model.model)
                except:
                    self._explainer = shap.KernelExplainer(
                        self._predict_function, self._background_data[:100]  # Smaller background for kernel
                    )

            logger.info("Enhanced SHAP explainer fitted successfully")
            return self

        except ImportError:
            logger.error("SHAP not available. Install with: pip install shap")
            return self
        except Exception as e:
            logger.error(f"Failed to fit SHAP explainer: {e}")
            return self

    def _predict_function(self, X: np.ndarray) -> np.ndarray:
        """Prediction function for SHAP explainer."""
        X_df = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_risk(X_df)

    def explain_prediction(
        self,
        X: pd.DataFrame,
        max_evals: int = 1000,
        clinical_context: bool = True
    ) -> Dict[str, Any]:
        """Generate enhanced SHAP explanation for predictions."""
        try:
            import shap

            if self._explainer is None:
                raise RuntimeError("Explainer not fitted. Call fit() first.")

            X_values = X.values

            # Calculate SHAP values
            if hasattr(self._explainer, 'shap_values'):
                # Tree explainer
                shap_values = self._explainer.shap_values(X_values)
            else:
                # Kernel explainer
                shap_values = self._explainer.shap_values(X_values, max_evals=max_evals)

            # Convert to DataFrame for easier handling
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)

            # Calculate prediction and base value
            predictions = self.model.predict_risk(X)
            base_value = self._explainer.expected_value if hasattr(self._explainer, 'expected_value') else 0

            explanations = []
            for i in range(len(X)):
                explanation = self._create_clinical_explanation(
                    X.iloc[i:i+1],
                    shap_df.iloc[i:i+1],
                    predictions[i],
                    base_value,
                    clinical_context
                )
                explanations.append(explanation)

            return {
                "explanations": explanations,
                "shap_values": shap_df,
                "predictions": predictions,
                "base_value": base_value,
                "feature_names": self.feature_names
            }

        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}")
            return {"error": str(e)}

    def _create_clinical_explanation(
        self,
        patient_data: pd.DataFrame,
        shap_values: pd.Series,
        prediction: float,
        base_value: float,
        clinical_context: bool = True
    ) -> ClinicalExplanation:
        """Create clinical explanation from SHAP values."""

        # Determine risk category
        risk_category = self._categorize_risk(prediction)

        # Identify key contributors
        feature_contributions = {}
        for feature, shap_val in shap_values.items():
            feature_contributions[feature] = float(shap_val)

        # Generate clinical interpretation
        clinical_interpretation = self._generate_clinical_interpretation(
            feature_contributions, prediction, risk_category
        )

        # Generate key findings
        key_findings = self._extract_key_findings(feature_contributions, patient_data)

        # Generate clinical recommendations
        clinical_recommendations = self._generate_clinical_recommendations(
            feature_contributions, risk_category, patient_data
        )

        # Calculate confidence
        confidence_score = self._calculate_confidence(shap_values)

        # Determine explanation strength
        explanation_strength = self._assess_explanation_strength(shap_values)

        return ClinicalExplanation(
            patient_id=str(patient_data.index[0]) if hasattr(patient_data.index, '__getitem__') else "unknown",
            predicted_risk=prediction,
            risk_category=risk_category,
            feature_contributions=feature_contributions,
            clinical_interpretation=clinical_interpretation,
            key_findings=key_findings,
            clinical_recommendations=clinical_recommendations,
            confidence_score=confidence_score,
            explanation_strength=explanation_strength
        )

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into clinical categories."""
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.4:
            return "moderate"
        elif risk_score < 0.7:
            return "high"
        else:
            return "very_high"

    def _generate_clinical_interpretation(
        self,
        contributions: Dict[str, float],
        risk_score: float,
        risk_category: str
    ) -> str:
        """Generate clinical interpretation of feature contributions."""

        # Sort features by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = sorted_features[:3]  # Top 3 contributors

        # Build interpretation
        if risk_category == "low":
            interpretation = f"Low risk prediction ({risk_score".2f"}) primarily due to "
        elif risk_category == "moderate":
            interpretation = f"Moderate risk prediction ({risk_score".2f"}) influenced by "
        elif risk_category == "high":
            interpretation = f"High risk prediction ({risk_score".2f"}) driven by "
        else:
            interpretation = f"Very high risk prediction ({risk_score".2f"}) strongly affected by "

        feature_descriptions = []
        for feature, contribution in top_features:
            if contribution > 0:
                direction = "increases"
            else:
                direction = "decreases"

            feature_descriptions.append(f"{feature} ({direction} risk)")

        interpretation += " and ".join(feature_descriptions)

        return interpretation

    def _extract_key_findings(
        self,
        contributions: Dict[str, float],
        patient_data: pd.DataFrame
    ) -> List[str]:
        """Extract key clinical findings from feature contributions."""

        findings = []

        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, contribution in sorted_features[:5]:  # Top 5
            if abs(contribution) > 0.05:  # Only significant contributions
                if contribution > 0:
                    impact = "increases"
                    finding = f"Elevated {feature} increases predicted risk"
                else:
                    impact = "decreases"
                    finding = f"Lower {feature} decreases predicted risk"

                findings.append(finding)

        return findings

    def _generate_clinical_recommendations(
        self,
        contributions: Dict[str, float],
        risk_category: str,
        patient_data: pd.DataFrame
    ) -> List[str]:
        """Generate clinical recommendations based on risk factors."""

        recommendations = []

        # Risk-based recommendations
        if risk_category in ["high", "very_high"]:
            recommendations.append("Consider intensive monitoring and early intervention")
            recommendations.append("Evaluate for clinical trial eligibility")

        # Feature-specific recommendations
        for feature, contribution in contributions.items():
            if contribution > 0.1:  # Strong risk factor
                if "age" in feature.lower():
                    recommendations.append("Age is a significant risk factor - consider age-appropriate care")
                elif "sofa" in feature.lower():
                    recommendations.append("High SOFA score indicates organ dysfunction - intensive care evaluation recommended")
                elif "stage" in feature.lower():
                    recommendations.append("Advanced disease stage - consider palliative care consultation")

        if not recommendations:
            recommendations.append("Continue standard clinical management")

        return recommendations

    def _calculate_confidence(self, shap_values: pd.Series) -> float:
        """Calculate confidence score for explanation."""
        # Simple confidence based on explanation consistency
        # In practice, this would be more sophisticated
        total_magnitude = abs(shap_values).sum()

        if total_magnitude > 0.5:
            return 0.9  # High confidence
        elif total_magnitude > 0.2:
            return 0.7  # Moderate confidence
        else:
            return 0.5  # Low confidence

    def _assess_explanation_strength(self, shap_values: pd.Series) -> str:
        """Assess overall strength of explanation."""
        total_magnitude = abs(shap_values).sum()

        if total_magnitude > 0.7:
            return "strong"
        elif total_magnitude > 0.4:
            return "moderate"
        else:
            return "weak"


class RiskStratifier:
    """Clinical risk stratification system."""

    def __init__(
        self,
        model: BaseSurvivalModel,
        risk_thresholds: Optional[Dict[str, float]] = None,
        clinical_context: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.risk_thresholds = risk_thresholds or {
            "low": 0.2,
            "moderate": 0.4,
            "high": 0.7,
            "very_high": 1.0
        }
        self.clinical_context = clinical_context or {}

    def stratify_patients(self, X: pd.DataFrame) -> List[RiskStratification]:
        """Stratify patients into risk categories with clinical context."""

        # Get predictions
        risk_scores = self.model.predict_risk(X)

        # Get feature contributions if explainer available
        feature_contributions = self._get_feature_contributions(X)

        stratifications = []
        for i in range(len(X)):
            stratification = self._create_risk_stratification(
                X.iloc[i:i+1],
                risk_scores[i],
                feature_contributions.get(i, {})
            )
            stratifications.append(stratification)

        return stratifications

    def _get_feature_contributions(self, X: pd.DataFrame) -> List[Dict[str, float]]:
        """Get feature contributions for risk stratification."""
        # For now, return empty dicts - would integrate with SHAP explainer
        return [{} for _ in range(len(X))]

    def _create_risk_stratification(
        self,
        patient_data: pd.DataFrame,
        risk_score: float,
        contributions: Dict[str, float]
    ) -> RiskStratification:

        # Determine risk category
        risk_category = "unknown"
        for category, threshold in self.risk_thresholds.items():
            if risk_score <= threshold:
                risk_category = category
                break

        # Calculate confidence interval (simplified)
        confidence_interval = (max(0, risk_score - 0.1), min(1, risk_score + 0.1))

        # Identify primary risk and protective factors
        primary_risk_factors = []
        protective_factors = []

        for feature, contribution in contributions.items():
            if contribution > 0.05:
                primary_risk_factors.append(feature)
            elif contribution < -0.05:
                protective_factors.append(feature)

        # Generate clinical recommendations
        clinical_recommendations = self._generate_risk_based_recommendations(
            risk_category, primary_risk_factors, protective_factors
        )

        # Determine urgency level
        urgency_level = self._determine_urgency_level(risk_category, primary_risk_factors)

        return RiskStratification(
            patient_id=str(patient_data.index[0]) if hasattr(patient_data.index, '__getitem__') else "unknown",
            risk_score=risk_score,
            risk_category=risk_category,
            confidence_interval=confidence_interval,
            primary_risk_factors=primary_risk_factors,
            protective_factors=protective_factors,
            clinical_recommendations=clinical_recommendations,
            urgency_level=urgency_level
        )

    def _generate_risk_based_recommendations(
        self,
        risk_category: str,
        risk_factors: List[str],
        protective_factors: List[str]
    ) -> List[str]:

        recommendations = []

        # Risk category-based recommendations
        if risk_category == "very_high":
            recommendations.extend([
                "Immediate clinical intervention required",
                "Consider intensive care unit admission",
                "Activate rapid response team",
                "Evaluate for advanced therapies"
            ])
        elif risk_category == "high":
            recommendations.extend([
                "Close monitoring required",
                "Consider specialist consultation",
                "Implement preventive measures",
                "Schedule follow-up within 24-48 hours"
            ])
        elif risk_category == "moderate":
            recommendations.extend([
                "Routine monitoring",
                "Address modifiable risk factors",
                "Schedule follow-up as clinically indicated"
            ])
        else:  # low risk
            recommendations.extend([
                "Continue routine care",
                "Monitor for changes in clinical status"
            ])

        # Factor-specific recommendations
        if "age" in risk_factors:
            recommendations.append("Consider age-appropriate screening and interventions")

        if any("sofa" in factor.lower() for factor in risk_factors):
            recommendations.append("Monitor organ function closely - SOFA score elevation indicates dysfunction")

        if any("stage" in factor.lower() for factor in risk_factors):
            recommendations.append("Advanced disease stage - consider palliative care consultation")

        return recommendations

    def _determine_urgency_level(self, risk_category: str, risk_factors: List[str]) -> str:
        """Determine clinical urgency level."""

        # Very high risk is always critical
        if risk_category == "very_high":
            return "critical"

        # High risk with critical factors is urgent
        critical_factors = ["sofa", "ventilator", "sepsis", "shock"]
        if risk_category == "high" and any(factor.lower() in critical_factors for factor in risk_factors):
            return "urgent"

        # Otherwise, routine care
        return "routine"


class ClinicalDecisionSupport:
    """Clinical decision support system for survival predictions."""

    def __init__(
        self,
        model: BaseSurvivalModel,
        clinical_context: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.clinical_context = clinical_context or {}
        self.shap_explainer: Optional[EnhancedSHAPExplainer] = None
        self.risk_stratifier: Optional[RiskStratifier] = None

    def initialize_explainers(
        self,
        X_background: pd.DataFrame,
        feature_names: List[str]
    ) -> ClinicalDecisionSupport:
        """Initialize explanation and stratification systems."""

        # Initialize SHAP explainer
        self.shap_explainer = EnhancedSHAPExplainer(
            self.model, feature_names, self.clinical_context
        ).fit(X_background)

        # Initialize risk stratifier
        self.risk_stratifier = RiskStratifier(
            self.model, clinical_context=self.clinical_context
        )

        logger.info("Clinical decision support system initialized")
        return self

    def generate_clinical_report(
        self,
        X: pd.DataFrame,
        patient_ids: Optional[List[str]] = None,
        include_detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive clinical decision support report."""

        if self.shap_explainer is None or self.risk_stratifier is None:
            raise RuntimeError("Explainers not initialized. Call initialize_explainers() first.")

        # Generate predictions
        predictions = self.model.predict_risk(X)

        # Generate SHAP explanations
        shap_explanations = self.shap_explainer.explain_prediction(X)

        # Generate risk stratifications
        risk_stratifications = self.risk_stratifier.stratify_patients(X)

        # Compile comprehensive report
        report = {
            "summary": self._generate_summary_report(predictions, risk_stratifications),
            "patient_reports": [],
            "population_insights": self._generate_population_insights(X, predictions, shap_explanations),
            "clinical_recommendations": self._generate_population_recommendations(risk_stratifications),
            "metadata": {
                "n_patients": len(X),
                "model_type": type(self.model).__name__,
                "explanation_method": "Enhanced SHAP with Clinical Context",
                "risk_assessment_method": "Multi-factor Risk Stratification"
            }
        }

        # Generate individual patient reports
        for i in range(len(X)):
            patient_report = self._generate_patient_report(
                X.iloc[i:i+1],
                predictions[i],
                shap_explanations["explanations"][i],
                risk_stratifications[i],
                include_detailed_analysis
            )
            report["patient_reports"].append(patient_report)

        return report

    def _generate_summary_report(
        self,
        predictions: np.ndarray,
        stratifications: List[RiskStratification]
    ) -> Dict[str, Any]:

        # Risk distribution
        risk_categories = [s.risk_category for s in stratifications]
        risk_distribution = {
            "low": risk_categories.count("low"),
            "moderate": risk_categories.count("moderate"),
            "high": risk_categories.count("high"),
            "very_high": risk_categories.count("very_high")
        }

        # Average risk score
        avg_risk = float(np.mean(predictions))

        # Urgency distribution
        urgency_levels = [s.urgency_level for s in stratifications]
        urgency_distribution = {
            "routine": urgency_levels.count("routine"),
            "urgent": urgency_levels.count("urgent"),
            "critical": urgency_levels.count("critical")
        }

        return {
            "total_patients": len(predictions),
            "average_risk_score": avg_risk,
            "risk_distribution": risk_distribution,
            "urgency_distribution": urgency_distribution,
            "high_risk_patients": risk_distribution["high"] + risk_distribution["very_high"],
            "critical_urgency_patients": urgency_distribution["critical"]
        }

    def _generate_patient_report(
        self,
        patient_data: pd.DataFrame,
        risk_score: float,
        explanation: ClinicalExplanation,
        stratification: RiskStratification,
        include_detailed_analysis: bool
    ) -> Dict[str, Any]:

        patient_id = explanation.patient_id

        report = {
            "patient_id": patient_id,
            "risk_assessment": {
                "predicted_risk": risk_score,
                "risk_category": explanation.risk_category,
                "confidence_score": explanation.confidence_score,
                "explanation_strength": explanation.explanation_strength
            },
            "clinical_interpretation": explanation.clinical_interpretation,
            "key_findings": explanation.key_findings,
            "clinical_recommendations": explanation.clinical_recommendations,
            "risk_stratification": {
                "risk_category": stratification.risk_category,
                "confidence_interval": stratification.confidence_interval,
                "primary_risk_factors": stratification.primary_risk_factors,
                "protective_factors": stratification.protective_factors,
                "urgency_level": stratification.urgency_level
            }
        }

        if include_detailed_analysis:
            report["detailed_analysis"] = {
                "feature_contributions": explanation.feature_contributions,
                "all_clinical_recommendations": stratification.clinical_recommendations,
                "risk_factors_count": len(stratification.primary_risk_factors),
                "protective_factors_count": len(stratification.protective_factors)
            }

        return report

    def _generate_population_insights(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        shap_explanations: Dict[str, Any]
    ) -> Dict[str, Any]:

        # Aggregate feature importance across population
        all_contributions = []
        for explanation in shap_explanations["explanations"]:
            all_contributions.append(explanation.feature_contributions)

        if all_contributions:
            # Average feature importance
            avg_contributions = pd.DataFrame(all_contributions).mean()

            # Most important features
            top_features = avg_contributions.abs().nlargest(5).index.tolist()

            # Risk factor prevalence
            risk_factor_prevalence = {}
            for feature in top_features:
                # Count how many patients have this as a significant risk factor
                significant_count = sum(
                    1 for contrib in all_contributions
                    if abs(contrib.get(feature, 0)) > 0.05
                )
                risk_factor_prevalence[feature] = significant_count / len(all_contributions)

            return {
                "top_risk_features": top_features,
                "average_feature_importance": avg_contributions.to_dict(),
                "risk_factor_prevalence": risk_factor_prevalence,
                "population_risk_distribution": {
                    "mean": float(np.mean(predictions)),
                    "median": float(np.median(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions))
                }
            }

        return {}

    def _generate_population_recommendations(
        self,
        stratifications: List[RiskStratification]
    ) -> List[str]:

        recommendations = []

        # Count by risk category
        risk_counts = {
            "low": sum(1 for s in stratifications if s.risk_category == "low"),
            "moderate": sum(1 for s in stratifications if s.risk_category == "moderate"),
            "high": sum(1 for s in stratifications if s.risk_category == "high"),
            "very_high": sum(1 for s in stratifications if s.risk_category == "very_high")
        }

        # Population-level recommendations
        if risk_counts["very_high"] > 0:
            recommendations.append(f"Immediate attention needed for {risk_counts['very_high']} very high-risk patients")

        if risk_counts["high"] > risk_counts["moderate"] * 0.5:
            recommendations.append("High proportion of high-risk patients - consider resource allocation review")

        # Aggregate risk factors
        all_risk_factors = []
        for stratification in stratifications:
            all_risk_factors.extend(stratification.primary_risk_factors)

        if all_risk_factors:
            from collections import Counter
            common_factors = Counter(all_risk_factors).most_common(3)

            for factor, count in common_factors:
                if count > len(stratifications) * 0.2:  # Present in >20% of patients
                    recommendations.append(f"Common risk factor '{factor}' affects {count} patients - consider targeted interventions")

        return recommendations


def create_clinical_interpretability_config(
    risk_thresholds: Optional[Dict[str, float]] = None,
    clinical_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create clinical interpretability configuration."""

    return {
        "risk_thresholds": risk_thresholds or {
            "low": 0.2,
            "moderate": 0.4,
            "high": 0.7,
            "very_high": 1.0
        },
        "clinical_context": clinical_context or {},
        "explanation_features": {
            "include_shap_values": True,
            "include_clinical_interpretation": True,
            "include_risk_stratification": True,
            "include_recommendations": True,
            "include_confidence_scores": True
        }
    }


def save_clinical_report(
    report: Dict[str, Any],
    output_path: Path,
    format: str = "json"
) -> None:
    """Save clinical report to file."""

    ensure_dir(output_path.parent)

    if format.lower() == "json":
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    elif format.lower() == "html":
        # Generate HTML report
        html_content = _generate_html_report(report)
        with open(output_path, 'w') as f:
            f.write(html_content)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_html_report(report: Dict[str, Any]) -> str:
    """Generate HTML report from clinical analysis."""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clinical Decision Support Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .summary {{ background: #fff; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 20px; }}
            .patient-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
            .risk-high {{ border-left: 4px solid #dc3545; }}
            .risk-moderate {{ border-left: 4px solid #ffc107; }}
            .risk-low {{ border-left: 4px solid #28a745; }}
            .recommendations {{ background: #e7f3ff; padding: 10px; border-radius: 4px; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Clinical Decision Support Report</h1>
            <p>Generated for {report['metadata']['n_patients']} patients using {report['metadata']['model_type']}</p>
        </div>

        <div class="summary">
            <h2>Population Summary</h2>
            <p><strong>Total Patients:</strong> {report['summary']['total_patients']}</p>
            <p><strong>Average Risk Score:</strong> {report['summary']['average_risk_score']".3f"}</p>
            <p><strong>High-Risk Patients:</strong> {report['summary']['high_risk_patients']}</p>
        </div>

        <h2>Individual Patient Reports</h2>
    """

    for patient_report in report["patient_reports"]:
        risk_class = f"risk-{patient_report['risk_assessment']['risk_category']}"

        html += f"""
        <div class="patient-card {risk_class}">
            <h3>Patient {patient_report['patient_id']}</h3>
            <p><strong>Risk Category:</strong> {patient_report['risk_assessment']['risk_category'].upper()}</p>
            <p><strong>Predicted Risk:</strong> {patient_report['risk_assessment']['predicted_risk']".3f"}</p>
            <p><strong>Confidence:</strong> {patient_report['risk_assessment']['confidence_score']".2f"}</p>

            <div class="recommendations">
                <h4>Clinical Recommendations:</h4>
                <ul>
        """

        for rec in patient_report["clinical_recommendations"]:
            html += f"<li>{rec}</li>"

        html += """
                </ul>
            </div>
        </div>
        """

    html += """
    </body>
    </html>
    """

    return html











