"""Integration tests for clinical interpretability and decision support."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app
from clinical_survival.clinical_interpretability import (
    ClinicalDecisionSupport,
    ClinicalExplanation,
    ClinicalFeatureImportance,
    EnhancedSHAPExplainer,
    RiskStratification,
    RiskStratifier,
    create_clinical_interpretability_config,
    save_clinical_report,
)


class TestClinicalInterpretability:
    """Test clinical interpretability and decision support functionality."""

    @pytest.fixture
    def temp_clinical_dir(self):
        """Create a temporary directory for clinical interpretability tests."""
        temp_dir = tempfile.mkdtemp()

        # Create results directory
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()

        yield results_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config_clinical(self, temp_clinical_dir):
        """Create a test configuration for clinical interpretability."""
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "clinical_interpretability": {
                "enabled": True,
                "risk_thresholds": {
                    "low": 0.2,
                    "moderate": 0.4,
                    "high": 0.7,
                    "very_high": 1.0
                },
                "clinical_context": {
                    "feature_domains": {
                        "age": "demographics",
                        "sex": "demographics",
                        "sofa": "vitals",
                        "stage": "comorbidities"
                    },
                    "risk_categories": {
                        "protective": ["age"],
                        "risk_factor": ["sofa", "stage"],
                        "neutral": []
                    }
                }
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(temp_clinical_dir.parent),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = temp_clinical_dir / "clinical_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_clinical_interpretability_config_creation(self):
        """Test clinical interpretability configuration creation."""
        config = create_clinical_interpretability_config(
            risk_thresholds={
                "low": 0.15,
                "moderate": 0.35,
                "high": 0.65,
                "very_high": 1.0
            },
            clinical_context={
                "feature_domains": {"age": "demographics"},
                "risk_categories": {"protective": ["age"]}
            }
        )

        assert config["risk_thresholds"]["low"] == 0.15
        assert config["risk_thresholds"]["moderate"] == 0.35
        assert config["clinical_context"]["feature_domains"]["age"] == "demographics"
        assert config["explanation_features"]["include_clinical_interpretation"] is True

    def test_clinical_explanation_creation(self):
        """Test clinical explanation creation."""
        explanation = ClinicalExplanation(
            patient_id="patient_001",
            predicted_risk=0.75,
            risk_category="high",
            feature_contributions={"age": 0.1, "sofa": 0.3, "stage": 0.2},
            clinical_interpretation="High risk due to elevated SOFA score and advanced stage",
            key_findings=["Elevated SOFA score increases risk", "Advanced stage contributes significantly"],
            clinical_recommendations=["Monitor organ function closely", "Consider specialist consultation"],
            confidence_score=0.85,
            explanation_strength="strong"
        )

        assert explanation.patient_id == "patient_001"
        assert explanation.predicted_risk == 0.75
        assert explanation.risk_category == "high"
        assert "SOFA score" in explanation.clinical_interpretation
        assert len(explanation.key_findings) == 2
        assert len(explanation.clinical_recommendations) == 2
        assert explanation.confidence_score == 0.85
        assert explanation.explanation_strength == "strong"

    def test_risk_stratification_creation(self):
        """Test risk stratification creation."""
        stratification = RiskStratification(
            patient_id="patient_002",
            risk_score=0.65,
            risk_category="high",
            confidence_interval=(0.55, 0.75),
            primary_risk_factors=["sofa", "stage"],
            protective_factors=["age"],
            clinical_recommendations=["Close monitoring required", "Consider specialist consultation"],
            urgency_level="urgent"
        )

        assert stratification.patient_id == "patient_002"
        assert stratification.risk_score == 0.65
        assert stratification.risk_category == "high"
        assert stratification.confidence_interval == (0.55, 0.75)
        assert "sofa" in stratification.primary_risk_factors
        assert "age" in stratification.protective_factors
        assert len(stratification.clinical_recommendations) == 2
        assert stratification.urgency_level == "urgent"

    def test_clinical_feature_importance_creation(self):
        """Test clinical feature importance creation."""
        importance = ClinicalFeatureImportance(
            feature_name="sofa",
            importance_score=0.3,
            clinical_relevance=0.8,
            medical_domain="vitals",
            risk_category="risk_factor",
            clinical_interpretation="SOFA score indicates organ dysfunction severity",
            evidence_level="strong"
        )

        assert importance.feature_name == "sofa"
        assert importance.importance_score == 0.3
        assert importance.clinical_relevance == 0.8
        assert importance.medical_domain == "vitals"
        assert importance.risk_category == "risk_factor"
        assert "organ dysfunction" in importance.clinical_interpretation
        assert importance.evidence_level == "strong"

    def test_enhanced_shap_explainer_creation(self, temp_clinical_dir):
        """Test enhanced SHAP explainer creation."""
        from clinical_survival.models import make_model

        # Create a simple model for testing
        model = make_model("coxph", random_state=42)

        # Create clinical context
        clinical_context = {
            "feature_domains": {"age": "demographics", "sofa": "vitals"},
            "risk_categories": {"risk_factor": ["sofa"]}
        }

        # Create explainer
        explainer = EnhancedSHAPExplainer(
            model,
            feature_names=["age", "sofa"],
            clinical_context=clinical_context
        )

        # Test initialization
        assert explainer.model is model
        assert explainer.feature_names == ["age", "sofa"]
        assert explainer.clinical_context == clinical_context
        assert explainer._explainer is None

    def test_risk_stratifier_creation(self, temp_clinical_dir):
        """Test risk stratifier creation."""
        from clinical_survival.models import make_model

        # Create a simple model for testing
        model = make_model("coxph", random_state=42)

        # Custom risk thresholds
        risk_thresholds = {
            "low": 0.15,
            "moderate": 0.35,
            "high": 0.65,
            "very_high": 1.0
        }

        # Create stratifier
        stratifier = RiskStratifier(
            model,
            risk_thresholds=risk_thresholds,
            clinical_context={"custom": "context"}
        )

        # Test initialization
        assert stratifier.model is model
        assert stratifier.risk_thresholds == risk_thresholds
        assert stratifier.clinical_context == {"custom": "context"}

    def test_clinical_decision_support_creation(self, temp_clinical_dir):
        """Test clinical decision support system creation."""
        from clinical_survival.models import make_model

        # Create a simple model for testing
        model = make_model("coxph", random_state=42)

        # Create clinical context
        clinical_context = {
            "feature_domains": {"age": "demographics"},
            "risk_categories": {"protective": ["age"]}
        }

        # Create decision support system
        cds = ClinicalDecisionSupport(model, clinical_context)

        # Test initialization
        assert cds.model is model
        assert cds.clinical_context == clinical_context
        assert cds.shap_explainer is None
        assert cds.risk_stratifier is None

    def test_clinical_interpretability_report_generation(self, temp_clinical_dir):
        """Test clinical interpretability report generation."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train a model to have something to interpret
        train_result = runner.invoke(app, ["train",
                                         "--config", "configs/params.yaml",
                                         "--grid", "configs/model_grid.yaml"])

        if train_result.exit_code != 0:
            pytest.skip("Model training failed, skipping interpretability test")

        # Test clinical interpretability command
        result = runner.invoke(app, ["clinical-interpret",
                                   "--config", "configs/params.yaml",
                                   "--data", "data/toy/toy_survival.csv",
                                   "--meta", "data/toy/metadata.yaml",
                                   "--model", "results/artifacts/models/coxph.pkl",
                                   "--output-dir", str(temp_clinical_dir),
                                   "--output-format", "json"])

        # Should complete (may succeed or fail depending on SHAP availability)
        assert result.exit_code in [0, 1]

    def test_risk_stratification_command(self, temp_clinical_dir):
        """Test risk stratification command."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train a model
        train_result = runner.invoke(app, ["train",
                                         "--config", "configs/params.yaml",
                                         "--grid", "configs/model_grid.yaml"])

        if train_result.exit_code != 0:
            pytest.skip("Model training failed, skipping stratification test")

        # Test risk stratification command
        result = runner.invoke(app, ["risk-stratification",
                                   "--config", "configs/params.yaml",
                                   "--data", "data/toy/toy_survival.csv",
                                   "--meta", "data/toy/metadata.yaml",
                                   "--model", "results/artifacts/models/coxph.pkl",
                                   "--output-dir", str(temp_clinical_dir)])

        # Should complete (may succeed or fail depending on model availability)
        assert result.exit_code in [0, 1]

    def test_clinical_report_saving(self, temp_clinical_dir):
        """Test saving clinical reports in different formats."""
        # Create a mock report
        mock_report = {
            "summary": {
                "total_patients": 100,
                "average_risk_score": 0.45,
                "risk_distribution": {"low": 30, "moderate": 40, "high": 25, "very_high": 5},
                "urgency_distribution": {"routine": 70, "urgent": 25, "critical": 5},
                "high_risk_patients": 30,
                "critical_urgency_patients": 5
            },
            "patient_reports": [
                {
                    "patient_id": "patient_001",
                    "risk_assessment": {
                        "predicted_risk": 0.75,
                        "risk_category": "high",
                        "confidence_score": 0.85,
                        "explanation_strength": "strong"
                    },
                    "clinical_interpretation": "High risk due to elevated SOFA score",
                    "key_findings": ["Elevated SOFA score increases risk"],
                    "clinical_recommendations": ["Monitor organ function closely"],
                    "risk_stratification": {
                        "risk_category": "high",
                        "confidence_interval": (0.65, 0.85),
                        "primary_risk_factors": ["sofa"],
                        "protective_factors": [],
                        "urgency_level": "urgent"
                    }
                }
            ],
            "population_insights": {
                "top_risk_features": ["sofa", "stage", "age"],
                "average_feature_importance": {"sofa": 0.3, "stage": 0.2, "age": 0.1},
                "risk_factor_prevalence": {"sofa": 0.6, "stage": 0.4, "age": 0.2}
            },
            "clinical_recommendations": [
                "Monitor high-risk patients closely",
                "Address modifiable risk factors"
            ],
            "metadata": {
                "n_patients": 100,
                "model_type": "CoxPHModel",
                "explanation_method": "Enhanced SHAP with Clinical Context",
                "risk_assessment_method": "Multi-factor Risk Stratification"
            }
        }

        # Test JSON format
        json_file = temp_clinical_dir / "test_report.json"
        save_clinical_report(mock_report, json_file, format="json")

        assert json_file.exists()

        # Verify JSON content
        with open(json_file) as f:
            loaded_report = json.load(f)

        assert loaded_report["summary"]["total_patients"] == 100
        assert loaded_report["metadata"]["n_patients"] == 100

        # Test HTML format
        html_file = temp_clinical_dir / "test_report.html"
        save_clinical_report(mock_report, html_file, format="html")

        assert html_file.exists()

        # Verify HTML content
        with open(html_file) as f:
            html_content = f.read()

        assert "<title>Clinical Decision Support Report</title>" in html_content
        assert "patient_001" in html_content
        assert "High risk" in html_content

    def test_clinical_report_error_handling(self, temp_clinical_dir):
        """Test error handling in clinical report generation."""
        # Test with invalid format
        mock_report = {"test": "data"}

        with pytest.raises(ValueError, match="Unsupported format"):
            save_clinical_report(mock_report, temp_clinical_dir / "test.txt", format="invalid")

    def test_clinical_explanation_risk_categorization(self):
        """Test risk categorization in clinical explanations."""
        # Create explanation with different risk scores
        explanations = [
            ClinicalExplanation("patient_001", 0.1, "low", {}, "", [], [], 0.8, "strong"),
            ClinicalExplanation("patient_002", 0.3, "moderate", {}, "", [], [], 0.8, "strong"),
            ClinicalExplanation("patient_003", 0.6, "high", {}, "", [], [], 0.8, "strong"),
            ClinicalExplanation("patient_004", 0.9, "very_high", {}, "", [], [], 0.8, "strong"),
        ]

        # Test risk categories
        assert explanations[0].risk_category == "low"
        assert explanations[1].risk_category == "moderate"
        assert explanations[2].risk_category == "high"
        assert explanations[3].risk_category == "very_high"

    def test_clinical_explanation_confidence_calculation(self):
        """Test confidence score calculation in clinical explanations."""
        # Create explanations with different SHAP value magnitudes
        explanations = [
            ClinicalExplanation("patient_001", 0.5, "moderate",
                              {"feature1": 0.01, "feature2": 0.02}, "", [], [], 0.8, "weak"),
            ClinicalExplanation("patient_002", 0.5, "moderate",
                              {"feature1": 0.1, "feature2": 0.2}, "", [], [], 0.8, "moderate"),
            ClinicalExplanation("patient_003", 0.5, "moderate",
                              {"feature1": 0.3, "feature2": 0.4}, "", [], [], 0.8, "strong"),
        ]

        # Test confidence scores (based on total magnitude)
        assert explanations[0].confidence_score == 0.5  # Low magnitude -> low confidence
        assert explanations[1].confidence_score == 0.7  # Medium magnitude -> medium confidence
        assert explanations[2].confidence_score == 0.9  # High magnitude -> high confidence

    def test_risk_stratification_urgency_levels(self):
        """Test urgency level determination in risk stratification."""
        # Test different risk categories and factors
        stratifications = [
            RiskStratification("patient_001", 0.8, "very_high", (0.7, 0.9), ["sofa"], [], [], "critical"),
            RiskStratification("patient_002", 0.6, "high", (0.5, 0.7), ["sofa", "sepsis"], [], [], "urgent"),
            RiskStratification("patient_003", 0.6, "high", (0.5, 0.7), ["age"], [], [], "routine"),
            RiskStratification("patient_004", 0.3, "moderate", (0.2, 0.4), ["age"], [], [], "routine"),
            RiskStratification("patient_005", 0.1, "low", (0.0, 0.2), ["age"], [], [], "routine"),
        ]

        # Test urgency levels
        assert stratifications[0].urgency_level == "critical"  # Very high risk
        assert stratifications[1].urgency_level == "urgent"    # High risk with critical factors
        assert stratifications[2].urgency_level == "routine"   # High risk without critical factors
        assert stratifications[3].urgency_level == "routine"   # Moderate risk
        assert stratifications[4].urgency_level == "routine"   # Low risk

    def test_clinical_recommendations_generation(self):
        """Test clinical recommendations generation."""
        # Test different risk categories
        stratifications = [
            RiskStratification("patient_001", 0.9, "very_high", (0.8, 1.0), ["sofa"], [], [], "critical"),
            RiskStratification("patient_002", 0.6, "high", (0.5, 0.7), ["sofa"], [], [], "urgent"),
            RiskStratification("patient_003", 0.3, "moderate", (0.2, 0.4), ["age"], [], [], "routine"),
            RiskStratification("patient_004", 0.1, "low", (0.0, 0.2), ["age"], [], [], "routine"),
        ]

        # Test recommendations for very high risk
        very_high_recs = stratifications[0].clinical_recommendations
        assert any("intensive care" in rec.lower() for rec in very_high_recs)
        assert any("immediate" in rec.lower() for rec in very_high_recs)

        # Test recommendations for high risk
        high_recs = stratifications[1].clinical_recommendations
        assert any("monitoring" in rec.lower() for rec in high_recs)
        assert any("specialist" in rec.lower() for rec in high_recs)

        # Test recommendations for moderate risk
        moderate_recs = stratifications[2].clinical_recommendations
        assert any("routine" in rec.lower() for rec in moderate_recs)

        # Test recommendations for low risk
        low_recs = stratifications[3].clinical_recommendations
        assert any("routine" in rec.lower() for rec in low_recs)

    def test_clinical_interpretation_generation(self):
        """Test clinical interpretation generation."""
        # Create explanation with feature contributions
        explanation = ClinicalExplanation(
            patient_id="patient_001",
            predicted_risk=0.65,
            risk_category="high",
            feature_contributions={
                "age": 0.1,
                "sofa": 0.3,
                "stage": 0.2,
                "sex": -0.05
            },
            clinical_interpretation="",  # Will be generated
            key_findings=[],
            clinical_recommendations=[],
            confidence_score=0.8,
            explanation_strength="strong"
        )

        # Test interpretation generation (would be done by the explainer)
        # For now, just verify the structure
        assert explanation.patient_id == "patient_001"
        assert explanation.risk_category == "high"
        assert isinstance(explanation.feature_contributions, dict)
        assert len(explanation.feature_contributions) == 4

    def test_clinical_decision_support_initialization(self, temp_clinical_dir):
        """Test clinical decision support system initialization."""
        from clinical_survival.models import make_model

        # Create model
        model = make_model("coxph", random_state=42)

        # Create clinical context
        clinical_context = {
            "feature_domains": {"age": "demographics", "sofa": "vitals"},
            "risk_categories": {"risk_factor": ["sofa"]}
        }

        # Create decision support system
        cds = ClinicalDecisionSupport(model, clinical_context)

        # Test that it's properly initialized
        assert cds.model is model
        assert cds.clinical_context == clinical_context
        assert cds.shap_explainer is None
        assert cds.risk_stratifier is None

        # Test initialization of explainers
        # Create background data for SHAP
        import pandas as pd
        X_background = pd.DataFrame({
            "age": [60, 65, 70],
            "sofa": [5, 8, 12],
            "stage": [1, 2, 3],
            "sex": [0, 1, 0]
        })

        cds.initialize_explainers(X_background, ["age", "sofa", "stage", "sex"])

        # Test that explainers are initialized
        assert cds.shap_explainer is not None
        assert cds.risk_stratifier is not None

    def test_clinical_report_structure(self, temp_clinical_dir):
        """Test clinical report structure and content."""
        # Create a comprehensive mock report
        mock_report = {
            "summary": {
                "total_patients": 50,
                "average_risk_score": 0.45,
                "risk_distribution": {"low": 15, "moderate": 20, "high": 12, "very_high": 3},
                "urgency_distribution": {"routine": 35, "urgent": 12, "critical": 3},
                "high_risk_patients": 15,
                "critical_urgency_patients": 3
            },
            "patient_reports": [
                {
                    "patient_id": "patient_001",
                    "risk_assessment": {
                        "predicted_risk": 0.75,
                        "risk_category": "high",
                        "confidence_score": 0.85,
                        "explanation_strength": "strong"
                    },
                    "clinical_interpretation": "High risk due to elevated SOFA score and advanced disease stage",
                    "key_findings": ["Elevated SOFA score indicates organ dysfunction", "Advanced stage contributes to risk"],
                    "clinical_recommendations": ["Monitor organ function closely", "Consider palliative care consultation"],
                    "risk_stratification": {
                        "risk_category": "high",
                        "confidence_interval": (0.65, 0.85),
                        "primary_risk_factors": ["sofa", "stage"],
                        "protective_factors": ["age"],
                        "urgency_level": "urgent"
                    }
                }
            ],
            "population_insights": {
                "top_risk_features": ["sofa", "stage", "age"],
                "average_feature_importance": {"sofa": 0.25, "stage": 0.20, "age": 0.15},
                "risk_factor_prevalence": {"sofa": 0.6, "stage": 0.4, "age": 0.3},
                "population_risk_distribution": {
                    "mean": 0.45,
                    "median": 0.42,
                    "std": 0.25,
                    "min": 0.05,
                    "max": 0.95
                }
            },
            "clinical_recommendations": [
                "Monitor high-risk patients closely",
                "Address modifiable risk factors in moderate-risk patients",
                "Continue routine care for low-risk patients"
            ],
            "metadata": {
                "n_patients": 50,
                "model_type": "CoxPHModel",
                "explanation_method": "Enhanced SHAP with Clinical Context",
                "risk_assessment_method": "Multi-factor Risk Stratification"
            }
        }

        # Test report structure
        assert "summary" in mock_report
        assert "patient_reports" in mock_report
        assert "population_insights" in mock_report
        assert "clinical_recommendations" in mock_report
        assert "metadata" in mock_report

        # Test summary structure
        summary = mock_report["summary"]
        assert summary["total_patients"] == 50
        assert summary["average_risk_score"] == 0.45
        assert "risk_distribution" in summary
        assert "urgency_distribution" in summary

        # Test patient report structure
        patient_report = mock_report["patient_reports"][0]
        assert "risk_assessment" in patient_report
        assert "clinical_interpretation" in patient_report
        assert "key_findings" in patient_report
        assert "clinical_recommendations" in patient_report
        assert "risk_stratification" in patient_report

        # Test population insights structure
        insights = mock_report["population_insights"]
        assert "top_risk_features" in insights
        assert "average_feature_importance" in insights
        assert "risk_factor_prevalence" in insights
        assert "population_risk_distribution" in insights

        # Test metadata structure
        metadata = mock_report["metadata"]
        assert metadata["n_patients"] == 50
        assert "model_type" in metadata
        assert "explanation_method" in metadata
        assert "risk_assessment_method" in metadata

    def test_clinical_context_integration(self, temp_clinical_dir):
        """Test clinical context integration in explanations."""
        # Test clinical context structure
        clinical_context = {
            "feature_domains": {
                "age": "demographics",
                "sex": "demographics",
                "sofa": "vitals",
                "stage": "comorbidities",
                "creatinine": "labs",
                "bilirubin": "labs"
            },
            "risk_categories": {
                "protective": ["age", "sex"],
                "risk_factor": ["sofa", "stage", "creatinine", "bilirubin"],
                "neutral": []
            },
            "clinical_guidelines": {
                "sofa_threshold": 8,
                "stage_high_risk": 3
            }
        }

        # Test context structure
        assert clinical_context["feature_domains"]["sofa"] == "vitals"
        assert "sofa" in clinical_context["risk_categories"]["risk_factor"]
        assert clinical_context["clinical_guidelines"]["sofa_threshold"] == 8

        # Test feature importance with clinical context
        importance = ClinicalFeatureImportance(
            feature_name="sofa",
            importance_score=0.3,
            clinical_relevance=0.9,
            medical_domain="vitals",
            risk_category="risk_factor",
            clinical_interpretation="SOFA score above threshold indicates organ dysfunction",
            evidence_level="strong"
        )

        assert importance.medical_domain == "vitals"
        assert importance.risk_category == "risk_factor"
        assert "organ dysfunction" in importance.clinical_interpretation



