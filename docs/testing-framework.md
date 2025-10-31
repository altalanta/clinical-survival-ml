# 🧪 Advanced Testing & Quality Assurance Framework

This document provides a comprehensive guide to the advanced testing framework implemented in Clinical Survival ML, designed to ensure model reliability, detect regressions, and validate against industry standards.

## Overview

The testing framework consists of five main components:

1.  **Synthetic Dataset Generation** - Generate realistic clinical datasets for testing
2.  **Performance Regression Testing** - Automated detection of performance degradations
3.  **Cross-Validation Integrity Checking** - Detect data leakage and validate CV setup
4.  **Comprehensive Benchmark Suite** - Compare against other survival analysis libraries
5.  **Property-Based Testing** - Verify code invariants and find edge cases

## 🏗️ Architecture

```
clinical_survival/
├── testing.py                    # Core testing framework
├── cli/commands.py               # CLI command implementations
├── cli/main.py                   # CLI command registration
└── tests/
    ├── baseline_performance.json  # Performance baselines
    ├── performance_regression/    # Regression test results
    ├── cv_integrity/             # CV integrity results
    └── benchmark_results/         # Benchmark results
    └── property/                  # Property-based tests
```

## 🔬 Synthetic Dataset Generation

Generate realistic synthetic datasets for comprehensive testing across different clinical scenarios.

### Available Scenarios

#### ICU Survival Dataset
```bash
clinical-ml synthetic-data --scenario icu --n-samples 2000
```

**Features Generated**:
- **Demographics**: age, sex, BMI
- **Clinical Scores**: SOFA, APACHE, GCS
- **Vital Signs**: temperature, heart rate, blood pressure, oxygen saturation
- **Laboratory Values**: creatinine, bilirubin, platelet count, WBC count, lactate, hemoglobin
- **Outcomes**: Realistic survival times with appropriate censoring rates

#### Cancer Survival Dataset
```bash
clinical-ml synthetic-data --scenario cancer --n-samples 1500
```

**Features Generated**:
- **Demographics**: age, sex
- **Cancer Characteristics**: cancer type, stage, grade, tumor size, lymph nodes, metastasis
- **Treatment Factors**: surgery, chemotherapy, radiation, targeted therapy
- **Comorbidities**: diabetes, hypertension, heart disease, COPD, kidney disease
- **Outcomes**: Cancer-specific survival patterns

#### Cardiovascular Dataset
```bash
clinical-ml synthetic-data --scenario cardiovascular --n-samples 3000
```

**Features Generated**:
- **Risk Factors**: age, sex, blood pressure, cholesterol (total, HDL, LDL), triglycerides
- **Clinical Measurements**: glucose, BMI, smoking status
- **Comorbidities**: diabetes, hypertension, family history
- **Outcomes**: Cardiovascular event times with realistic follow-up periods

### Implementation Details

The synthetic data generator uses realistic statistical distributions and clinical relationships:

```python
from clinical_survival.testing import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator(random_state=42)

# Generate ICU dataset
features, outcomes = generator.generate_icu_dataset(
    n_samples=1000,
    survival_time_range=(1, 365),  # 1 day to 1 year
    censoring_rate=0.3             # 30% censored
)

# Features are pandas DataFrame with realistic clinical values
# Outcomes contain 'time' and 'event' columns
```

## 🧪 Performance Regression Testing

Automated detection of performance degradations to prevent model quality issues.

### How It Works

1. **Baseline Establishment**: First run establishes performance baselines
2. **Regression Detection**: Subsequent runs compare against baselines
3. **Tolerance-Based Alerts**: Configurable tolerance for acceptable performance changes
4. **Multi-Model Testing**: Tests all configured models simultaneously

### Usage Examples

```bash
# Basic regression testing
clinical-ml performance-regression --config configs/params.yaml

# Strict tolerance for production environments
clinical-ml performance-regression --config configs/params.yaml --tolerance 0.02

# Custom baseline file for different environments
clinical-ml performance-regression --baseline-file tests/prod_baselines.json

# Custom output directory for results
clinical-ml performance-regression --output-dir tests/regression_results
```

### Configuration

Performance regression testing uses these key parameters:

```yaml
# In configs/params.yaml or via CLI options
testing:
  regression:
    enabled: true
    tolerance: 0.05           # 5% performance degradation tolerance
    baseline_file: "tests/baseline_performance.json"
    models_to_test: ["coxph", "rsf", "xgb_cox", "xgb_aft"]
```

### Output Format

Results are saved as JSON with detailed metrics:

```json
{
  "coxph_regression": {
    "passed": true,
    "value": 0.782,
    "threshold": 0.743,
    "execution_time": 45.2,
    "error_message": null
  },
  "rsf_regression": {
    "passed": false,
    "value": 0.735,
    "threshold": 0.760,
    "execution_time": 67.8,
    "error_message": "Performance degraded below threshold"
  }
}
```

## 🔍 Cross-Validation Integrity Checking

Detect data leakage and validate cross-validation setup integrity.

### Detection Capabilities

The integrity checker identifies several types of issues:

1. **Data Leakage**: Suspiciously consistent performance across CV folds
2. **Overfitting**: Unrealistically high performance scores
3. **CV Setup Issues**: Inconsistent fold performance patterns

### Usage Examples

```bash
# Basic CV integrity check
clinical-ml cv-integrity --config configs/params.yaml

# More thorough validation with additional folds
clinical-ml cv-integrity --config configs/params.yaml --cv-folds 10

# Custom output for detailed analysis
clinical-ml cv-integrity --output-dir tests/detailed_cv_analysis
```

### Implementation Details

The integrity checker uses statistical analysis of CV fold performance:

```python
from clinical_survival.testing import CrossValidationIntegrityChecker

checker = CrossValidationIntegrityChecker()

# Check CV integrity
result = checker.check_cv_integrity(
    model_constructor=your_model_constructor,
    X=features_dataframe,
    y=outcomes_dataframe,
    cv_folds=5
)

if not result.passed:
    print(f"CV Integrity Issue: {result.error_message}")
```

### Suspicious Patterns Detected

- **Low Variance**: Standard deviation < 0.01 across CV folds (suggests data leakage)
- **High Performance**: Mean concordance > 0.95 (suggests overfitting or leakage)
- **Inconsistent Performance**: Large performance differences between folds

## 🎲 Property-Based Testing

To improve the robustness of our utility functions and data processing pipelines, we use property-based testing with the `Hypothesis` library. Instead of writing tests for specific examples, we write tests that assert general properties of our code and let Hypothesis generate a wide range of data to try and find counterexamples.

### How It Works

1.  **Strategies**: We define "strategies" that tell Hypothesis how to generate random, but valid, data for our functions (e.g., a strategy to create a pandas DataFrame with specific columns and dtypes).
2.  **`@given` Decorator**: The `@given` decorator runs a test function many times, each time with a new piece of data generated by the strategy.
3.  **Assertions**: The test function asserts "properties" that should hold true for all possible inputs. For example, a data splitting function should always produce two disjoint sets of data whose sizes sum to the original.

### Example: Testing `stratified_event_split`

The `stratified_event_split` function is a critical utility for creating training and testing sets. We test it with the following properties:
*   The total number of samples is conserved.
*   The train and test sets are disjoint.
*   The test set size is approximately correct.
*   The event rate is approximately preserved across the splits.

```python
# From: tests/property/test_stratified_split.py

from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column

from clinical_survival.utils import stratified_event_split

@given(
    data_frames([...]),  # Strategy to generate DataFrames
    st.floats(min_value=0.1, max_value=0.9),
    st.integers(min_value=0, max_value=100),
)
def test_stratified_event_split_properties(df, test_size, seed):
    """
    Tests properties of the stratified_event_split function.
    """
    # ... function logic ...
    train_df, test_df = stratified_event_split(...)

    # Assert properties:
    assert len(train_df) + len(test_df) == len(df)
    assert train_df.index.intersection(test_df.index).empty
    # ... other assertions ...
```

This approach is highly effective at finding edge-case bugs in data manipulation and algorithmic code that might be missed by traditional example-based tests.

## 🏆 Comprehensive Benchmark Suite

Compare performance against industry-standard survival analysis libraries.

### Benchmarked Libraries

#### scikit-survival
- **CoxPHSurvivalAnalysis**: Cox proportional hazards model
- **RandomSurvivalForest**: Ensemble survival forest

#### lifelines
- **CoxPHFitter**: Cox proportional hazards implementation

### Usage Examples

```bash
# Run full benchmark suite
clinical-ml benchmark-suite --config configs/params.yaml

# Benchmark against specific libraries only
clinical-ml benchmark-suite --include-sksurv --include-lifelines

# Custom output for comparison analysis
clinical-ml benchmark-suite --output-dir tests/benchmark_comparison
```

### Benchmark Results Format

```json
{
  "sksurv_coxph_concordance": {
    "passed": true,
    "value": 0.776,
    "threshold": 0.6,
    "execution_time": 12.3
  },
  "lifelines_coxph_concordance": {
    "passed": true,
    "value": 0.781,
    "threshold": 0.6,
    "execution_time": 8.7
  },
  "our_coxph_concordance": {
    "passed": true,
    "value": 0.784,
    "threshold": 0.6,
    "execution_time": 15.2
  }
}
```

## 🔄 Integration with CI/CD

The testing framework integrates seamlessly with continuous integration pipelines.

### CI/CD Workflow Example

```yaml
# .github/workflows/quality-assurance.yml
name: Quality Assurance

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Generate synthetic test data
      run: clinical-ml synthetic-data --scenario icu --n-samples 500 --random-state 42

    - name: Run performance regression tests
      run: clinical-ml performance-regression --config configs/params.yaml --tolerance 0.02

    - name: Check CV integrity
      run: clinical-ml cv-integrity --config configs/params.yaml --cv-folds 3

    - name: Run benchmark suite
      run: clinical-ml benchmark-suite --config configs/params.yaml

    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: tests/
```

### Makefile Integration

```makefile
.PHONY: test-quality
test-quality: ## Run comprehensive quality assurance tests
	@echo "🔬 Running synthetic data generation..."
	clinical-ml synthetic-data --scenario icu --n-samples 1000 --random-state 42
	@echo "🧪 Running performance regression tests..."
	clinical-ml performance-regression --config configs/params.yaml --tolerance 0.02
	@echo "🔍 Checking CV integrity..."
	clinical-ml cv-integrity --config configs/params.yaml --cv-folds 3
	@echo "🏆 Running benchmark suite..."
	clinical-ml benchmark-suite --config configs/params.yaml
	@echo "✅ Quality assurance tests completed!"
```

## 📊 Monitoring and Reporting

### Performance Tracking

The framework automatically tracks performance over time:

```bash
# View performance trends
cat tests/baseline_performance.json

# Check recent regression test results
cat tests/performance_regression/regression_results.json
```

### Integration with Monitoring

Testing results integrate with the existing monitoring system:

```bash
# Run tests as part of monitoring workflow
clinical-ml monitor --config configs/params.yaml --data new_data.csv
clinical-ml performance-regression --config configs/params.yaml
clinical-ml cv-integrity --config configs/params.yaml
```

## 🛠️ Customization

### Custom Test Scenarios

You can extend the framework with custom test scenarios:

```python
from clinical_survival.testing import SyntheticDatasetGenerator

class CustomGenerator(SyntheticDatasetGenerator):
    def generate_custom_scenario(self, **kwargs):
        # Your custom synthetic data generation logic
        pass
```

### Custom Performance Metrics

Add custom performance metrics for specialized use cases:

```python
def custom_performance_metric(y_true, y_pred):
    # Your custom metric calculation
    return score
```

## 🚨 Troubleshooting

### Common Issues

#### Performance Regression Detection
- **False Positives**: Adjust tolerance if legitimate performance changes are flagged
- **Missing Baselines**: First run establishes baselines - subsequent runs will detect regressions

#### CV Integrity Issues
- **Data Leakage**: Check preprocessing pipeline for information leakage
- **Overfitting**: Reduce model complexity or increase regularization

#### Benchmark Issues
- **Library Installation**: Ensure optional dependencies are installed for benchmarks
- **Performance Differences**: Expected - different libraries may have different strengths

### Debugging

Enable verbose output for detailed debugging:

```bash
clinical-ml performance-regression --config configs/params.yaml --verbose
clinical-ml cv-integrity --config configs/params.yaml --verbose
clinical-ml benchmark-suite --config configs/params.yaml --verbose
```

## 📈 Best Practices

1. **Regular Testing**: Run quality tests before major releases
2. **Baseline Management**: Update baselines after intentional performance improvements
3. **CI Integration**: Include tests in automated pipelines
4. **Documentation**: Document any intentional performance changes
5. **Alerting**: Set up notifications for regression detection

## 🎯 Benefits

- **🔒 Quality Assurance**: Prevent deployment of degraded models
- **🛡️ Risk Mitigation**: Early detection of data leakage and overfitting
- **📊 Competitive Intelligence**: Understand performance vs. industry standards
- **🔄 Continuous Improvement**: Data-driven model quality enhancement
- **📈 Stakeholder Confidence**: Demonstrable quality metrics for clinical deployment

This advanced testing framework ensures that clinical survival models maintain the highest standards of reliability, accuracy, and safety for healthcare applications.






