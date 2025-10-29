# 📚 Examples Gallery

This directory contains practical examples showing how to use Clinical Survival ML for different clinical scenarios and use cases.

## 🏥 Clinical Scenarios

### 1. ICU Mortality Prediction (`icu_survival/`)
**Scenario**: Predict 30-day mortality for ICU patients using SOFA scores and vital signs.

```bash
cd examples/icu_survival/
clinical-ml run --config configs/icu_params.yaml
```

**Key Features**:
- Real-time risk stratification
- Counterfactual explanations for treatment decisions
- Integration with EHR systems

### 2. Cancer Survival Analysis (`cancer_survival/`)
**Scenario**: Predict survival outcomes for cancer patients based on staging and biomarkers.

```bash
cd examples/cancer_survival/
clinical-ml run --config configs/cancer_params.yaml
```

**Key Features**:
- Multi-time horizon predictions (1-year, 5-year survival)
- Treatment effect estimation
- Clinical trial design support

### 3. Cardiovascular Risk Assessment (`cardio_risk/`)
**Scenario**: Assess cardiovascular event risk using clinical and genetic factors.

```bash
cd examples/cardio_risk/
clinical-ml run --config configs/cardio_params.yaml
```

**Key Features**:
- Risk stratification for preventive care
- Lifestyle intervention recommendations
- Population health insights

## 🎯 Technical Examples

### 4. Large Dataset Handling (`large_dataset/`)
**Scenario**: Training on datasets with 100k+ patients efficiently.

```bash
cd examples/large_dataset/
clinical-ml run --config configs/large_params.yaml --max-memory-gb 32
```

**Key Features**:
- Automatic memory optimization
- Dataset partitioning for memory efficiency
- GPU acceleration for faster training

### 5. Model Deployment (`deployment/`)
**Scenario**: Deploying models as production APIs with monitoring.

```bash
cd examples/deployment/
clinical-ml register-model --model models/survival_model.pkl --model-name production_model
clinical-ml deploy-model --version-id v1.0.0 --environment production
```

**Key Features**:
- REST API deployment
- Model versioning and rollback
- Performance monitoring and drift detection

### 6. Advanced Interpretability (`interpretability/`)
**Scenario**: Deep dive into model explanations for regulatory compliance.

```bash
cd examples/interpretability/
clinical-ml clinical-interpret --config configs/interpret_params.yaml --output-format html
clinical-ml counterfactual --model xgb_cox --target-risk 0.2 --n-counterfactuals 5
```

**Key Features**:
- SHAP explanations with clinical context
- Counterfactual analysis for treatment planning
- Regulatory compliance reporting

## 🚀 Getting Started with Examples

1. **Browse the examples**:
   ```bash
   ls examples/
   ```

2. **Choose an example** that matches your use case

3. **Run the example**:
   ```bash
   cd examples/your_chosen_example/
   clinical-ml run --config configs/params.yaml
   ```

4. **Explore the results**:
   ```bash
   open results/report.html
   ```

## 📖 Example Structure

Each example includes:

- **`data/`**: Sample datasets (anonymized)
- **`configs/`**: Configuration files tailored to the use case
- **`notebooks/`**: Jupyter notebooks with detailed explanations
- **`README.md`**: Step-by-step guide for the specific example
- **`scripts/`**: Additional utility scripts

## 🛠️ Creating Your Own Example

To create a new example for your specific use case:

1. **Copy an existing example**:
   ```bash
   cp -r examples/template/ examples/your_usecase/
   ```

2. **Customize the configuration**:
   - Update `configs/params.yaml` for your data structure
   - Modify `configs/features.yaml` for your feature types
   - Adjust model parameters in `configs/model_grid.yaml`

3. **Add your data**:
   - Place your CSV data in `data/your_data.csv`
   - Create `data/metadata.yaml` describing column types

4. **Document your example**:
   - Update `README.md` with your specific use case
   - Add a Jupyter notebook explaining the approach

## 🤝 Contributing Examples

We welcome new examples! Please:

1. **Use realistic, anonymized data** (never include PHI)
2. **Include comprehensive documentation**
3. **Follow the established structure**
4. **Test thoroughly** before submitting

Submit your example as a Pull Request with:
- Clear description of the clinical scenario
- Step-by-step instructions
- Expected outputs and interpretations
- Any special configuration requirements





