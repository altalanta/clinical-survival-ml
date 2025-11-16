# Example Custom Model Plugin

This directory contains an example of a plugin that adds a new model (`custom_svm`) to the clinical-survival framework.

## Installation

To use this plugin, you need to install it into the same Python environment where `clinical-survival-ml` is installed. Navigate to this directory and run:

```bash
pip install .
```

This command installs the plugin and makes it discoverable by the main application through its `entry_points`.

## Usage

Once installed, you can use the `custom_svm` model just like any of the built-in models. Simply add it to the `models` list in your `configs/params.yaml` file:

```yaml
models:
  - coxph
  - rsf
  - custom_svm  # <-- Your custom model
```

When you run the training pipeline, the framework will automatically discover and use your custom model.

