import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
import json
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Clinical Survival Analysis Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- App State ---
if 'artifacts_path' not in st.session_state:
    st.session_state.artifacts_path = Path("results/artifacts")

# --- Helper Functions ---
@st.cache_data
def load_model(model_name):
    """Load a trained model pipeline."""
    model_path = st.session_state.artifacts_path / "models" / f"{model_name}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_counterfactuals(model_name):
    """Load counterfactual explanations."""
    cf_path = st.session_state.artifacts_path / "counterfactuals" / f"{model_name}_counterfactuals.json"
    if cf_path.exists():
        with open(cf_path, "r") as f:
            return json.load(f)
    return None

def load_all_model_names():
    """Scan the models directory to find all available models."""
    model_dir = st.session_state.artifacts_path / "models"
    if not model_dir.exists():
        return []
    return [p.stem for p in model_dir.glob("*.joblib")]

# --- Sidebar ---
st.sidebar.title("🩺 Clinical Survival ML")
st.sidebar.markdown("### Navigation")

# Check if artifacts exist
if not st.session_state.artifacts_path.exists():
    st.error("Artifacts directory not found! Please run the training pipeline first (`clinical-ml training run`).")
    st.stop()
    
available_models = load_all_model_names()
if not available_models:
    st.warning("No trained models found in the artifacts directory.")
    st.stop()

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Performance", "Patient-Level Explanations", "Counterfactual Explorer"],
    key="navigation"
)
st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides an interactive way to explore the results of the clinical survival analysis pipeline.")


# --- Main Content ---
if page == "Overview":
    st.title("📊 Dashboard Overview")
    st.markdown("This section provides a high-level summary of the pipeline results.")
    # Placeholder for overview content

elif page == "Model Performance":
    st.title("📈 Model Performance Details")
    st.markdown("Compare the performance of different models using various metrics.")
    # Placeholder for performance content

elif page == "Patient-Level Explanations":
    st.title("🧠 Patient-Level Explanations (SHAP)")
    st.markdown("Explore how different features contribute to predictions for individual patients.")
    # Placeholder for SHAP content

elif page == "Counterfactual Explorer":
    st.title("🔬 Counterfactual Explorer")
    st.markdown("Discover 'what-if' scenarios for high-risk patients.")

    # --- Model Selection ---
    selected_model = st.selectbox(
        "Select a model to explore its counterfactuals:",
        available_models,
        key="cf_model_select"
    )

    if selected_model:
        cf_data = load_counterfactuals(selected_model)

        if not cf_data:
            st.warning(f"No counterfactual data found for model: **{selected_model}**")
            st.info("Please ensure the pipeline was run with counterfactuals enabled.")
        else:
            st.header(f"Explanations for `{selected_model}`")

            # Extract test data and counterfactuals
            test_data_df = pd.DataFrame.from_dict(cf_data["test_data"])
            cf_list = cf_data["cfs_list"]

            if not cf_list:
                st.info("No counterfactuals were generated for the selected instances.")
                st.stop()
            
            # --- Instance Selection ---
            instance_indices = list(range(len(test_data_df)))
            selected_instance_idx = st.selectbox(
                "Select a high-risk patient instance to examine:",
                instance_indices,
                format_func=lambda x: f"Patient Instance #{x+1}",
                key="cf_instance_select"
            )
            
            st.subheader("Original Patient Data (High Risk)")
            original_instance = test_data_df.iloc[[selected_instance_idx]]
            st.dataframe(original_instance)

            st.subheader("Counterfactuals (Suggested Changes for Low Risk)")
            counterfactuals_for_instance = pd.DataFrame.from_dict(cf_list[selected_instance_idx])
            
            if counterfactuals_for_instance.empty:
                st.write("No counterfactuals found for this instance.")
            else:
                # Highlight the changes
                def highlight_diff(data, original):
                    attr = 'background-color: {}'.format('lightgreen')
                    other = original.iloc[0]
                    return pd.DataFrame(np.where(data.ne(other), attr, ''),
                                        index=data.index, columns=data.columns)

                st.dataframe(counterfactuals_for_instance.style.apply(
                    highlight_diff, original=original_instance, axis=None
                ))
