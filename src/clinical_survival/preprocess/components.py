from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Component Registry ---
# Maps string names from the config to actual sklearn transformer classes.
# This makes the system extensible. To add a new component, just add it here.
COMPONENT_REGISTRY = {
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
    "OneHotEncoder": OneHotEncoder,
    # Add other transformers here as needed, e.g.:
    # "MinMaxScaler": MinMaxScaler,
    # "OrdinalEncoder": OrdinalEncoder,
}


def get_component(name: str):
    """
    Retrieves a component class from the registry by its string name.
    """
    if name not in COMPONENT_REGISTRY:
        raise ValueError(
            f"Unknown component '{name}'. "
            f"Available components: {list(COMPONENT_REGISTRY.keys())}"
        )
    return COMPONENT_REGISTRY[name]









