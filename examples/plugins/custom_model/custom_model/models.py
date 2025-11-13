from sklearn.svm import SVC


class CustomSVM(SVC):
    """
    A custom SVM model with pre-defined parameters for demonstration.
    """

    def __init__(self, **kwargs):
        # Set custom defaults, but allow them to be overridden
        defaults = {"kernel": "rbf", "C": 1.5, "gamma": "scale", "probability": True}
        defaults.update(kwargs)
        super().__init__(**defaults)
