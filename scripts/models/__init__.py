from .supervised_lr import LogisticRegression
from .supervised_nonlinear_nn import BinaryClassificationModel, ClassificationModel

def load_probe(probe_type):
    if probe_type == "logistic_regression":
        return LogisticRegression
    elif probe_type == "nonlinear_nn":
        return BinaryClassificationModel
    elif probe_type == "multiclass_nn":
        return ClassificationModel
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")