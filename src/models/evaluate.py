from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluate a classification model.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        
    }