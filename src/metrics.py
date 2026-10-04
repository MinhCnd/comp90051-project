import numpy as np


def get_ece(predictions: np.ndarray, labels: np.ndarray, num_bins=10) -> float:
    """Calculate Expected Calibration Error for binary classification predictions.

    Reference: https://arxiv.org/html/2501.19047v2
    
    Args:
        predictions (np.ndarray): Array of shape (n, 2) with predicted probabilities and predicted labels.
        labels (np.ndarray): Array of true binary labels (0 or 1).
        num_bins (int, optional): Number of bins to use for calibration. Defaults to 10.

    Returns:
        float: Expected Calibration Error (ECE).
    """
    pred_probs = predictions[:, 0]
    pred_labels = predictions[:, 1]
    
    bin_size = 1.0 / num_bins
    ece = 0.0
    
    for i in range(num_bins):
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size

        # Get the mask for predictions that fall into the current bin.
        mask = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
        pred_in_bin_count = np.sum(mask)
        
        if pred_in_bin_count > 0:
            bin_accuracy = np.sum(pred_labels[mask] == labels[mask]) / pred_in_bin_count
            bin_confidence = np.mean(pred_probs[mask])
            ece += abs(bin_accuracy - bin_confidence) * pred_in_bin_count / len(predictions)
    
    return ece


def get_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Area under the ROC curve (AUC) for binary classification predictions.
    
    Args:
        predictions (np.ndarray): Array of shape (n, 2) with predicted probabilities and predicted labels.
        labels (np.ndarray): Array of true binary labels (0 or 1).

    Returns:
        float: Area under the ROC curve (AUC).
    """
    pred_probs = predictions[:, 0]
    pred_labels = predictions[:, 1]

    # Sort by predicted probabilities in descending order
    desc_order = np.argsort(-pred_probs)
    pred_probs = pred_probs[desc_order]
    pred_labels = pred_labels[desc_order]
    labels = labels[desc_order]

    # Calculate true positive and false positive rates
    tpr = np.cumsum(pred_labels == labels) / np.sum(labels == 1)
    fpr = np.cumsum(pred_labels != labels) / np.sum(labels == 0)

    # Append (0,0) at the beginning
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Calculate AUC using the trapezoidal rule
    auc = np.trapezoid(tpr, fpr)
    return auc

def get_f1_precision_recall(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    f1 = calculate_f1_score(predictions, labels)
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)

    return f1, precision, recall

def calculate_brier_score(prediction_probs: np.ndarray, labels: np.ndarray) -> float:
    if prediction_probs.ndim != 1:
        raise ValueError("prediction_probs must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != prediction_probs.shape[0]:
        raise ValueError("labels must have shape (n,)")

    labels = labels.astype(float)
    return np.mean((prediction_probs - labels)**2)


def calculate_recall(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    tp = np.sum((predictions == 1) & (labels == 1))
    fn = np.sum((predictions == 0) & (labels == 1))


    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)


def calculate_precision(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))


    return 0.0 if (tp + fp) == 0 else tp / (tp + fp)

def calculate_f1_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")
    if len(predictions) == 0:
        raise ValueError("predictions must not be empty")

    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    return (tp + tn) / len(predictions)

def calculate_error_rate(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")
    if len(predictions) == 0:
        raise ValueError("predictions must not be empty")

    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    return (fn + fp) / len(predictions)

def calculate_specificity(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))


    return 0.0 if (tn + fp) == 0 else tn / (tn + fp)


