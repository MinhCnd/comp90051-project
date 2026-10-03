import numpy as np

def calculate_ece(predictions: np.ndarray, labels: np.ndarray, num_bins=10) -> float:
    """Calculate Expected Calibration Error for binary classification predictions.

    Reference: https://arxiv.org/html/2501.19047v2
    
    Args:
        predictions (np.ndarray): Array of shape (n, 2) with predicted probabilities and predicted labels.
        labels (np.ndarray): Array of true binary labels (0 or 1).
        num_bins (int, optional): Number of bins to use for calibration. Defaults to 10.

    Returns:
        float: Expected Calibration Error (ECE).
    """
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions must have shape (n, 2)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")
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


def calculate_brier_score(prediction_probs: np.ndarray, labels: np.ndarray) -> float:
    if prediction_probs.ndim != 1:
        raise ValueError("prediction_probs must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != prediction_probs.shape[0]:
        raise ValueError("labels must have shape (n,)")

    labels = labels.astype(float)
    return np.mean((prediction_probs - labels)**2)

    pass

def calculate_recall(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    TP = np.sum(predictions[(predictions == 1) & (labels == 1)])
    FN = np.sum(predictions[(predictions == 0) & (labels == 1)])


    return 0.0 if (TP + FN) == 0 else TP / (TP + FN)

def calculate_auc(predictions: np.ndarray, labels: np.ndarray) -> float:


    pass

def calculate_f1_score(predictions: np.ndarray, labels: np.ndarray) -> float:

    return 2 * (calculate_precision(predictions, labels) * calculate_recall(predictions, labels)) / (calculate_precision(predictions, labels) + calculate_recall(predictions, labels))

def calculate_precision(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    TP = np.sum(predictions[(predictions == 1) & (labels == 1)])
    FP = np.sum(predictions[(predictions == 1) & (labels == 0)])


    return 0.0 if (TP + FP) == 0 else TP / (TP + FP)

def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")
    if len(predictions) == 0:
        raise ValueError("predictions must not be empty")

    TP = np.sum(predictions[(predictions == 1) & (labels == 1)])
    TN = np.sum(predictions[(predictions == 0) & (labels == 0)])

    return (TP + TN) / len(predictions)

def calculate_error_rate(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")
    if len(predictions) == 0:
        raise ValueError("predictions must not be empty")

    FP = np.sum(predictions[(predictions == 1) & (labels == 0)])
    FN = np.sum(predictions[(predictions == 0) & (labels == 1)])

    return (FN + FP) / len(predictions)

def calculate_specificity(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    TN = np.sum(predictions[(predictions == 0) & (labels == 0)])
    FP = np.sum(predictions[(predictions == 1) & (labels == 0)])


    return 0.0 if (TN + FP) == 0 else TN / (TN + FP)


