import numpy as np


def get_ece(predictions: np.ndarray, labels: np.ndarray, num_bins=10) -> float:
    """Calculate Expected Calibration Error for binary classification predictions.

    Reference: https://arxiv.org/html/2501.19047v2
    
    Args:
        predictions (np.ndarray): Array of shape (n, 2) with max predicted probabilities and predicted labels.
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


def get_roc_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Area under the ROC curve (AUC) for binary classification predictions.
    
    Args:
        predictions (np.ndarray): Array of shape (n, 2) with predicted probabilities(for the positive class) and predicted labels.
        labels (np.ndarray): Array of true binary labels (0 or 1).

    Returns:
        float: Area under the ROC curve (AUC).
    """
    y_prob = predictions[:, 0]
    y_preds = predictions[:, 1]

    # Sort samples by predicted probabilities in descending order
    desc_indx = np.argsort(-y_prob)
    y_prob_sorted = y_prob[desc_indx]
    labels_sorted = labels[desc_indx]

    # Identify indices corresponding to distinct prob threshold
    distinct_value_indices = np.where(np.diff(y_prob_sorted) != 0)[0]
    threshold_indxs = np.r_[distinct_value_indices, y_prob_sorted.size - 1]

    # Calculate cumulative TP and FP at unique thresholds
    tps = np.cumsum(labels_sorted == 1)[threshold_indxs]
    fps = np.cumsum(labels_sorted == 0)[threshold_indxs]

    # Prepend origin
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # Normalize to get TPR and FPR
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Calculate AUC using the trapezoidal rule
    auc = float(np.trapezoid(tpr, fpr))
    return auc

def get_f1_precision_recall(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Calculate F1 score, precision, and recall for the predictions.

    Args:
        predictions (np.ndarray): Array of predicted labels (0 or 1).
        labels (np.ndarray): Array of true labels (0 or 1).

    Returns:
        tuple[float, float, float]: F1 score, precision, and recall.
    """
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
    """Calculate the precision of the predictions.

    Args:
        predictions (np.ndarray): Array of predicted labels (0 or 1).
        labels (np.ndarray): Array of true labels (0 or 1).

    Returns:
        float: Precision score.
    """
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))


    return 0.0 if (tp + fp) == 0 else tp / (tp + fp)

def calculate_f1_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the F1 score of the predictions.

    Args:
        predictions (np.ndarray): Array of predicted labels (0 or 1).
        labels (np.ndarray): Array of true labels (0 or 1).

    Returns:
        float: F1 score.
    """
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
    """Calculate the accuracy of the predictions.

    Args:
        predictions (np.ndarray): Array of predicted labels (0 or 1).
        labels (np.ndarray): Array of true labels (0 or 1).

    Returns:
        float: Accuracy score.
    """
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
    """Calculate the error rate of the predictions.

    Args:
        predictions (np.ndarray): Array of predicted labels (0 or 1).
        labels (np.ndarray): Array of true labels (0 or 1).

    Returns:
        float: Error rate.
    """
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")
    if len(predictions) == 0:
        raise ValueError("predictions must not be empty")

    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    return (fn + fp) / len(predictions)

def best_f1_threshold(probs, labels, grid=np.arange(0.10, 0.91, 0.01)):
    """Return (threshold, f1) maximising F1 over the grid."""
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        f1 = calculate_f1_score(preds, labels)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1

def calculate_specificity(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the specificity of the predictions.

    Args:
        predictions (np.ndarray): Array of predicted labels (0 or 1).
        labels (np.ndarray): Array of true labels (0 or 1).

    Returns:
        float: Specificity score.
    """
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape (n, 1)")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
        raise ValueError("labels must have shape (n,)")

    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))


    return 0.0 if (tn + fp) == 0 else tn / (tn + fp)


def get_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """ Calculate a set of eval metrics for the given predictions and labels.

    Args:
        predictions (np.ndarray): Array of shape (n, 3) where the first column is the probability of class 0, the second column is the probability of class 1, and the third column is the predicted label (0 or 1).
        labels (np.ndarray): Array of true labels (0 or 1).

    Returns:
        dict: Dictionary containing accuracy, error rate, specificity, precision, recall, F1 score, ECE and ROC AUC.
    """

    p0 = predictions[:, 0]
    p1 = predictions[:, 1]
    pred_labels = predictions[:, 2]
    max_probs = np.maximum(p0, p1)
    max_probs = np.concatenate([max_probs[:, None], pred_labels[:, None]], axis=1)
    p1_labels = np.concatenate([p1[:, None], pred_labels[:, None]], axis=1)
    auc = get_roc_auc(p1_labels, labels)
    f1, precision, recall = get_f1_precision_recall(pred_labels, labels)
    ece = get_ece(max_probs, labels)
    return {
        "accuracy": calculate_accuracy(pred_labels, labels),
        "error_rate": calculate_error_rate(pred_labels, labels),
        "specificity": calculate_specificity(pred_labels, labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ece": ece,
        "auc": auc
    }