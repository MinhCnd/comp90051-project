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
