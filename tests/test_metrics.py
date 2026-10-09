import numpy as np
import pytest

from src.metrics import (calculate_accuracy, calculate_brier_score,
                         calculate_precision, calculate_recall,
                         get_ece)


def test_ece_perfectly_calibrated():
    # Confidence 1.0, all predictions correct -> ECE 0.
    predictions = np.array([[1.0, 1], [1.0, 0], [1.0, 1]])
    labels = np.array([1, 0, 1])
    assert get_ece(predictions, labels) == pytest.approx(0.0)

def test_ece_overconfident():
    # Confidence 1.0 but all wrong -> ECE 1.
    predictions = np.array([[1.0, 1], [1.0, 1]])
    labels = np.array([0, 0])
    assert get_ece(predictions, labels) == pytest.approx(1.0)

def test_ece_single_bin_mixed():
    # Confidence 0.75, accuracy 0.5 -> ECE 0.25.
    predictions = np.array([[0.75, 1], [0.75, 1], [0.75, 1], [0.75, 1]])
    labels = np.array([1, 1, 0, 0])
    assert get_ece(predictions, labels) == pytest.approx(0.25)

def test_ece_multiple_bins():
    # Bin (0.1, 0.2]: conf 0.15, acc 1 -> 0.85 * 0.5
    # Bin (0.9, 1.0]: conf 0.95, acc 1 -> 0.05 * 0.5
    predictions = np.array([[0.15, 0], [0.95, 1]])
    labels = np.array([0, 1])
    assert get_ece(predictions, labels) == pytest.approx(0.45)

def test_ece_zero_probability_excluded():
    # Probability 0.0 is outside every bin (lower bound exclusive).
    predictions = np.array([[0.0, 0], [1.0, 1]])
    labels = np.array([0, 1])
    assert get_ece(predictions, labels) == pytest.approx(0.0)

def test_ece_num_bins():
    predictions = np.array([[0.6, 1], [0.6, 0]])
    labels = np.array([1, 1])
    # Conf 0.6, acc 0.5 -> 0.1 regardless of bin count.
    assert get_ece(predictions, labels, num_bins=5) == pytest.approx(0.1)
    assert get_ece(predictions, labels, num_bins=20) == pytest.approx(0.1)

def test_ece_returns_float_in_range():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.01, 1.0, 100)
    predictions = np.column_stack([probs, (probs > 0.5).astype(int)])
    labels = rng.integers(0, 2, 100)
    ece = get_ece(predictions, labels)
    assert ece >= 0.0
    assert ece <= 1.0

def test_ece_custom():
    predictions = np.array([[0.78, 1],[0.64, 1],[0.92, 0],[0.58, 0],[0.51, 1],[0.85,0],[0.7, 1],[0.63, 0],[0.83, 1]])
    labels = np.array([1, 1, 1 ,0, 0, 0, 1, 1, 1])
    assert get_ece(predictions, labels, 5) == pytest.approx(0.10445, rel=1e-3)

#Recall
def test_recall_perfect():
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 1, 0, 0])
    assert calculate_recall(preds, labels) == 1.0


def test_recall_partial():
    preds = np.array([1, 0, 0, 1])
    labels = np.array([1, 1, 1, 0])
    assert calculate_recall(preds, labels) == pytest.approx(1 / 3)


def test_recall_all_missed():
    preds = np.array([0, 0, 0])
    labels = np.array([1, 1, 0])
    assert calculate_recall(preds, labels) == 0.0


def test_recall_no_positive_labels():
    preds = np.array([1, 0, 1])
    labels = np.array([0, 0, 0])
    assert calculate_recall(preds, labels) == 0.0


def test_recall_invalid_predictions_shape():
    with pytest.raises(ValueError):
        calculate_recall(np.array([[1], [0]]), np.array([1, 0]))


def test_recall_invalid_labels_shape():
    with pytest.raises(ValueError):
        calculate_recall(np.array([1, 0]), np.array([1, 0, 1]))
    with pytest.raises(ValueError):
        calculate_recall(np.array([1, 0]), np.array([[1], [0]]))

def test_calculate_precision_perfect():
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 1, 0, 0])
    assert calculate_precision(preds, labels) == 1.0


def test_calculate_precision_mixed():
    preds = np.array([1, 1, 1, 0])
    labels = np.array([1, 0, 1, 1])
    assert calculate_precision(preds, labels) == pytest.approx(2 / 3)


def test_calculate_precision_no_positive_predictions():
    preds = np.array([0, 0, 0])
    labels = np.array([1, 0, 1])
    assert calculate_precision(preds, labels) == 0.0


def test_calculate_precision_all_false_positives():
    preds = np.array([1, 1])
    labels = np.array([0, 0])
    assert calculate_precision(preds, labels) == 0.0


def test_calculate_precision_invalid_predictions_ndim():
    with pytest.raises(ValueError, match="predictions"):
        calculate_precision(np.array([[1], [0]]), np.array([1, 0]))


def test_calculate_precision_invalid_labels_ndim():
    with pytest.raises(ValueError, match="labels"):
        calculate_precision(np.array([1, 0]), np.array([[1], [0]]))


def test_calculate_precision_length_mismatch():
    with pytest.raises(ValueError, match="labels"):
        calculate_precision(np.array([1, 0, 1]), np.array([1, 0]))

def test_calculate_accuracy_all_correct():
    preds = np.array([1, 0, 1, 0])
    labels = np.array([1, 0, 1, 0])
    assert calculate_accuracy(preds, labels) == 1.0


def test_calculate_accuracy_all_wrong():
    preds = np.array([1, 0, 1, 0])
    labels = np.array([0, 1, 0, 1])
    assert calculate_accuracy(preds, labels) == 0.0


def test_calculate_accuracy_partial():
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 0, 0, 1])
    assert calculate_accuracy(preds, labels) == 0.5


def test_calculate_accuracy_invalid_predictions_ndim():
    with pytest.raises(ValueError):
        calculate_accuracy(np.array([[1], [0]]), np.array([1, 0]))


def test_calculate_accuracy_invalid_labels_shape():
    with pytest.raises(ValueError):
        calculate_accuracy(np.array([1, 0]), np.array([1, 0, 1]))
    with pytest.raises(ValueError):
        calculate_accuracy(np.array([1, 0]), np.array([[1], [0]]))


def test_calculate_accuracy_empty():
    with pytest.raises(ValueError):
        calculate_accuracy(np.array([]), np.array([]))

def test_calculate_brier_score_perfect():
    probs = np.array([1.0, 0.0, 1.0])
    labels = np.array([1, 0, 1])
    assert calculate_brier_score(probs, labels) == pytest.approx(0.0)


def test_calculate_brier_score_worst():
    probs = np.array([0.0, 1.0])
    labels = np.array([1, 0])
    assert calculate_brier_score(probs, labels) == pytest.approx(1.0)


def test_calculate_brier_score_mixed():
    probs = np.array([0.5, 0.8, 0.2, 0.9])
    labels = np.array([1, 1, 0, 0])
    expected = (0.25 + 0.04 + 0.04 + 0.81) / 4
    assert calculate_brier_score(probs, labels) == pytest.approx(expected)


def test_calculate_brier_score_uninformative():
    probs = np.full(4, 0.5)
    labels = np.array([0, 1, 0, 1])
    assert calculate_brier_score(probs, labels) == pytest.approx(0.25)


def test_calculate_brier_score_invalid_probs_shape():
    with pytest.raises(ValueError, match="prediction_probs"):
        calculate_brier_score(np.array([[0.5], [0.4]]), np.array([1, 0]))


def test_calculate_brier_score_invalid_labels_shape():
    with pytest.raises(ValueError, match="labels"):
        calculate_brier_score(np.array([0.5, 0.4]), np.array([[1], [0]]))


def test_calculate_brier_score_length_mismatch():
    with pytest.raises(ValueError, match="labels"):
        calculate_brier_score(np.array([0.5, 0.4]), np.array([1, 0, 1]))