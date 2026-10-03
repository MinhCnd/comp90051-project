from src.metrics import calculate_ece
import numpy as np
import pytest

def test_ece_perfectly_calibrated():
    # Confidence 1.0, all predictions correct -> ECE 0.
    predictions = np.array([[1.0, 1], [1.0, 0], [1.0, 1]])
    labels = np.array([1, 0, 1])
    assert calculate_ece(predictions, labels) == pytest.approx(0.0)

def test_ece_overconfident():
    # Confidence 1.0 but all wrong -> ECE 1.
    predictions = np.array([[1.0, 1], [1.0, 1]])
    labels = np.array([0, 0])
    assert calculate_ece(predictions, labels) == pytest.approx(1.0)

def test_ece_single_bin_mixed():
    # Confidence 0.75, accuracy 0.5 -> ECE 0.25.
    predictions = np.array([[0.75, 1], [0.75, 1], [0.75, 1], [0.75, 1]])
    labels = np.array([1, 1, 0, 0])
    assert calculate_ece(predictions, labels) == pytest.approx(0.25)

def test_ece_multiple_bins():
    # Bin (0.1, 0.2]: conf 0.15, acc 1 -> 0.85 * 0.5
    # Bin (0.9, 1.0]: conf 0.95, acc 1 -> 0.05 * 0.5
    predictions = np.array([[0.15, 0], [0.95, 1]])
    labels = np.array([0, 1])
    assert calculate_ece(predictions, labels) == pytest.approx(0.45)

def test_ece_zero_probability_excluded():
    # Probability 0.0 is outside every bin (lower bound exclusive).
    predictions = np.array([[0.0, 0], [1.0, 1]])
    labels = np.array([0, 1])
    assert calculate_ece(predictions, labels) == pytest.approx(0.0)

def test_ece_num_bins():
    predictions = np.array([[0.6, 1], [0.6, 0]])
    labels = np.array([1, 1])
    # Conf 0.6, acc 0.5 -> 0.1 regardless of bin count.
    assert calculate_ece(predictions, labels, num_bins=5) == pytest.approx(0.1)
    assert calculate_ece(predictions, labels, num_bins=20) == pytest.approx(0.1)

def test_ece_returns_float_in_range():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.01, 1.0, 100)
    predictions = np.column_stack([probs, (probs > 0.5).astype(int)])
    labels = rng.integers(0, 2, 100)
    ece = calculate_ece(predictions, labels)
    assert ece >= 0.0
    assert ece <= 1.0

def test_ece_custom():
    predictions = np.array([[0.78, 1],[0.64, 1],[0.92, 0],[0.58, 0],[0.51, 1],[0.85,0],[0.7, 1],[0.63, 0],[0.83, 1]])
    labels = np.array([1, 1, 1 ,0, 0, 0, 1, 1, 1])
    assert calculate_ece(predictions, labels, 5) == pytest.approx(0.10445)  # Example expected value

