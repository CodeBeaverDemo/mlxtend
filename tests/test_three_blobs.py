import numpy as np
import pytest
from mlxtend.data.three_blobs import three_blobs_data

def test_returned_shape_and_values():
    """Test that three_blobs_data returns correct shape and appropriate label values."""
    X, y = three_blobs_data()
    # Check if X is a 2D array with exactly two feature columns
    assert X.ndim == 2, "X should be a 2D array"
    assert X.shape[1] == 2, "X should have exactly two columns"
    # Check that y is a 1D array and the number of samples match
    assert y.ndim == 1, "y should be a 1D array"
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"
    # Check that the unique labels (if any) are within the set {0, 1, 2}
    unique_labels = set(np.unique(y))
    expected_labels = {0, 1, 2}
    assert unique_labels.issubset(expected_labels), "Labels must be a subset of {0, 1, 2}"

def fake_genfromtxt(fname, delimiter):
    """Fake np.genfromtxt to simulate a known dataset for testing."""
    return np.array([[1, 2, 0],
                        [3, 4, 1],
                        [5, 6, 2]])

def test_three_blobs_data_with_monkeypatch(monkeypatch):
    """Test three_blobs_data using monkeypatch to simulate file input."""
    import mlxtend.data.three_blobs as tb
    monkeypatch.setattr(tb.np, "genfromtxt", fake_genfromtxt)
    X, y = tb.three_blobs_data()
    np.testing.assert_array_equal(X, np.array([[1, 2],
                                                    [3, 4],
                                                    [5, 6]]))
    np.testing.assert_array_equal(y, np.array([0, 1, 2]))

def fake_genfromtxt_float(fname, delimiter):
    """Fake np.genfromtxt that returns float labels to test integer conversion."""
    return np.array([[7, 8, 0.0],
                        [9, 10, 1.0],
                        [11, 12, 2.0]])

def test_three_blobs_data_label_conversion(monkeypatch):
    """Test that three_blobs_data converts the label column to integer dtype even when provided as float."""
    import mlxtend.data.three_blobs as tb
    monkeypatch.setattr(tb.np, "genfromtxt", fake_genfromtxt_float)
    X, y = tb.three_blobs_data()
    # Ensure that the label array is converted to an integer type
    assert issubclass(y.dtype.type, np.integer), "Labels should be of integer type"
    np.testing.assert_array_equal(y, np.array([0, 1, 2]))
def test_data_path():
    """Test that the DATA_PATH variable points to a valid file location structure."""
    from mlxtend.data import three_blobs
    # Check that the file path contains a 'data' subdirectory and ends with the expected filename
    assert "data" in three_blobs.DATA_PATH, "DATA_PATH should include 'data' directory"
    assert three_blobs.DATA_PATH.endswith("three_blobs.csv.gz"), "DATA_PATH should end with 'three_blobs.csv.gz'"

def fake_genfromtxt_empty(fname, delimiter):
    """Fake np.genfromtxt that returns an empty array to simulate missing data."""
    return np.array([])

def test_three_blobs_data_empty(monkeypatch):
    """Test that three_blobs_data raises an error when no data is returned (empty array)."""
    import mlxtend.data.three_blobs as tb
    monkeypatch.setattr(tb.np, "genfromtxt", fake_genfromtxt_empty)
    with pytest.raises(IndexError):
        tb.three_blobs_data()

def fake_genfromtxt_1d(fname, delimiter):
    """Fake np.genfromtxt that returns a 1D array to simulate an unexpected data format."""
    return np.array([1, 2, 0])

def test_three_blobs_data_1d(monkeypatch):
    """Test that three_blobs_data raises an error when the data from genfromtxt is not 2D."""
    import mlxtend.data.three_blobs as tb
    monkeypatch.setattr(tb.np, "genfromtxt", fake_genfromtxt_1d)
    with pytest.raises(IndexError):
        tb.three_blobs_data()
def fake_genfromtxt_invalid(fname, delimiter):
    """Fake np.genfromtxt that returns non-numeric labels to simulate invalid label conversion."""
    return np.array([[13, 14, 'a'],
                        [15, 16, 'b'],
                        [17, 18, 'c']])

def test_three_blobs_data_invalid_dtype(monkeypatch):
    """Test that three_blobs_data raises a ValueError when labels cannot be converted to integers."""
    import mlxtend.data.three_blobs as tb
    monkeypatch.setattr(tb.np, "genfromtxt", fake_genfromtxt_invalid)
    with pytest.raises(ValueError):
        tb.three_blobs_data()

def fake_genfromtxt_one_sample(fname, delimiter):
    """Fake np.genfromtxt that returns a 1-sample dataset."""
    return np.array([[21, 22, 0]])

def test_three_blobs_data_one_sample(monkeypatch):
    """Test that three_blobs_data correctly processes a dataset with only one sample."""
    import mlxtend.data.three_blobs as tb
    monkeypatch.setattr(tb.np, "genfromtxt", fake_genfromtxt_one_sample)
    X, y = tb.three_blobs_data()
    # X should be 2D with one row and two columns and y should be 1D with a single element.
    assert X.ndim == 2 and X.shape == (1,2), "X should be a 2D array with one row and two columns"
    assert y.ndim == 1 and y.shape[0] == 1, "y should be a 1D array with one element"
    np.testing.assert_array_equal(X, np.array([[21, 22]]))
    np.testing.assert_array_equal(y, np.array([0]))