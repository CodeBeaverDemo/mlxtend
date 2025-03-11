import numpy as np
import pytest
import os
from mlxtend.data.autompg import autompg_data

def test_autompg_data_returns_correct_arrays(monkeypatch):
    """Test that autompg_data returns correct X and y arrays given valid input data."""
    # Create dummy data with shape (3, 6): 5 features + 1 target (3 samples, 6 columns)
    dummy_data = np.array([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]
    ])

    # Monkeypatch np.genfromtxt to return dummy_data regardless of file name or delimiter.
    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)

    X, y = autompg_data()

    # X should be all columns except the last one, y should be the last column.
    expected_X = dummy_data[:, :-1]
    expected_y = dummy_data[:, -1]
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)

def test_autompg_data_empty(monkeypatch):
    """Test that autompg_data returns empty arrays when input data is empty."""
    # Create an empty dummy data with 5 columns (4 features + 1 target).
    dummy_data = np.empty((0, 5))

    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)

    X, y = autompg_data()

    assert X.shape == (0, 4)  # since dummy_data has 5 columns and X excludes the last column
    assert y.shape == (0,)

def test_autompg_data_invalid_input(monkeypatch):
    """Test that autompg_data raises an error when data is invalid (e.g., 1-dimensional)."""
    # Return a 1D array instead of a 2D array, which will cause slicing to fail.
    dummy_data = np.array([1, 2, 3, 4, 5])

    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)

    with pytest.raises(IndexError):
        autompg_data()
def test_autompg_data_single_sample(monkeypatch):
    """Test that autompg_data correctly parses a dataset with a single sample."""
    # Create dummy data with a single sample (row) with 6 columns (5 features + 1 target)
    dummy_data = np.array([[10, 20, 30, 40, 50, 60]])
    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)

    X, y = autompg_data()
    expected_X = dummy_data[:, :-1]
    expected_y = dummy_data[:, -1]
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)

def test_autompg_data_one_column(monkeypatch):
    """Test that autompg_data returns correct shapes when the input data has only one column.
    In such a case, since X is taken as all columns except the last and the only column is the target,
    X will be an empty array of shape (n, 0) and y will have shape (n,)."""
    dummy_data = np.array([[100], [200], [300]])
    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)

    X, y = autompg_data()
    # X should have 0 columns since dummy_data has only one column
    assert X.shape == (3, 0)
    # y should be a 1-dimensional array of length 3
    assert y.shape == (3,)

def test_autompg_data_none(monkeypatch):
    """Test that autompg_data raises a TypeError when np.genfromtxt returns None (no data)."""
    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: None)
    with pytest.raises(TypeError):
        autompg_data()
def test_autompg_data_calls_genfromtxt(monkeypatch):
    """Test that autompg_data calls np.genfromtxt with the correct file path and delimiter."""
    calls = []

    def dummy_genfromtxt(fname, delimiter):
        calls.append((fname, delimiter))
        # Return dummy data with two samples:
        # 2 features (all columns except target) and 1 target column.
        return np.array([[1, 2, 3], [4, 5, 6]])

    monkeypatch.setattr(np, "genfromtxt", dummy_genfromtxt)

    X, y = autompg_data()

    assert calls, "np.genfromtxt was not called"
    fname, delim = calls[0]
    # Check that the file path ends with the expected subdirectory/filename.
    expected_ending = os.path.join("data", "autompg.csv.gz")
    assert fname.endswith(expected_ending), "The file path used is incorrect."
    assert delim == ",", "The delimiter used is not a comma."

    # Check that X and y are correctly parsed:
    expected_X = np.array([[1, 2], [4, 5]])
    expected_y = np.array([3, 6])
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)
def test_autompg_data_with_nans(monkeypatch):
    """Test that autompg_data returns arrays that correctly preserve np.nan values."""
    # Create dummy data with np.nan values.
    dummy_data = np.array([
        [1.0, np.nan, 3.0, 4.0],
        [5.0, 6.0, np.nan, 8.0]
    ])
    # Monkeypatch np.genfromtxt to return dummy_data
    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)
    
    X, y = autompg_data()
    expected_X = dummy_data[:, :-1]
    expected_y = dummy_data[:, -1]
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)

def test_autompg_data_list_input(monkeypatch):
    """Test that autompg_data raises a TypeError when np.genfromtxt returns a list instead of an ndarray."""
    # Return a normal Python list rather than a NumPy array.
    dummy_data = [[1, 2, 3], [4, 5, 6]]
    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)

    with pytest.raises(TypeError):
        autompg_data()

def test_autompg_data_non_numeric(monkeypatch):
    """Test that autompg_data correctly parses datasets containing non-numeric (string) values."""
    dummy_data = np.array([
        ["a", "b", "c"],
        ["d", "e", "f"]
    ])
    monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)

    X, y = autompg_data()
    expected_X = dummy_data[:, :-1]
    expected_y = dummy_data[:, -1]
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)