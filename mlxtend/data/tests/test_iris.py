import os
import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mlxtend.data import iris_data

this_dir, this_filename = os.path.split(__file__)
this_dir = re.sub("^stset", "", this_dir[::-1])[::-1]
DATA_PATH = os.path.join(this_dir, "data", "iris.csv.gz")


def test_iris_data_uci():
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=",")
    original_uci_data_x, original_uci_data_y = tmp[:, :-1], tmp[:, -1]
    original_uci_data_y = original_uci_data_y.astype(int)
    iris_x, iris_y = iris_data()
    assert_array_equal(original_uci_data_x, iris_x)
    assert_array_equal(original_uci_data_y, iris_y)


def test_iris_data_r():
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=",")
    original_r_data_x, original_r_data_y = tmp[:, :-1], tmp[:, -1]
    original_r_data_y = original_r_data_y.astype(int)
    original_r_data_x[34] = [4.9, 3.1, 1.5, 0.2]
    original_r_data_x[37] = [4.9, 3.6, 1.4, 0.1]
    iris_x, iris_y = iris_data(version="corrected")
    assert_array_equal(original_r_data_x, iris_x)


def test_iris_invalid_choice():
    with pytest.raises(ValueError) as excinfo:
        iris_data(version="bla")
        assert excinfo.value.message == "version must be 'uci' or 'corrected'."

def test_iris_data_invalid_version_type():
    """Test that providing a non-string version value raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        iris_data(version=None)
    # Check that the error message contains the expected text
    assert "version must be 'uci' or 'corrected'" in str(excinfo.value)

def test_iris_data_dtype_and_shape():
    """Test that iris_data returns numpy arrays with expected dtypes and shapes for both versions."""
    for version in ["uci", "corrected"]:
        X, y = iris_data(version=version)
        # Check types: X should be a float array, y an integer array.
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        # Check expected shapes: there are 150 samples and 4 features
        assert X.shape == (150, 4)
        assert y.shape == (150,)
        # Check that X's dtype is float and y's dtype is a kind of integer.
        assert X.dtype in [np.float64, np.float32]
        assert np.issubdtype(y.dtype, np.integer)

def test_iris_data_file_not_found(monkeypatch):
    """Test that iris_data propagates file not found errors from np.genfromtxt."""
    def fake_genfromtxt(*args, **kwargs):
        raise IOError("File not found")
    # Patch np.genfromtxt so that it raises an IOError to simulate a missing file.
    monkeypatch.setattr(np, "genfromtxt", fake_genfromtxt)
    with pytest.raises(IOError) as excinfo:
        iris_data(version="uci")
    assert "File not found" in str(excinfo.value)
def test_iris_data_empty_version(monkeypatch):
    """Test that providing an empty string as version raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        iris_data(version="")
    # Check that the error message contains the expected text
    assert "version must be 'uci' or 'corrected'" in str(excinfo.value)

def test_iris_data_uppercase_version(monkeypatch):
    """Test that providing an uppercase version string (e.g., 'UCI') raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        iris_data(version="UCI")
    # Check that the error message contains the expected text
    assert "version must be 'uci' or 'corrected'" in str(excinfo.value)

def test_iris_data_incorrect_shape(monkeypatch):
    """Test that iris_data raises an IndexError when the data shape is insufficient for the 'corrected' version.
    This simulates a scenario where np.genfromtxt returns an array with too few rows.
    """
    def fake_genfromtxt(*args, **kwargs):
        # Simulate a small array with only 30 rows (instead of the expected 150) and 5 columns
        return np.zeros((30, 5))

    monkeypatch.setattr(np, "genfromtxt", fake_genfromtxt)
    with pytest.raises(IndexError):
        iris_data(version="corrected")

def test_iris_data_returns_distinct_arrays():
    """Test that iris_data returns distinct array objects on consecutive calls,
    so that modifications to one do not affect the other.
    """
    iris_x1, iris_y1 = iris_data()
    iris_x2, iris_y2 = iris_data()
    # Check that the returned arrays are not the same objects in memory
    assert iris_x1 is not iris_x2
    assert iris_y1 is not iris_y2
def test_iris_data_numeric_version():
    """Test that providing a non-string numeric version (e.g., 123) raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        iris_data(version=123)
    # Check that the error message contains the expected text
    assert "version must be 'uci' or 'corrected'" in str(excinfo.value)

def test_iris_data_whitespace_version():
    """Test that providing a version string with extra whitespace (e.g., ' uci ') raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        iris_data(version=" uci ")
    # Check that the error message contains the expected text
    assert "version must be 'uci' or 'corrected'" in str(excinfo.value)

def test_iris_data_empty_file(monkeypatch):
    """Test that iris_data raises an IndexError when np.genfromtxt returns an empty array (simulating an empty data file)."""
    monkeypatch.setattr(np, "genfromtxt", lambda *args, **kwargs: np.array([]))
    with pytest.raises(IndexError):
        iris_data(version="uci")