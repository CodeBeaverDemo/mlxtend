import numpy as np
import pytest
from mlxtend.data.mnist import mnist_data

class TestMNIST:
    """Tests for the mnist_data function using monkeypatch to simulate input data."""

    @staticmethod
    def dummy_genfromtxt(fname, delimiter):
        # Create dummy data: 10 samples with 784 features each and 1 label column.
        # The labels will be sequential numbers from 0 to 9.
        data = np.hstack([np.random.rand(10, 784), np.arange(10).reshape(10, 1)])
        return data

    @staticmethod
    def dummy_genfromtxt_invalid(fname, delimiter):
        # Return an invalid 1D array to simulate erroneous data format.
        return np.array([1, 2, 3])
    @staticmethod
    def dummy_genfromtxt_empty(fname, delimiter):
        # Return an empty array with shape (0, 785) to simulate a dataset with no samples.
        return np.empty((0, 785))

    @staticmethod
    def dummy_genfromtxt_invalid_label(fname, delimiter):
        # Create dummy data with non-numeric labels that cannot be converted to int.
        data = np.hstack([np.random.rand(5, 784), np.array(['a']*5).reshape(5, 1)])
        return data
    @staticmethod
    def dummy_genfromtxt_single(fname, delimiter):
        """Create dummy data with a single sample for testing."""
        # Create dummy data: 1 sample with 784 features and 1 label column.
        data = np.hstack([np.zeros((1, 784)), np.array([7]).reshape(1, 1)])
        return data
    def test_mnist_returns_correct_shapes_and_types(self, monkeypatch):
        """Test that mnist_data returns X and y with expected shapes and correct data types."""
        monkeypatch.setattr(np, "genfromtxt", self.dummy_genfromtxt)
        X, y = mnist_data()
        # Check that X has 10 samples and 784 features.
        assert X.shape == (10, 784)
        # Check that y has 10 samples.
        assert y.shape == (10,)
        # Check that y's values are of integer type.
        assert np.issubdtype(y.dtype, np.integer)

    def test_mnist_invalid_data_raises_error(self, monkeypatch):
        """Test that mnist_data raises an IndexError when the data returned has an unexpected shape."""
        monkeypatch.setattr(np, "genfromtxt", self.dummy_genfromtxt_invalid)
        with pytest.raises(IndexError):
            _ = mnist_data()
    
    def test_mnist_empty_data(self, monkeypatch):
        """Test that mnist_data returns empty arrays when no data is provided."""
        monkeypatch.setattr(np, "genfromtxt", self.dummy_genfromtxt_empty)
        X, y = mnist_data()
        # When empty data is provided, X should have shape (0, 784) and y should have shape (0,)
        assert X.shape == (0, 784)
        assert y.shape == (0,)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_mnist_invalid_labels_raises_error(self, monkeypatch):
        """Test that mnist_data raises a ValueError when labels cannot be converted to int."""
        monkeypatch.setattr(np, "genfromtxt", self.dummy_genfromtxt_invalid_label)
        with pytest.raises(ValueError):
            _ = mnist_data()
    def test_mnist_single_sample(self, monkeypatch):
        """Test that mnist_data correctly handles a single sample input."""
        monkeypatch.setattr(np, "genfromtxt", self.dummy_genfromtxt_single)
        X, y = mnist_data()
        # Check that X has shape (1, 784) and y has shape (1,)
        assert X.shape == (1, 784)
        assert y.shape == (1,)
        # Check that the label can be correctly converted and that its value is as expected (7 in this case)
        assert np.issubdtype(y.dtype, np.integer)
        assert y[0] == 7
    
    def test_mnist_labels_preservation(self, monkeypatch):
        """Test that mnist_data returns labels that are exactly preserved from the input after conversion."""
        # Create a fixed dummy dataset where X is filled with a constant and labels are 0..9.
        dummy_data = np.hstack([np.full((10, 784), 3.14), np.arange(10).reshape(10, 1)])
        monkeypatch.setattr(np, "genfromtxt", lambda fname, delimiter: dummy_data)
        _, y = mnist_data()
        expected_labels = np.arange(10)
        np.testing.assert_array_equal(y, expected_labels)
    def dummy_genfromtxt_float_labels(self, fname, delimiter):
        """Return dummy data with float labels that should be convertible to int."""
        data = np.hstack([np.random.rand(10, 784), np.arange(10, dtype=float).reshape(10, 1)])
        return data

    def test_mnist_float_labels(self, monkeypatch):
        """Test that mnist_data correctly converts float labels to int."""
        monkeypatch.setattr(np, "genfromtxt", self.dummy_genfromtxt_float_labels)
        X, y = mnist_data()
        # Check that X has 10 samples and 784 features.
        assert X.shape == (10, 784)
        # Check that y has 10 samples, is of integer type and has the expected values.
        assert y.shape == (10,)
        assert np.issubdtype(y.dtype, np.integer)
        np.testing.assert_array_equal(y, np.arange(10))

    def dummy_genfromtxt_unexpected_features(self, fname, delimiter):
        """Return dummy data with an unexpected number of feature columns (500 instead of 784)."""
        data = np.hstack([np.random.rand(10, 500), np.arange(10).reshape(10, 1)])
        return data

    def test_mnist_unexpected_feature_count(self, monkeypatch):
        """Test that mnist_data handles data with an unexpected number of features."""
        monkeypatch.setattr(np, "genfromtxt", self.dummy_genfromtxt_unexpected_features)
        X, y = mnist_data()
        # Check that X shape is (10, 500) as provided by dummy_genfromtxt_unexpected_features
        assert X.shape == (10, 500)
        # y should still have 10 samples.
        assert y.shape == (10,)
        assert np.issubdtype(y.dtype, np.integer)