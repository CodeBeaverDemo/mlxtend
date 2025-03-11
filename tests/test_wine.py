import os
import numpy as np
import pytest
from mlxtend.data.wine import wine_data, DATA_PATH

def test_wine_data_returns_correct_values(tmp_path, monkeypatch):
    """Test that wine_data returns the correct feature matrix and labels after monkeypatching DATA_PATH to a custom test CSV file."""
    # Create a temporary CSV file with sample data:
    # Each row has 13 feature columns and 1 label column
    data = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 0],
        [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 1],
        [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 2],
    ])
    temp_csv = tmp_path / "wine.csv"
    np.savetxt(temp_csv, data, delimiter=",")

    # Patch the DATA_PATH to point to the temporary CSV file
    monkeypatch.setattr("mlxtend.data.wine.DATA_PATH", str(temp_csv))

    X, y = wine_data()
    # Check that the feature matrix matches the expected shape and values
    np.testing.assert_array_equal(X, data[:, :-1])
    # Check that the label vector matches and is of integer type
    np.testing.assert_array_equal(y, data[:, -1].astype(int))

def test_wine_data_file_not_found(monkeypatch):
    """Test that wine_data raises an error when the CSV file is not found."""
    # Monkeypatch DATA_PATH to point to an invalid file path
    monkeypatch.setattr("mlxtend.data.wine.DATA_PATH", "non_existing_file.csv")
    with pytest.raises(OSError):
        wine_data()