import os
import pytest
import numpy as np
from mlxtend.data.boston_housing import boston_housing_data

    
def test_boston_housing_data_valid(tmp_path, monkeypatch):
    """Test that boston_housing_data returns correct arrays when given a valid CSV file."""
    # Create a temporary CSV file with 2 rows and 14 columns (13 features and 1 target)
    row1 = ",".join(str(x) for x in range(14))
    row2 = ",".join(str(x) for x in range(14, 28))
    content = row1 + "\n" + row2
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)

    # Monkeypatch the module's DATA_PATH to point to the temporary CSV file
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))

    # Call the function and check the returned arrays
    X, y = boston_housing_data()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (2, 13)
    assert y.shape == (2,)

    # Validate some values: first row's features and target value
    expected_row1 = [float(x) for x in range(13)]
    assert np.allclose(X[0], expected_row1)
    assert y[0] == 13.0

def test_boston_housing_data_file_not_found(monkeypatch):
    """Test that boston_housing_data raises an error if the CSV file is missing."""
    # Set the DATA_PATH to a non-existent file
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", "non_existent_file.csv")
    with pytest.raises(Exception):
            boston_housing_data()
def test_boston_housing_data_single_row(tmp_path, monkeypatch):
    """Test that function raises an error when the CSV file has a single row (resulting in a 1D array)."""
    single_row = ",".join(str(x) for x in range(14))
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(single_row)
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    with pytest.raises(IndexError):
        boston_housing_data()

def test_boston_housing_data_trailing_newline(tmp_path, monkeypatch):
    """Test that the CSV file with a trailing empty newline is read correctly."""
    row1 = ",".join(str(x) for x in range(14))
    row2 = ",".join(str(x) for x in range(14, 28))
    content = row1 + "\n" + row2 + "\n"
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    X, y = boston_housing_data()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (2, 13)
    assert y.shape == (2,)

def test_boston_housing_data_invalid_numeric(tmp_path, monkeypatch):
    """Test that non-numeric data in the CSV file results in NaN in the returned arrays."""
    # Construct row1 with "non_numeric" at column index 12 (feature column) and row2 with valid floats.
    row1 = ",".join(str(x) if x != 12 else "non_numeric" for x in range(14))
    row2 = ",".join(str(x) for x in range(14, 28))
    content = row1 + "\n" + row2
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    X, y = boston_housing_data()
    # Verify that the non-numeric field is converted to NaN
    assert np.isnan(X[0, 12]), "Expected a NaN value where conversion failed"
def test_boston_housing_data_empty_file(tmp_path, monkeypatch):
    """Test that an empty CSV file causes an IndexError during slicing due to no data."""
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text("")
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    with pytest.raises(IndexError):
        boston_housing_data()

def test_boston_housing_data_extra_whitespaces(tmp_path, monkeypatch):
    """Test that the CSV file with extra whitespaces surrounding the numbers is read correctly."""
    row1 = " , ".join(str(x) for x in range(14))
    row2 = " , ".join(str(x) for x in range(14, 28))
    content = row1 + "\n" + row2
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    X, y = boston_housing_data()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (2, 13)
    assert y.shape == (2,)

def test_boston_housing_data_inconsistent_columns(tmp_path, monkeypatch):
    """Test that a CSV file with rows having inconsistent number of columns raises an error."""
    # Row1 has 14 columns, row2 has 13 columns
    row1 = ",".join(str(x) for x in range(14))
    row2 = ",".join(str(x) for x in range(13, 26))  # has 13 numbers
    content = row1 + "\n" + row2
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    with pytest.raises(Exception):
        boston_housing_data()
def test_boston_housing_data_with_header(tmp_path, monkeypatch):
    """Test that a CSV file containing a header row returns arrays where the header row converts to NaN."""
    # Create a CSV file with a header row followed by two valid data rows (14 columns each)
    header = ",".join("Header"+str(i) for i in range(14))
    row1 = ",".join(str(x) for x in range(14))
    row2 = ",".join(str(x) for x in range(14, 28))
    content = header + "\n" + row1 + "\n" + row2
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)

    # Redirect the function to use the temporary CSV file
    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))

    X, y = boston_housing_data()

    # Check that the header row was converted to NaN values
    assert np.all(np.isnan(X[0])), "Header row conversion should result in NaNs for features"
    assert np.isnan(y[0]), "Header row conversion should result in NaN for target"

    # Validate the remaining rows are parsed correctly
    expected_row1 = [float(x) for x in range(13)]
    assert np.allclose(X[1], expected_row1)
    assert y[1] == 13.0

def test_boston_housing_data_exponential_format(tmp_path, monkeypatch):
    """Test that numbers in exponential notation are correctly parsed."""
    # Create a CSV file with 2 rows using exponential format
    fmt_exp = lambda x: f"{float(x):.1e}"
    row1 = ",".join(fmt_exp(x) for x in range(14))
    row2 = ",".join(fmt_exp(x) for x in range(14, 28))
    content = row1 + "\n" + row2
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)

    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    X, y = boston_housing_data()

    # Validate the first row's features and target using expected float values
    expected_row1 = [float(x) for x in range(13)]
    assert np.allclose(X[0], expected_row1)
    # y[0] is taken from the last column of row1; we compare it to the float conversion of that value.
    expected_y0 = float(row1.split(",")[-1])
    assert y[0] == expected_y0

def test_boston_housing_data_extra_columns(tmp_path, monkeypatch):
    """Test that a CSV file with extra columns (e.g., 15 instead of 14) is parsed consistently using slicing."""
    # Create a CSV file with 2 rows and 15 columns (14 features and 1 target by slicing)
    row1 = ",".join(str(x) for x in range(15))
    row2 = ",".join(str(x) for x in range(15, 30))
    content = row1 + "\n" + row2
    tmp_csv = tmp_path / "boston_housing.csv"
    tmp_csv.write_text(content)

    monkeypatch.setattr("mlxtend.data.boston_housing.DATA_PATH", str(tmp_csv))
    X, y = boston_housing_data()

    # For each row, X is all but the last column. Therefore, X.shape should be (2, 14) and y.shape should be (2,)
    assert X.shape == (2, 14)
    assert y.shape == (2,)

    # Validate the target values extracted from the last column of each row
    assert y[0] == 14.0
    assert y[1] == 29.0