import os
import struct
import tempfile
import numpy as np
import pytest

from mlxtend.data.local_mnist import loadlocal_mnist

def create_labels_file(file_path, n_labels, labels_data):
    # Write header: magic number and count, then label data
    with open(file_path, "wb") as f:
        f.write(struct.pack(">II", 2049, n_labels))
        f.write(bytearray(labels_data))

def create_images_file(file_path, n_images, rows, cols, images_data):
    # Write header: magic number, number of images, rows, cols, then image data
    with open(file_path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n_images, rows, cols))
        f.write(bytearray(images_data))

def test_loadlocal_mnist_success():
    """Test that loadlocal_mnist correctly loads valid MNIST data."""
    n_labels = 2
    rows, cols = 28, 28
    labels = [7, 3]
    # Create two dummy images: first image with all zeros and second image with all ones
    image1 = [0] * (rows * cols)
    image2 = [1] * (rows * cols)
    images_data = image1 + image2

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        create_labels_file(labels_path, n_labels, labels)
        create_images_file(images_path, n_labels, rows, cols, images_data)

        images_loaded, labels_loaded = loadlocal_mnist(images_path, labels_path)
        assert images_loaded.shape == (n_labels, rows * cols)
        assert labels_loaded.shape == (n_labels,)
        np.testing.assert_array_equal(images_loaded[0], np.array(image1, dtype=np.uint8))
        np.testing.assert_array_equal(images_loaded[1], np.array(image2, dtype=np.uint8))
        np.testing.assert_array_equal(labels_loaded, np.array(labels, dtype=np.uint8))

def test_loadlocal_mnist_invalid_images():
    """Test that loadlocal_mnist raises an error when image file data is incomplete."""
    n_labels = 2
    rows, cols = 28, 28
    labels = [0, 1]
    # Provide image data that is insufficient (only one image worth)
    image_data = [0] * (rows * cols)

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        create_labels_file(labels_path, n_labels, labels)
        create_images_file(images_path, 1, rows, cols, image_data)  # Intentionally wrong number of images

        with pytest.raises(ValueError):
            loadlocal_mnist(images_path, labels_path)

def test_loadlocal_mnist_nonexistent_file():
    """Test that loadlocal_mnist raises an error when provided file paths do not exist."""
    with pytest.raises(FileNotFoundError):
        loadlocal_mnist("nonexistent_images_file", "nonexistent_labels_file")

def test_images_data_type_uint8():
    """Test that loaded images are of type uint8."""
    n_labels = 1
    rows, cols = 28, 28
    labels = [5]
    image_data = [123] * (rows * cols)

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        create_labels_file(labels_path, n_labels, labels)
        create_images_file(images_path, n_labels, rows, cols, image_data)

        images_loaded, _ = loadlocal_mnist(images_path, labels_path)
        assert images_loaded.dtype == np.uint8
def test_loadlocal_mnist_inconsistent_labels():
    """Test that loadlocal_mnist handles an inconsistent labels file where the header count
    does not match the actual number of labels written. The function uses the actual number of labels
    read from the file to reshape the image file."""
    n_labels_header = 2
    actual_labels = [9]  # only one label provided, even though header says 2
    rows, cols = 28, 28
    # Provide exactly 1 image because the actual labels read will be 1.
    image_data = [200] * (rows * cols)

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        # Write header that claims 2 labels, but only write 1 label.
        with open(labels_path, "wb") as f:
            f.write(struct.pack(">II", 2049, n_labels_header))
            f.write(bytearray(actual_labels))

        # Now create images file with exactly 1 image.
        create_images_file(images_path, len(actual_labels), rows, cols, image_data)

        images_loaded, labels_loaded = loadlocal_mnist(images_path, labels_path)
        assert images_loaded.shape == (1, rows * cols)
        np.testing.assert_array_equal(labels_loaded, np.array(actual_labels, dtype=np.uint8))

def test_loadlocal_mnist_empty_labels_file():
    """Test that loadlocal_mnist raises an error when the labels file is empty (i.e. the header cannot be read)."""
    rows, cols = 28, 28
    image_data = [100] * (rows * cols)

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        # Create an empty labels file.
        open(labels_path, "wb").close()

        create_images_file(images_path, 1, rows, cols, image_data)

        with pytest.raises(struct.error):
            loadlocal_mnist(images_path, labels_path)

def test_loadlocal_mnist_empty_images_file():
    """Test that loadlocal_mnist raises an error when the images file is empty (i.e., header cannot be read)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        create_labels_file(labels_path, 1, [2])
        # Create an empty images file.
        open(images_path, "wb").close()

        with pytest.raises(struct.error):
            loadlocal_mnist(images_path, labels_path)
def test_loadlocal_mnist_extra_labels():
    """Test that loadlocal_mnist correctly loads MNIST data when the labels file contains extra labels than indicated by its header.
    The function should load as many images as there are actual labels read, even if the header's count is lower.
    """
    rows, cols = 28, 28
    header_count = 1
    extra_labels = [4, 5]  # two labels are actually written even though header indicates one
    # Create two dummy images using distinct pixel values.
    image1 = [50] * (rows * cols)
    image2 = [100] * (rows * cols)
    images_data = image1 + image2

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        # Write the labels file with the header count set to header_count but write extra_labels (length 2)
        with open(labels_path, "wb") as f:
            f.write(struct.pack(">II", 2049, header_count))
            f.write(bytearray(extra_labels))

        # Create the images file with 2 images (matching the actual labels written)
        create_images_file(images_path, len(extra_labels), rows, cols, images_data)

        images_loaded, labels_loaded = loadlocal_mnist(images_path, labels_path)
        assert images_loaded.shape == (len(extra_labels), rows * cols)
        np.testing.assert_array_equal(labels_loaded, np.array(extra_labels, dtype=np.uint8))

def test_loadlocal_mnist_extra_images():
    """Test that loadlocal_mnist raises a ValueError when the images file contains more image data than expected.
    In this case, the labels file indicates a single label, but the images file contains two images worth of data;
    attempting to reshape the extra bytes into (1, 784) will fail.
    """
    n_labels = 1
    rows, cols = 28, 28
    labels = [8]
    # Create two dummy images even though we have only one label.
    image1 = [77] * (rows * cols)
    image2 = [88] * (rows * cols)
    images_data = image1 + image2

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")

        create_labels_file(labels_path, n_labels, labels)
        # Create an images file with header indicating 2 images (rather than 1) and supply data for 2 images.
        create_images_file(images_path, 2, rows, cols, images_data)

        with pytest.raises(ValueError):
            loadlocal_mnist(images_path, labels_path)
def test_loadlocal_mnist_corrupted_labels_header():
    """Test that loadlocal_mnist raises an error when the labels file header is corrupted (i.e., incomplete header bytes)."""
    rows, cols = 28, 28
    # Create a valid images file with one image of constant value.
    image_data = [150] * (rows * cols)
    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "corrupted_labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "images.idx3-ubyte")
        # Write corrupted labels file with only 4 bytes (should be 8 bytes)
        with open(labels_path, "wb") as f:
            f.write(b'\x00\x00\x08\x01')
        create_images_file(images_path, 1, rows, cols, image_data)
        with pytest.raises(struct.error):
            loadlocal_mnist(images_path, labels_path)

def test_loadlocal_mnist_corrupted_images_header():
    """Test that loadlocal_mnist raises an error when the images file header is corrupted (i.e., incomplete header bytes)."""
    n_labels = 1
    rows, cols = 28, 28
    labels = [3]
    with tempfile.TemporaryDirectory() as tmpdir:
        labels_path = os.path.join(tmpdir, "labels.idx1-ubyte")
        images_path = os.path.join(tmpdir, "corrupted_images.idx3-ubyte")
        create_labels_file(labels_path, n_labels, labels)
        # Write a corrupted images file with only 10 bytes (less than the expected 16 bytes for header)
        with open(images_path, "wb") as f:
            f.write(b'\x00' * 10)
        with pytest.raises(struct.error):
            loadlocal_mnist(images_path, labels_path)