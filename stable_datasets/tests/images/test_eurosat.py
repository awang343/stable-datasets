import numpy as np
from PIL import Image

from stable_datasets.images import EuroSAT


def test_eurosat_dataset():
    # EuroSAT ships as one archive with no official split; the builder
    # exposes it as a single "train" split (same pattern as AWA2).
    eurosat = EuroSAT(split="train")

    # Test 1: Check the dataset size matches the known total (27,000 images).
    assert len(eurosat) == 27000, f"Expected 27000 images, got {len(eurosat)}."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = eurosat[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"EuroSAT images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"EuroSAT images should have 3 channels (RGB), got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range (10 classes)
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9 (10 classes), got {label}."

    # Test 5: Validate that all 10 classes are represented.
    labels_set = {eurosat[i]["label"] for i in range(0, len(eurosat), 97)}
    assert len(labels_set) == 10, f"Expected all 10 classes to be represented, found {len(labels_set)}."
