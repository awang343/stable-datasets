import numpy as np
from PIL import Image

from stable_datasets.images import OxfordIIITPet


def test_oxford_iiit_pet_dataset():
    pets_train = OxfordIIITPet(split="train")

    # Test 1: Non-empty, roughly the known trainval size (3680 images).
    assert len(pets_train) > 0, "Expected non-empty training split."
    assert 3000 <= len(pets_train) <= 3680, f"Unexpected trainval split size: {len(pets_train)}."

    # Test 2: Check sample keys
    sample = pets_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Pet images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Pet images should have 3 channels (RGB), got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range (37 breeds)
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 37, f"Label should be in range [0, 36], got {label}."

    # Test 5: Test split, roughly the known size (3669 images).
    pets_test = OxfordIIITPet(split="test")
    assert 3000 <= len(pets_test) <= 3669, f"Unexpected test split size: {len(pets_test)}."
