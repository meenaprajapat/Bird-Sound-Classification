"""Load the trained CNN model and the class-index -> species-name mapping."""

import json
from pathlib import Path

from tensorflow import keras

MODEL_PATH = Path("bird_model.h5")
LABELS_PATH = Path("prediction.json")


def load_model_and_labels(model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
    """Load the Keras model and prediction dictionary.

    Returns:
        (model, labels_dict) on success.

    Raises:
        FileNotFoundError: if the model or labels file is missing.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found in the project directory.")
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} not found in the project directory.")

    model = keras.models.load_model(str(model_path))
    with open(labels_path, "r") as f:
        labels = json.load(f)

    return model, labels
