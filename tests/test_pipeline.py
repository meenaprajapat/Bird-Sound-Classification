"""Unit tests for the KooKoo AI inference pipeline.

Run with:  pytest -q
"""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.audio_processing import AudioValidationError, extract_mfcc_features
from src.model_loader import load_model_and_labels
from src.predictor import predict_bird_species


@pytest.fixture(scope="module")
def model_and_labels():
    return load_model_and_labels()


def _write_tone(freq=440, seconds=2.0, sr=22050):
    """Create a temp WAV containing a simple sine tone; return its path."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * freq * t).astype("float32")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, sr)
    return tmp.name


def test_labels_has_114_classes(model_and_labels):
    _, labels = model_and_labels
    assert len(labels) == 114


def test_mfcc_feature_shape():
    path = _write_tone()
    try:
        features, audio, sr, mfccs = extract_mfcc_features(path)
        assert features.shape == (1, 40, 1)
        assert len(audio) > 0
        assert mfccs.shape[0] == 40
    finally:
        os.remove(path)


def test_silent_audio_is_rejected():
    sr = 22050
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, np.zeros(sr, dtype="float32"), sr)
    try:
        with pytest.raises(AudioValidationError):
            extract_mfcc_features(tmp.name)
    finally:
        os.remove(tmp.name)


def test_prediction_returns_valid_species(model_and_labels):
    model, labels = model_and_labels
    path = _write_tone()
    try:
        result = predict_bird_species(path, model, labels)
        assert result["species"] in labels.values()
        assert 0.0 <= result["confidence"] <= 100.0
        assert len(result["top_k"]) == 5
    finally:
        os.remove(path)
