"""Run the model on extracted features and format prediction results."""

import numpy as np

from src.audio_processing import extract_mfcc_features


def predict_bird_species(audio_file, model, labels, top_k: int = 5):
    """Predict the bird species for an audio file.

    Args:
        audio_file: path to an audio file.
        model: loaded Keras model.
        labels: dict mapping stringified class index -> species name.
        top_k: how many ranked predictions to return.

    Returns:
        dict with keys:
            species     -> top predicted species name
            confidence  -> top confidence as a percentage (0-100)
            top_k       -> list of {Species, Confidence, Score} dicts
            audio       -> raw audio samples
            sample_rate -> audio sample rate
            mfccs       -> full MFCC matrix (for the heatmap)

    Raises:
        AudioValidationError: propagated from feature extraction.
    """
    features, audio, sample_rate, mfccs = extract_mfcc_features(audio_file)

    prediction = model.predict(features, verbose=0)
    probs = prediction[0]

    top_index = int(np.argmax(probs))
    species = labels[str(top_index)]
    confidence = float(np.max(probs) * 100)

    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_predictions = [
        {
            "Species": labels[str(int(idx))],
            "Confidence": f"{probs[idx] * 100:.2f}%",
            "Score": float(probs[idx] * 100),
        }
        for idx in top_indices
    ]

    return {
        "species": species,
        "confidence": confidence,
        "top_k": top_predictions,
        "audio": audio,
        "sample_rate": sample_rate,
        "mfccs": mfccs,
    }
