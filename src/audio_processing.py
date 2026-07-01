"""Audio loading, validation, and MFCC feature extraction."""

import librosa
import numpy as np

N_MFCC = 40


class AudioValidationError(Exception):
    """Raised when an audio clip is unsuitable for classification."""


def extract_mfcc_features(audio_file, n_mfcc: int = N_MFCC):
    """Load an audio file and extract mean MFCC features for the model.

    The feature pipeline matches the training notebook exactly:
    load at native sample rate -> 40 MFCCs -> mean over time -> reshape to (1, 40, 1).

    Args:
        audio_file: path to an audio file readable by librosa.
        n_mfcc: number of MFCC coefficients to extract.

    Returns:
        (features, audio, sample_rate, mfccs) where `features` has shape (1, n_mfcc, 1).

    Raises:
        AudioValidationError: if the clip is empty, too short, or silent.
    """
    audio, sample_rate = librosa.load(audio_file)

    if audio is None or len(audio) == 0:
        raise AudioValidationError(
            "This file contains no readable audio. Please try a different recording."
        )

    duration = len(audio) / sample_rate
    if duration < 0.5:
        raise AudioValidationError(
            "This clip is very short (under 0.5s). Please upload a longer recording "
            "for a reliable prediction."
        )

    if np.max(np.abs(audio)) < 1e-4:
        raise AudioValidationError(
            "This recording is silent or nearly silent. Please upload an audible bird call."
        )

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)

    features = np.expand_dims(mfccs_mean, axis=0)   # batch dimension
    features = np.expand_dims(features, axis=2)     # channel dimension

    return features, audio, sample_rate, mfccs
