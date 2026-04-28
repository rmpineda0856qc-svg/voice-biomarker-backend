"""Voice biomarker extraction using parselmouth (Praat Python bindings).

Extracts research-grade acoustic features from a sustained vowel recording:
- F0 (fundamental frequency / pitch) in Hz
- Jitter (local) in %
- Shimmer (local) in %
- HNR (Harmonics-to-Noise Ratio) in dB

These are the same measurements used in Praat, which is the gold standard
for voice quality research.
"""
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import noisereduce as nr
import io
import tempfile
import os
from typing import Dict, Tuple


def preprocess_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Load audio, resample to 16 kHz mono, trim silence, and reduce noise.

    Returns (audio_array, sample_rate).
    """
    # Write bytes to a temp file because librosa.load needs a path or file-like
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # Load and resample to 16 kHz mono
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)

    if len(y) < sr * 0.5:
        raise ValueError("Audio too short — need at least 0.5 seconds.")

    # Trim leading/trailing silence (top_db = 25 means trim segments 25 dB below peak)
    y, _ = librosa.effects.trim(y, top_db=25)

    # Noise reduction (stationary noise profile)
    try:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75)
    except Exception:
        # If noise reduction fails (e.g., audio too short), continue with original
        pass

    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.95

    return y, sr


def extract_biomarkers(audio_bytes: bytes) -> Dict[str, float]:
    """Extract F0, Jitter, Shimmer, and HNR from an audio file.

    Args:
        audio_bytes: Raw bytes of a WAV file.

    Returns:
        Dict with keys: f0_hz, jitter_pct, shimmer_pct, hnr_db
    """
    y, sr = preprocess_audio(audio_bytes)

    # Create a parselmouth Sound object from the numpy array
    sound = parselmouth.Sound(y, sampling_frequency=sr)

    # --- Pitch (F0) ---
    # Praat's "To Pitch" with standard time step and reasonable F0 range for adult voice
    pitch = call(sound, "To Pitch", 0.0, 75, 500)  # time step, min pitch, max pitch
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")  # 0, 0 = over entire sound
    if np.isnan(f0_mean) or f0_mean == 0:
        # Voiced frames not found — try with wider pitch range
        pitch = call(sound, "To Pitch", 0.0, 50, 600)
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        if np.isnan(f0_mean) or f0_mean == 0:
            raise ValueError(
                "No voiced sound detected. Please record a sustained vowel like /a/."
            )

    # --- Point Process (needed for jitter and shimmer) ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    # --- Jitter (local) ---
    # Get jitter (local): period_start, period_end, shortest_period, longest_period, max_factor
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # --- Shimmer (local) ---
    shimmer_local = call(
        [sound, point_process], "Get shimmer (local)",
        0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # --- HNR (Harmonics-to-Noise Ratio) ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_db = call(harmonicity, "Get mean", 0, 0)

    # Handle potential NaN results
    def safe_float(v, default=0.0):
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                return default
            return fv
        except (TypeError, ValueError):
            return default

    return {
        "f0_hz": round(safe_float(f0_mean), 2),
        "jitter_pct": round(safe_float(jitter_local) * 100, 3),  # Praat returns ratio, convert to %
        "shimmer_pct": round(safe_float(shimmer_local) * 100, 3),
        "hnr_db": round(safe_float(hnr_db), 2),
    }


def compute_deltas(current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    """Compute deviation of current biomarkers from the user's baseline."""
    return {
        "delta_f0": round(current["f0_hz"] - baseline["f0_hz"], 2),
        "delta_jitter": round(current["jitter_pct"] - baseline["jitter_pct"], 3),
        "delta_shimmer": round(current["shimmer_pct"] - baseline["shimmer_pct"], 3),
        "delta_hnr": round(current["hnr_db"] - baseline["hnr_db"], 2),
    }
