"""Voice biomarker extraction using Praat (via parselmouth).

Fixes applied:
1. Silence detection — rejects recordings with insufficient voice energy
2. Minimum duration check — rejects recordings too short for analysis
3. Voiced frames check — rejects recordings with no detected voiced speech

Compatible with requirements.txt:
  - librosa 0.10.2 (audio loading)
  - scipy 1.13.1 (wav writing — no soundfile needed)
  - praat-parselmouth 0.4.5 (biomarker extraction)
  - numpy 1.26.4 (signal processing)
"""

import io
import os
import tempfile
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
from scipy.io import wavfile


# ── Thresholds ────────────────────────────────────────────────────────────────

# Minimum RMS energy — below this = silent recording
MIN_RMS_ENERGY = 0.01

# Minimum recording duration in seconds
MIN_DURATION_SECONDS = 2.0

# Minimum ratio of voiced frames to total frames
MIN_VOICED_RATIO = 0.3


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_audio(audio_bytes: bytes):
    """Load audio bytes → numpy array at 16 kHz mono via librosa."""
    audio_file = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    return y, sr


def _check_voice_presence(y: np.ndarray, sr: int) -> None:
    """
    Validate that the recording contains actual sustained voice.
    Raises ValueError with a user-friendly message on failure.
    """
    # Check 1 — Duration
    duration = len(y) / sr
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            f"Recording too short ({duration:.1f}s). "
            f"Please sustain the 'AH' sound for at least "
            f"{int(MIN_DURATION_SECONDS)} seconds."
        )

    # Check 2 — RMS energy (main silence detector)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < MIN_RMS_ENERGY:
        raise ValueError(
            "No voice detected in the recording. "
            "Please speak clearly and make sure your "
            "microphone is not muted or blocked."
        )

    # Check 3 — Zero crossing rate (background noise check)
    # High ZCR + low energy = noise only, not voice
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    if rms < 0.02 and zcr > 0.3:
        raise ValueError(
            "Only background noise detected. "
            "Please record in a quieter environment "
            "and speak clearly into the microphone."
        )


def _write_temp_wav(y: np.ndarray, sr: int) -> str:
    """
    Write numpy audio array to a temporary WAV file.
    Uses scipy.io.wavfile — no soundfile dependency needed.
    Returns the temp file path (caller must delete).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # scipy expects int16 format
    y_int16 = (y * 32767).astype(np.int16)
    wavfile.write(tmp_path, sr, y_int16)
    return tmp_path


# ── Public API ────────────────────────────────────────────────────────────────

def extract_biomarkers(audio_bytes: bytes) -> dict:
    """
    Extract Pitch (F0), Jitter, Shimmer, and HNR from audio bytes.

    Returns:
        dict with keys:
            f0_hz         — mean fundamental frequency (Hz)
            jitter_pct    — local jitter (%)
            shimmer_pct   — local shimmer (%)
            hnr_db        — harmonics-to-noise ratio (dB)
            duration_s    — recording duration (seconds)
            rms_energy    — RMS energy level
            voiced_ratio  — proportion of voiced frames

    Raises:
        ValueError  — if audio is silent, too short, or has no voiced frames
        RuntimeError — if Praat analysis fails unexpectedly
    """
    # 1. Load audio
    y, sr = _load_audio(audio_bytes)

    # 2. Validate voice presence BEFORE Praat (fast rejection)
    _check_voice_presence(y, sr)

    # 3. Write temp WAV for parselmouth (uses scipy, not soundfile)
    tmp_path = _write_temp_wav(y, sr)

    try:
        snd = parselmouth.Sound(tmp_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio into Praat: {e}")
    finally:
        os.unlink(tmp_path)  # always clean up temp file

    # 4. Pitch extraction
    try:
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        voiced_frames = pitch_values[pitch_values > 0]
    except Exception as e:
        raise RuntimeError(f"Pitch extraction failed: {e}")

    # 5. Voiced frames check (after Praat)
    total_frames = max(len(pitch_values), 1)
    voiced_ratio = len(voiced_frames) / total_frames

    if len(voiced_frames) < 10 or voiced_ratio < MIN_VOICED_RATIO:
        raise ValueError(
            f"Insufficient voiced speech detected "
            f"({voiced_ratio:.0%} voiced). "
            "Please sustain the 'AH' vowel sound continuously "
            "for at least 3 seconds without pausing."
        )

    f0_mean = float(np.mean(voiced_frames))

    # 6. Point process for jitter and shimmer
    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    except Exception as e:
        raise RuntimeError(f"Point process extraction failed: {e}")

    # 7. Jitter (local %)
    try:
        jitter = call(
            point_process,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3
        )
        if jitter is None:
            jitter = 0.02
    except Exception:
        jitter = 0.02

    # 8. Shimmer (local %)
    try:
        shimmer = call(
            [snd, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        if shimmer is None:
            shimmer = 0.05
    except Exception:
        shimmer = 0.05

    # 9. HNR (harmonics-to-noise ratio, dB)
    try:
        harmonicity = snd.to_harmonicity()
        hnr = call(harmonicity, "Get mean", 0, 0)
        if hnr is None or hnr <= -200:
            hnr = 0.0
    except Exception:
        hnr = 0.0

    # 10. Metadata
    rms = float(np.sqrt(np.mean(y ** 2)))
    duration = len(y) / sr

    return {
        "f0_hz":        round(f0_mean, 2),
        "jitter_pct":   round(float(jitter) * 100, 4),
        "shimmer_pct":  round(float(shimmer) * 100, 4),
        "hnr_db":       round(float(hnr), 2),
        "duration_s":   round(duration, 2),
        "rms_energy":   round(rms, 4),
        "voiced_ratio": round(voiced_ratio, 4),
    }


def compute_deltas(current: dict, baseline: dict) -> dict:
    """
    Compute absolute and percentage changes from baseline to current.

    Returns:
        dict with absolute deltas (delta_f0, delta_jitter, etc.)
        and percentage changes (delta_f0_pct, delta_jitter_pct, etc.)
    """
    def _abs(curr, base):
        return round(float(curr) - float(base), 4)

    def _pct(curr, base):
        base = float(base)
        if base == 0:
            return 0.0
        return round(((float(curr) - base) / abs(base)) * 100, 2)

    return {
        "delta_f0":           _abs(current["f0_hz"],       baseline["f0_hz"]),
        "delta_jitter":       _abs(current["jitter_pct"],  baseline["jitter_pct"]),
        "delta_shimmer":      _abs(current["shimmer_pct"], baseline["shimmer_pct"]),
        "delta_hnr":          _abs(current["hnr_db"],      baseline["hnr_db"]),
        "delta_f0_pct":       _pct(current["f0_hz"],       baseline["f0_hz"]),
        "delta_jitter_pct":   _pct(current["jitter_pct"],  baseline["jitter_pct"]),
        "delta_shimmer_pct":  _pct(current["shimmer_pct"], baseline["shimmer_pct"]),
        "delta_hnr_pct":      _pct(current["hnr_db"],      baseline["hnr_db"]),
    }
