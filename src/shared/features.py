import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy

SAMPLING_RATE = 173.61  # Hz

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 60),
}

BAND_NAMES = list(BANDS.keys())
FEATURE_NAMES = [f"{b}_power_norm" for b in BAND_NAMES] + ["entropy", "rms"]  # 7 features


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract a flat feature vector from a single EEG window.

    Features (7): normalised band powers (delta, theta, alpha, beta, gamma),
                  spectral entropy, RMS amplitude.
    """
    # Use Welch's method to estimate the power spectral density 
    freqs, psd = welch(window, fs=SAMPLING_RATE, nperseg=min(len(window), 128))

    df = freqs[1] - freqs[0]  # frequency resolution

    # Get the absolute band powers (Use rectangle rule (could also integrate w trapezoidal rule?))
    powers = np.array([
        np.sum(psd[(freqs >= lo) & (freqs <= hi)]) * df
        for lo, hi in BANDS.values()
    ])

    total_power = np.sum(psd) * df
    
    # Normalise band powers to sum to 1 (relative band power)
    powers_norm = powers / (total_power + 1e-12)

    p = psd / (psd.sum() + 1e-12)
    raw_entropy = float(scipy_entropy(p))
    entropy = raw_entropy / np.log(len(p)) if len(p) > 1 else 0.0

    # root mean square amplitude 
    rms = float(np.sqrt(np.mean(window ** 2)))

    return np.array([*powers_norm, entropy, rms], dtype=np.float64)