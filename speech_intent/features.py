import numpy as np
import librosa
from scipy.stats import skew

def extract_audio_features(audio_path):
    """
    Extract vocal features for emotion analysis.
    
    Returns:
        np.array: feature vector
    """
    y, sr = librosa.load(audio_path, sr=None)
    y, _ = librosa.effects.trim(y)
    
    if len(y) == 0:
        return np.zeros(88)  # return zeros if audio is empty

    # Normalize (avoid division by zero)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # --- Pitch using librosa.yin for robustness ---
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    voiced = f0 > 0
    avg_pitch = float(np.mean(f0[voiced])) if np.any(voiced) else 0
    std_pitch = float(np.std(f0[voiced])) if np.any(voiced) else 0
    min_pitch = float(np.min(f0[voiced])) if np.any(voiced) else 0
    max_pitch = float(np.max(f0[voiced])) if np.any(voiced) else 0

    # --- RMS (energy) ---
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))
    min_rms = float(np.min(rms))
    max_rms = float(np.max(rms))

    # --- Spectral features ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    features = [
        avg_pitch, std_pitch, min_pitch, max_pitch,
        avg_rms, std_rms, min_rms, max_rms,
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(zcr), np.std(zcr)
    ]

    # --- MFCCs + delta + delta-delta ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta1 = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    
    for arr in [mfccs, delta1, delta2]:
        features.extend(np.mean(arr, axis=1))
        features.extend(np.std(arr, axis=1))
        features.extend(skew(arr, axis=1))

    return np.array(features)

def get_feature_count():
    # Create dummy audio with proper sample rate
    dummy_audio = np.zeros(22050)
    # This would need a real audio file to work properly
    return 88  # Expected feature vector length

def normalize_features(features):
    """Normalize features to zero mean and unit variance."""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # prevent division by zero
    return (features - mean) / std

def pad_or_truncate(features, target_length=88):
    """Pad or truncate feature vector to target length."""
    if len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), 'constant')
    else:
        return features[:target_length]
    return features[:target_length] 

def prepare_features(features):
    features = pad_or_truncate(features)
    features = normalize_features(features)
    return features



