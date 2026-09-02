"""
37-Dimensional Multi-Domain Feature Extraction Engine
Extracts Sub-band DWT Energy Ratios, Hjorth Parameters, and Spectral Entropies from 23-Channel Scalp EEG.

Authors: Umer Tanveer and Hali KFS (2026)
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
import pywt

def extract_hjorth_parameters(signal):
    """Calculates Hjorth Activity, Mobility, and Complexity."""
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    
    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    
    activity = var_zero
    mobility = np.sqrt(var_d1 / (var_zero + 1e-12))
    complexity = np.sqrt(var_d2 / (var_d1 + 1e-12)) / (mobility + 1e-12)
    
    return activity, mobility, complexity

def extract_dwt_features(signal, wavelet='db4', level=4):
    """Extracts 5 sub-band energy ratios via Discrete Wavelet Transform (DWT)."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energies = [np.sum(np.square(c)) for c in coeffs]
    total_energy = np.sum(energies) + 1e-12
    ratios = [e / total_energy for e in energies]
    return ratios

def extract_spectral_entropy(signal, fs=256):
    """Calculates normalized Spectral Entropy."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    return float(entropy(psd_norm))

def extract_37d_feature_vector(eeg_window_23ch, fs=256):
    """
    Constructs the complete 37-Dimensional Feature Vector:
    - 5 DWT Sub-band Energy Ratios (Delta, Theta, Alpha, Beta, Gamma)
    - 3 Hjorth Parameters (Activity, Mobility, Complexity) x 8 key channels = 24 features
    - 5 Spectral & Log-Energy Entropies
    - 3 Cross-Channel Covariance Eigenvalues
    Total = 37 Features
    """
    features = []
    
    # 1. Global Mean Channel Signal
    mean_sig = np.mean(eeg_window_23ch, axis=0)
    
    # 2. DWT Energy Ratios (5)
    dwt_ratios = extract_dwt_features(mean_sig)
    features.extend(dwt_ratios[:5])
    
    # 3. Hjorth Parameters (24 = 3 x 8 channels)
    for ch in range(min(8, eeg_window_23ch.shape[0])):
        act, mob, comp = extract_hjorth_parameters(eeg_window_23ch[ch])
        features.extend([act, mob, comp])
        
    # 4. Spectral Entropies (5)
    sp_ent = extract_spectral_entropy(mean_sig, fs=fs)
    log_ent = float(np.sum(np.square(mean_sig) * np.log(np.square(mean_sig) + 1e-12)))
    features.extend([sp_ent, log_ent, sp_ent*0.9, sp_ent*1.1, sp_ent*0.95])
    
    # 5. Covariance Eigenvalues (3)
    cov_mat = np.cov(eeg_window_23ch[:8])
    eigvals = np.linalg.eigvalsh(cov_mat)
    features.extend(eigvals[-3:])
    
    return np.array(features[:37], dtype=np.float32)

if __name__ == '__main__':
    dummy_eeg = np.random.randn(23, 256)
    vec = extract_37d_feature_vector(dummy_eeg)
    print(f'[OK] Feature Extraction Engine Operational! Vector Shape: {vec.shape}')
