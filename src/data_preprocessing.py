"""
data_preprocessing.py
---------------------
Modul untuk memuat dan melakukan preprocessing sinyal audio PCG.
Termasuk filter Butterworth dan penanganan durasi audio.
"""
import numpy as np
from scipy.signal import butter, filtfilt
import librosa

def butter_bandpass_filter(data, lowcut=25.0, highcut=400.0, fs=2000, order=4):
    """Mengaplikasikan Butterworth bandpass filter untuk membersihkan noise."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def pad_or_truncate_audio(y, target_length):
    """Menyamakan durasi audio menjadi panjang target tertentu."""
    if len(y) < target_length:
        return np.pad(y, (0, target_length - len(y)), 'constant')
    return y[:target_length]

def extract_features(y, sr):
    """Mengekstrak Mel-Spectrogram dan 33 fitur statistik untuk model."""
    pass # Implementasi penuh ada di Notebook (Train_Ensemble_CRNN_XGBoost_Colab.ipynb)
