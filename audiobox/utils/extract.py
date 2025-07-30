import torch
import torchaudio
import numpy as np
import torchcrepe
import librosa
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

# Constants
TARGET_HOP_SIZE = 1200  # For 48kHz → 40Hz
TARGET_RATE = 21.5  # Hz

def compute_loudness(waveform, sample_rate):
    window = torch.hann_window(1024, device=waveform.device)
    spectrogram = torch.abs(torch.stft(waveform, n_fft=1024, hop_length=TARGET_HOP_SIZE, window=window, return_complex=True))
    freqs = torch.linspace(0, sample_rate // 2, spectrogram.shape[1])
    a_weighting = 2.0 - torch.log10(1 + (freqs / 1000.0)**2)
    a_weighted = spectrogram * a_weighting[None, :, None]
    loudness = torch.sqrt((a_weighted ** 2).mean(dim=1))
    loudness_db = 20 * torch.log10(loudness + 1e-6)
    return loudness_db.squeeze().cpu().numpy()

def compute_pitch(waveform, sample_rate, device='cpu'):
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    pitch, periodicity = torchcrepe.predict(
        waveform,
        sample_rate=16000,
        hop_length=400,
        model='tiny',
        fmin=50.0,
        fmax=2000.0,
        return_periodicity=True,
        device=device
    )

    pitch_probs = torchcrepe.embed(waveform, sample_rate=16000, model='tiny', device=device)
    pitch_probs[pitch_probs < 0.1] = 0
    return pitch.squeeze().cpu().numpy(), pitch_probs.detach().cpu().numpy()

def compute_centroid(waveform, sample_rate):
    y = waveform.squeeze().numpy()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, hop_length=TARGET_HOP_SIZE)[0]
    midi_like = 69 + 12 * np.log2(centroid / 440.0 + 1e-6)
    midi_like = np.clip(midi_like, 0, 127)
    scaled = midi_like / 127.0
    return scaled

def align_and_filter(control_signal, original_rate, target_rate=TARGET_RATE, median_filter_size=10):
    # Interpolate to match 40Hz frame rate (e.g., DiT latent rate)
    original_len = len(control_signal)
    duration_sec = original_len / original_rate
    t_orig = np.linspace(0, duration_sec, num=original_len)
    t_target = np.linspace(0, duration_sec, num=int(duration_sec * target_rate))
    
    f_interp = interp1d(t_orig, control_signal, kind='linear', fill_value="extrapolate")
    aligned = f_interp(t_target)
    
    # Apply median filter
    filtered = median_filter(aligned, size=median_filter_size)
    return filtered

def normalize_to_01(signal, min_val=None, max_val=None):
    """Min-max normalize to [0, 1]. Optionally use fixed min/max."""
    if min_val is None:
        min_val = np.min(signal)
    if max_val is None:
        max_val = np.max(signal)
    return np.clip((signal - min_val) / (max_val - min_val + 1e-6), 0.0, 1.0)

def normalize_to_01_relative(pitch):
    pitch = pitch.copy()
    pitch[pitch == 0] = np.nan
    min_pitch = np.nanmin(pitch)
    max_pitch = np.nanmax(pitch)
    norm = (pitch - min_pitch) / (max_pitch - min_pitch + 1e-6)
    return np.nan_to_num(norm)

def get_dynamic_paper(waveform: torch.Tensor, sr: int, median_filter_size:int = 10) -> torch.Tensor:
    """
    Get loudness, pitch, brightness (spectral centroid)
    """
    loudness = compute_loudness(waveform, sr)
    centroid = compute_centroid(waveform, sr)
    pitch, _ = compute_pitch(waveform.to(dtype=torch.float32), sr)
    print('loudness ; ', loudness.shape)

    # Align and filter all control signals
    loudness_aligned = align_and_filter(loudness, original_rate=sr / TARGET_HOP_SIZE, median_filter_size=median_filter_size)
    centroid_aligned = align_and_filter(centroid, original_rate=sr / TARGET_HOP_SIZE, median_filter_size=median_filter_size)
    pitch_aligned = align_and_filter(pitch, original_rate=1000 / 25, median_filter_size=median_filter_size)  # torchcrepe gives ~25ms hops
    print('loudness_aligned ; ', loudness_aligned.shape)

    # Example usage after alignment
    # loudness_norm = normalize_to_01(loudness_aligned, min_val=loudness_aligned.min().item(), max_val=loudness_aligned.max().item())  # dB 기준
    # centroid_norm = normalize_to_01(centroid_aligned)  # 이미 scaled MIDI였지만 정규화 보정
    # pitch_norm = normalize_to_01_relative(pitch_aligned)  # CREPE 범위

    loudness_norm = loudness_aligned
    centroid_norm = centroid_aligned
    pitch_norm = pitch_aligned

    # Stack together into (3, T) format
    min_len = min([loudness_norm.shape[0], centroid_norm.shape[0], pitch_norm.shape[0]])

    loudness_norm = loudness_norm[:min_len]
    centroid_norm = centroid_norm[:min_len]
    pitch_norm = pitch_norm[:min_len]

    control_np = np.stack([loudness_norm, centroid_norm, pitch_norm])  # (3, T)
    T = control_np.shape[1]
    if T < 400:
        control_np = np.pad(control_np, ((0, 0), (0, 400 - T)))  # Pad 시간 축 (우측)
    control_tensor = torch.tensor(control_np, dtype=torch.float32)  # (3, 400)

    return control_tensor
