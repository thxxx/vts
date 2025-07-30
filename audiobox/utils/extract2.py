import torch
import torchaudio
import torchcrepe
import torch.nn.functional as F
from scipy.signal import medfilt
import numpy as np


# -------------------------------
# Parameters
# -------------------------------
TARGET_FRAMERATE = 21.5  # desired control frame rate
HOP_LENGTH = int(48000 / TARGET_FRAMERATE)  # assuming 48kHz input audio
N_FFT = 2048


# -------------------------------
# Utility: Median Filter (per-frame)
# -------------------------------
def apply_median_filter(tensor, kernel_size=9):
    # tensor: [T] or [B, T]
    if tensor.ndim == 1:
        return torch.from_numpy(medfilt(tensor.numpy(), kernel_size=kernel_size))
    else:
        filtered = []
        for t in tensor:
            filtered.append(torch.from_numpy(medfilt(t.numpy(), kernel_size=kernel_size)))
        return torch.stack(filtered)


# -------------------------------
# Loudness (A-weighted RMS)
# -------------------------------
def compute_loudness(waveform, sample_rate):
    specs = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to(waveform.device)
    
    spec = specs(waveform)  # [B, F, T]
    freqs = torch.linspace(0, sample_rate // 2, spec.shape[1], device=waveform.device)
    A_weighting = 2.0 + 20.0 * torch.log10(freqs + 1e-6)
    weighted_spec = spec * A_weighting[None, :, None]
    loudness = torch.sqrt(torch.mean(weighted_spec**2, dim=1))  # RMS across freq
    return loudness  # [B, T]


# -------------------------------
# Spectral Centroid (scaled MIDI-like)
# -------------------------------
def compute_centroid(waveform, sample_rate):
    specs = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to(waveform.device)
    spec = specs(waveform)  # [B, F, T]
    freqs = torch.linspace(0, sample_rate // 2, spec.shape[1], device=waveform.device)
    centroid = (spec * freqs[None, :, None]).sum(dim=1) / (spec.sum(dim=1) + 1e-6)
    midi_like = 69 + 12 * torch.log2(centroid / 440.0 + 1e-6)
    scaled = (midi_like / 127.0).clamp(0, 1)
    return scaled  # [B, T]


# -------------------------------
# Pitch Probabilities (CREPE)
# -------------------------------
def compute_pitch_probabilities(waveform, sample_rate):
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    print(waveform.shape, 'waveform')

    pitch, periodicity = torchcrepe.predict(
        waveform,
        sample_rate=sample_rate,
        hop_length=int(16000 / TARGET_FRAMERATE),
        model='tiny',
        fmin=50,
        fmax=2000,
        return_periodicity=True,
        batch_size=128,
        device=waveform.device,
        pad=True
    )

    # Mask low periodicity
    pitch[periodicity < 0.5] = 0.0
    return pitch, periodicity  # [T], [T]


# -------------------------------
# Main Feature Extraction Function
# -------------------------------
def extract_sketch2sound_controls(waveform, sample_rate, median_kernel=9):
    loudness = compute_loudness(waveform, sample_rate)[0]  # [T]
    centroid = compute_centroid(waveform, sample_rate)[0]  # [T]
    pitch, periodicity = compute_pitch_probabilities(waveform, sample_rate)

    # Apply median filtering
    loudness = apply_median_filter(loudness.cpu(), median_kernel)
    centroid = apply_median_filter(centroid.cpu(), median_kernel)
    pitch = apply_median_filter(pitch.cpu(), median_kernel)

    return {
        "loudness": loudness,           # [T]
        "centroid": centroid,           # [T]
        "pitch": pitch,                 # [T]
        "periodicity": periodicity      # [T]
    }

