import os
import torch
import torchaudio
import numpy as np
import torchcrepe
import librosa
from scipy.ndimage import median_filter
from functools import lru_cache
import pandas as pd
from tqdm import tqdm
from einops import rearrange
import concurrent.futures
import time

# ----------------- Constants ----------------- #
TARGET_RATE = 21.5      # Hz
SAMPLE_RATE = 44100     # 목표 샘플레이트 (예시)
INSERT_FOLDER_NAME = "dynamic_context"  # 새로 삽입할 폴더 이름
TARGET_HOP_SIZE = int(0.025 * SAMPLE_RATE)  # For 48kHz → 40Hz

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"사용할 장치: {device}")

# ----------------- Utility Functions ----------------- #
@lru_cache(maxsize=4)
def get_hann_window(window_length: int, device_str: str):
    """캐싱을 이용해 Hann 윈도우를 생성합니다."""
    return torch.hann_window(window_length, device=device_str)

# TorchScript 컴파일 여부 (실험적인 단계)
def jit_script_if_possible(func):
    try:
        return torch.jit.script(func)
    except Exception as e:
        print(f"JIT 컴파일 실패 ({func.__name__}): {e}")
        return func

@jit_script_if_possible
def compute_loudness(waveform: torch.Tensor, sample_rate: int, hop_length: int):
    window = get_hann_window(1024, str(waveform.device))
    spectrogram = torch.abs(
        torch.stft(
            waveform, n_fft=1024, hop_length=hop_length,
            window=window, return_complex=True
        )
    )
    freqs = torch.linspace(0, sample_rate // 2, spectrogram.shape[1], device=waveform.device)
    a_weighting = 2.0 - torch.log10(1 + (freqs / 1000.0)**2)
    a_weighted = spectrogram * a_weighting[None, :, None]
    loudness = torch.sqrt((a_weighted ** 2).mean(dim=1))
    loudness_db = 20 * torch.log10(loudness + 1e-6)
    return loudness_db.squeeze()

def compute_pitch(waveform, sample_rate, device):
    # 만약 sample_rate가 16000이 아니면 resample 후, waveform을 해당 device로 이동합니다.
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)
    pitch, _ = torchcrepe.predict(
        waveform,
        sample_rate=16000,
        hop_length=400,
        model='tiny',
        fmin=50.0,
        fmax=2000.0,
        return_periodicity=True,
        device=device  # 여기서 지정한 local device 사용
    )
    torch.cuda.empty_cache()
    return pitch.squeeze()


def compute_centroid(waveform, sample_rate, hop_length: int = TARGET_HOP_SIZE):
    y = waveform.squeeze().cpu().numpy()  # CPU 기반 라이브러리 사용
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, hop_length=hop_length)[0]
    midi_like = 69 + 12 * np.log2(centroid / 440.0 + 1e-6)
    return np.clip(midi_like, 0, 127) / 127.0

def align_and_filter(control_signal, original_rate, target_rate=TARGET_RATE, median_filter_size=10):
    original_len = len(control_signal)
    duration_sec = original_len / original_rate
    t_orig = np.linspace(0, duration_sec, num=original_len)
    t_target = np.linspace(0, duration_sec, num=int(duration_sec * target_rate))
    aligned = np.interp(t_target, t_orig, control_signal)
    filtered = median_filter(aligned, size=median_filter_size)
    return filtered

def normalize_to_01(signal, min_val=None, max_val=None):
    min_val = np.min(signal) if min_val is None else min_val
    max_val = np.max(signal) if max_val is None else max_val
    norm = (signal - min_val) / (max_val - min_val + 1e-6)
    return np.clip(norm, 0.0, 1.0)

def normalize_to_01_relative(pitch):
    pitch = pitch.copy()
    pitch[pitch == 0] = np.nan
    min_pitch = np.nanmin(pitch)
    max_pitch = np.nanmax(pitch)
    norm = (pitch - min_pitch) / (max_pitch - min_pitch + 1e-6)
    return np.nan_to_num(norm)

def get_dynamic_paper(waveform: torch.Tensor, sr: int, median_filter_size: int = 10) -> torch.Tensor:
    """
    Loudness, 피치, brightness(스펙트럼 센트로이드)를 계산하여 tensor로 반환합니다.
    """
    # 가능하면 GPU에서 계산 후, 최종 결과만 CPU로 이동
    device_local = waveform.device
    loudness = compute_loudness(waveform, sr, TARGET_HOP_SIZE).cpu().numpy()
    centroid = compute_centroid(waveform, sr, TARGET_HOP_SIZE)
    pitch = compute_pitch(waveform.to(dtype=torch.float32), sample_rate=sr, device=device_local).cpu().numpy()
    
    # 컨트롤 신호 보간 및 필터링 (여기서는 CPU 기반 numpy 사용)
    loudness_aligned = align_and_filter(loudness, original_rate=sr / TARGET_HOP_SIZE, median_filter_size=median_filter_size)
    centroid_aligned = align_and_filter(centroid, original_rate=sr / TARGET_HOP_SIZE, median_filter_size=median_filter_size)
    pitch_aligned = align_and_filter(pitch, original_rate=40.0, median_filter_size=median_filter_size)
    
    loudness_norm = normalize_to_01(loudness_aligned)
    centroid_norm = normalize_to_01(centroid_aligned)
    pitch_norm = normalize_to_01_relative(pitch_aligned)
    
    min_len = min(loudness_norm.shape[0], centroid_norm.shape[0], pitch_norm.shape[0])
    loudness_norm = loudness_norm[:min_len]
    centroid_norm = centroid_norm[:min_len]
    pitch_norm = pitch_norm[:min_len]
    control_np = np.stack([loudness_norm, centroid_norm, pitch_norm])
    
    T = control_np.shape[1]
    if T < 400:
        control_np = np.pad(control_np, ((0, 0), (0, 400 - T)))
    return torch.tensor(control_np, dtype=torch.float32)

def build_new_npy_path(orig_audio_path: str) -> (str, str):
    """원본 오디오 경로에서 저장할 npy 파일 경로를 생성합니다."""
    audio_wavpath = orig_audio_path.replace("/data/", "/")
    path_parts = audio_wavpath.split(os.sep)
    if len(path_parts) < 4:
        raise ValueError(f"경로 형식이 예상과 다름: {audio_wavpath}")
    new_dir = os.path.join(os.sep, path_parts[1], path_parts[2], INSERT_FOLDER_NAME, *path_parts[3:-1])
    os.makedirs(new_dir, exist_ok=True)
    base_name = os.path.splitext(path_parts[-1])[0] + ".npy"
    return os.path.join(new_dir, base_name), audio_wavpath

import os
from scipy.ndimage import median_filter
from functools import lru_cache
import pandas as pd
from tqdm import tqdm
from einops import rearrange
import concurrent.futures
import multiprocessing as mp

# (중략: 기존 함수 정의 부분은 동일)

def process_file(args):
    row, gpu_id = args
    device_local = torch.device(f"cuda:{gpu_id}")
    
    try:
        orig_audio_path = row['audio_path']
        npy_path, audio_wavpath = build_new_npy_path(orig_audio_path)
        if os.path.exists(npy_path):
            return npy_path

        waveform, sr = torchaudio.load(audio_wavpath)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 필요하다면 각 함수에 device_local 인수를 전달하도록 수정하세요.
        dynamic_context = get_dynamic_paper(waveform.to(device_local), sr=sr)  # 예시: 내부에서 device_local 활용
        dynamic_context = rearrange(dynamic_context, 's t -> t s')
        
        np.save(npy_path, dynamic_context.cpu().numpy())
        return npy_path
    except Exception as e:
        print(f"파일 처리 중 에러 발생 {row.get('audio_path')}: {e}")
        return None

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    df = pd.read_csv("./total_voice_0409.csv")
    print(f"CSV 파일 행 수: {len(df)}")
    rows = df.to_dict("records")
    
    gpu_ids = [0, 1, 2, 3]  # 사용 가능한 GPU ID 목록
    args_list = [(row, gpu_ids[i % len(gpu_ids)]) for i, row in enumerate(rows)]

    spawn_context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(mp_context=spawn_context) as executor:
        results = list(tqdm(executor.map(process_file, args_list, chunksize=10), total=len(args_list)))
    
    print("처리 완료. 저장된 파일 경로:")
    for r in results:
        if r is not None:
            print(r)