import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from einops import rearrange
from functools import lru_cache
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torchcrepe
import gc
from scipy.interpolate import interp1d

# ----------------- Constants ----------------- #
TARGET_RATE = 21.5
SAMPLE_RATE = 48000
INSERT_FOLDER_NAME = "dynamic_context"
TARGET_HOP_SIZE = 1200

# ----------------- Utility Functions ----------------- #
@lru_cache(maxsize=16)
def get_hann_window(window_length: int, device_str: str):
    return torch.hann_window(window_length).to(torch.device(device_str))

def compute_loudness(waveform: torch.Tensor, sample_rate: int, hop_length: int):
    window = get_hann_window(1024, str(waveform.device))
    spectrogram = torch.abs(torch.stft(
        waveform, n_fft=1024, hop_length=hop_length,
        window=window, return_complex=True))
    freqs = torch.linspace(0, sample_rate // 2, spectrogram.shape[1], device=waveform.device)
    a_weighting = 2.0 - torch.log10(1 + (freqs / 1000.0) ** 2)
    a_weighted = spectrogram * a_weighting[None, :, None]
    loudness = torch.sqrt((a_weighted ** 2).mean(dim=1))
    loudness_db = 20 * torch.log10(loudness + 1e-6)
    return loudness_db.squeeze()

def compute_pitch(waveform, sample_rate, device):
    # device를 문자열로 변환하여 torchcrepe 내부 캐시에 GPU 별로 저장되도록 함
    device_str = str(device)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    pitch, _ = torchcrepe.predict(
        waveform,
        sample_rate=16000,
        hop_length=400,
        model='tiny',
        fmin=50.0,
        fmax=2000.0,
        return_periodicity=True,
        device=device_str
    )
    pitch_probs = torchcrepe.embed(waveform, sample_rate=16000, model='tiny', device=device)
    print("pitch_probs : ", pitch_probs)
    pitch_probs[pitch_probs < 0.2] = 0
    return pitch.squeeze(), pitch_probs.detach()

def fast_spectral_centroid(waveform: torch.Tensor, sr: int, hop_length: int):
    stft = torch.stft(waveform, n_fft=1024, hop_length=hop_length, return_complex=True)
    magnitudes = stft.abs()
    freqs = torch.linspace(0, sr / 2, magnitudes.shape[1], device=waveform.device)
    centroids = (freqs[:, None] * magnitudes[0]).sum(dim=0) / (magnitudes[0].sum(dim=0) + 1e-8)
    midi_like = 69 + 12 * torch.log2(centroids / 440.0 + 1e-6)
    return midi_like.clamp(0, 127).cpu().numpy() / 127.0

def compute_centroid(waveform, sample_rate, hop_length: int = TARGET_HOP_SIZE):
    return fast_spectral_centroid(waveform=waveform.squeeze(), sr=sample_rate, hop_length=hop_length)

def align_and_filter(control_signal, original_rate, target_rate=TARGET_RATE, median_filter_size=10):
    duration_sec = len(control_signal) / original_rate
    t_orig = np.linspace(0, duration_sec, num=len(control_signal))
    t_target = np.linspace(0, duration_sec, num=int(duration_sec * target_rate))

    # f_interp = interp1d(t_orig, control_signal, kind='linear', fill_value="extrapolate")
    # filtered = f_interp(t_target)

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

def get_dynamic_paper(waveform: torch.Tensor, sr: int, device: torch.device, median_filter_size: int = 10):
    loudness = compute_loudness(waveform, sr, TARGET_HOP_SIZE).cpu().numpy()
    centroid = compute_centroid(waveform, sr, TARGET_HOP_SIZE)
    pitch = compute_pitch(waveform.to(dtype=torch.float32), sample_rate=sr, device=device)[0].cpu().numpy()

    loudness_aligned = align_and_filter(loudness, sr / TARGET_HOP_SIZE, median_filter_size=median_filter_size)
    centroid_aligned = align_and_filter(centroid, sr / TARGET_HOP_SIZE, median_filter_size=median_filter_size)
    pitch_aligned = align_and_filter(pitch, 40.0, median_filter_size=median_filter_size)

    # loudness_norm = normalize_to_01(loudness_aligned)
    # centroid_norm = normalize_to_01(centroid_aligned)
    # pitch_norm = normalize_to_01_relative(pitch_aligned)

    loudness_norm = loudness_aligned
    centroid_norm = centroid_aligned
    pitch_norm = pitch_aligned

    print("Lens : ", len(loudness_norm), len(centroid_norm), len(pitch_norm))
    min_len = min(len(loudness_norm), len(centroid_norm), len(pitch_norm))
    control_np = np.stack([
        loudness_norm[:min_len],
        centroid_norm[:min_len],
        pitch_norm[:min_len]
    ])
    if min_len < 400:
        control_np = np.pad(control_np, ((0, 0), (0, 400 - min_len)))
    
    return torch.tensor(control_np, dtype=torch.float32)

def build_new_npy_path(orig_audio_path: str):
    audio_wavpath = orig_audio_path.replace("/data/", "/")
    path_parts = audio_wavpath.split(os.sep)
    new_dir = os.path.join(os.sep, path_parts[1], path_parts[2], INSERT_FOLDER_NAME, *path_parts[3:-1])
    os.makedirs(new_dir, exist_ok=True)
    base_name = os.path.splitext(path_parts[-1])[0] + ".npy"
    return os.path.join(new_dir, base_name), audio_wavpath

def process_file(row, gpu_id):
    # GPU 할당: 각 프로세스는 전달받은 gpu_id에 해당하는 디바이스를 사용
    device = torch.device(f"cuda:{gpu_id}")
    try:
        npy_path, audio_wavpath = build_new_npy_path(row['audio_path'])
        if os.path.exists(npy_path):
            return npy_path

        waveform, sr = torchaudio.load(audio_wavpath)
        waveform = waveform.to(device)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE).to(device)
            waveform = resampler(waveform)
            sr = SAMPLE_RATE

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        dynamic_context = get_dynamic_paper(waveform, sr, device)
        dynamic_context = rearrange(dynamic_context, 's t -> t s')
        np.save(npy_path, dynamic_context.cpu().numpy())
        del dynamic_context
        del waveform
        return npy_path
    except Exception as e:
        print(f"[GPU {gpu_id}] 오류 발생: {e} | 파일: {row.get('audio_path')}")
        return None

def parallel_process(rows, gpu_ids):
    tasks = []
    results = []
    # 각 행에 대해 라운드로빈 방식으로 gpu_id를 할당
    for i, row in enumerate(rows):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((row, gpu_id))
    # ProcessPoolExecutor를 사용하여 각 작업을 별도 프로세스에서 실행
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(process_file, row, gpu_id) for row, gpu_id in tasks]
        for f in tqdm(futures, desc="Processing files"):
            result = f.result()
            if result is not None:
                results.append(result)
    return results

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    df = pd.read_csv("./toextract.csv")
    print(f"✅ CSV 로드 완료: {len(df)}개 항목")
    # df = df.drop_duplicates(subset='audio_path', keep='first')
    # print(f"✅ 중복 제거 완료: {len(df)}개 항목")
    # df.to_csv("toextract.csv")

    rows = df.to_dict("records")
    # 이미 처리된 항목 건너뛰기
    rows = [r for r in rows if not os.path.exists(build_new_npy_path(r['audio_path'])[0])]
    print(f"🔍 처리할 항목 수: {len(rows)}")
    gpu_ids = [0, 1, 2, 3]
    for i in range(5):
        print(f"\n\n{i} 번째\n\n")
        try:
            results = parallel_process(rows[i*10000 : i*10000 + 10000], gpu_ids)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print("심각한 에러 : ", e)
            torch.cuda.empty_cache()
            gc.collect()
            continue
        print("🎉 처리 완료. 저장된 파일 목록:")
