import os
import numpy as np
import matplotlib.pyplot as plt
import os
import torchaudio
import torch
import torchaudio
import numpy as np
import torchcrepe
import librosa
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from einops import rearrange
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
from utils.utils import get_dynamic
from utils.extract import get_dynamic_paper
# from utils.extract2 import extract_sketch2sound_controls
from make_dynamics2 import get_dynamic_paper

# def count_all_files(directory):
#     count = 0
#     for root, dirs, files in os.walk(directory):
#         count += len(files)
#     return count

# # 예시 사용
# folder_path = "/home/khj6051/dynamic_context"  # 여기에 확인하고 싶은 폴더 경로 입력
# total_files = count_all_files(folder_path)
# print(f"총 파일 수: {total_files}")

# print(np.load('/home/khj6051/dynamic_context/zapsplat-audios/animals/344_audio_BIRDFowl_Angry_goose_wings_flapping_hissing_344_Audio_Geese_1718.npy').shape)
# print(np.load('/home/khj6051/dynamic_context/zapsplat-audios/food-and-drink/page_1/zapsplat_food_drink_can_pop_soft_drink_cola_full_unopened_knock_over_on_concrete_001_108871.npy').shape)

# # npy 파일 불러오기
# data = np.load('/home/khj6051/dynamic_context/zapsplat-audios/animals/344_audio_BIRDFowl_Angry_goose_wings_flapping_hissing_344_Audio_Geese_1718.npy')  # 파일 경로 바꿔줘야 함

SAMPLE_RATE = 44100

audios = [
    '/home/khj6051/alignment-v3/audiobox/sste.wav',
    # '/home/khj6051/alignment-v3/audiobox/voice_samples/piung.wav',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/beepbeep.m4a',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/charging.m4a',
]

waveform, sr = torchaudio.load(audios[0])
print("duration : ", waveform.shape[-1]/sr)
if sr != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    # waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
# waveform = waveform.mean(dim=0, keepdim=True)
# dynamic_context = get_dynamic_paper(waveform, sr=SAMPLE_RATE, median_filter_size=10)
waveform = waveform.mean(dim=0, keepdim=True).to('cuda')
dynamic = get_dynamic(waveform.cpu(), 400)

import torch
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

vals = ['loudness', 'centroid', 'pitch']
axes[0].plot(dynamic[1][0][0].squeeze())
axes[0].set_title('loud')
axes[0].set_ylabel("Value")
axes[1].plot(dynamic[2].squeeze())
axes[1].set_title('centroid')
axes[1].set_ylabel("Value")
axes[1].set_xlabel("Index")
plt.tight_layout()
plt.savefig("components_all_in_one2.png")
plt.close()

# 시각화
plt.figure(figsize=(10, 4))
plt.imshow(dynamic[0][0, :50, :].T, aspect='auto', origin='lower', cmap='viridis')  # [channel x time]
plt.colorbar(label='Value')
plt.xlabel('Time Frame')
plt.ylabel('Channel')
plt.title('Feature Map (16 channels over 400 time steps)')
plt.tight_layout()
plt.savefig("components_all_in_one3.png")
plt.close()

print("dynamic : ", dynamic.shape)
# dynamic_context = get_dynamic_paper(waveform, sr=SAMPLE_RATE, median_filter_size=10, device='cuda')
# data = rearrange(dynamic_context, 's t -> t s')

# Load audio (48kHz mono preferred)
# waveform, sr = torchaudio.load('/home/khj6051/alignment-v3/audiobox/sste.wav')
# waveform = waveform.to("cuda")  # or "cpu"

# features = extract_sketch2sound_controls(waveform, sr, median_kernel=10)

# print("Loudness:", features["loudness"].shape)
# print("Centroid:", features["centroid"].shape)
# print("Pitch:", features["pitch"].shape)


# output_dir = 'plots'
# os.makedirs(output_dir, exist_ok=True)

# fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# vals = ['loudness', 'centroid', 'pitch']
# # for i in range(3):
# #     aa = features[vals[i]]
# #     axes[i].plot(aa)
# #     axes[i].set_title(vals[i])
# #     axes[i].set_ylabel("Value")
# #     if i == 2:
# #         axes[i].set_xlabel("Index")

# for i in range(3):
#     nonzero_length = np.count_nonzero(data[:, i])
#     print(f"0이 아닌 부분의 길이: {nonzero_length}")
#     axes[i].plot(data[:nonzero_length, i])
#     axes[i].set_title(vals[i])
#     axes[i].set_ylabel("Value")
#     if i == 2:
#         axes[i].set_xlabel("Index")

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "components_all_in_one2.png"))
# plt.close()