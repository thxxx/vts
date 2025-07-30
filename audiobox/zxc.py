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
import torch
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100

audios = [
    '/home/khj6051/alignment-v3/audiobox/voice_samples/ss_science_fiction_page_1_512123-Energy_Weapon_Charging_01_-SCIWeap-energy-charging-weapon.mp3',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/Crisp_notification_sound_indicating_success_in_the_user_interface..wav',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/CC-DS Body Fall Concrete Soft 02-glued.wav',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/ss_comic_film_fx_page_1001_218135-ICE_Skate_blade_metal_scrape_21.mp3',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/piung.wav',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/beepbeep.m4a',
    '/home/khj6051/alignment-v3/audiobox/voice_samples/charging.m4a',
]

for idx in [0, 1, 2, 3, 4, 5, 6]:
    waveform, sr = torchaudio.load(audios[idx])
    print("duration : ", waveform.shape[-1]/sr)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
        # waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    # waveform = waveform.mean(dim=0, keepdim=True)
    # dynamic_context = get_dynamic_paper(waveform, sr=SAMPLE_RATE, median_filter_size=10)
    waveform = waveform.mean(dim=0, keepdim=True).to('cuda')
    dynamic = get_dynamic(waveform.cpu(), 400)

    print(dynamic.shape)

    # fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # vals = ['loudness', 'centroid', 'pitch']
    # axes[0].plot(dynamic[1][0][0].squeeze())
    # axes[0].set_title('loud')
    # axes[0].set_ylabel("Value")
    # axes[1].plot(dynamic[2][0][0].squeeze())
    # axes[1].set_title('centroid')
    # axes[1].set_ylabel("Value")
    # axes[1].set_xlabel("Index")
    # plt.tight_layout()
    # plt.savefig("components_all_in_one24.png")
    # plt.close()

    # 시각화
    plt.figure(figsize=(10, 4))
    plt.imshow(dynamic[:int(waveform.shape[-1]/sr*10)*3, :].T, aspect='auto', origin='lower', cmap='viridis')  # [channel x time]
    plt.colorbar(label='Value')
    plt.xlabel('Time Frame')
    plt.ylabel('Channel')
    plt.title('Feature Map (16 channels over 400 time steps)')
    plt.tight_layout()
    plt.savefig(f"components_all_in_one_{idx}.png")
    plt.close()

    print("dynamic : ", dynamic.shape)