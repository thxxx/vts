original_video_path = '/workspace/AVE_Dataset/AVE/'
res = 224
fps = 24
seconds = 8
import subprocess
import os
from tqdm import tqdm
from multiprocessing import Pool
import ffmpeg

import librosa
from einops import rearrange
import re
import numpy as np
import torch

processed_video_path = f'{original_video_path[:-1]}_processed/'
os.makedirs(processed_video_path, exist_ok=True)

from torchvision.io import read_video
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
import torch.nn.functional as F

print("\n\nvideo synchformer output latent 추출 후 저장 시작\n\n")

device = 'cuda'

video_files = os.listdir(processed_video_path)
print(f"\nTotal videos nun : {len(video_files)}\n")
synchformer_latents_dir = f'{processed_video_path[:-1]}_syncs/'
os.makedirs(synchformer_latents_dir, exist_ok=True)

print("\n\n███         \
░░░███      \
  ░░░███    \
    ░░░███   \
     ███░    \
   ███░      \
 ███░         Video LTX-video vae encoder output latent 추출 후 저장 시작\n\n")

ltx_latents_dir = f'{processed_video_path[:-1]}_latents/'
os.makedirs(ltx_latents_dir, exist_ok=True)
files = sorted(os.listdir(processed_video_path))  # 정렬 권장 (일관된 순서)
print(f"\nTotal videos num : {len(files)}\n")

ltxv_model_path = '/workspace/vts/ltxv-2b-0.9.8-distilled.safetensors'
vae = CausalVideoAutoencoder.from_pretrained(ltxv_model_path)
vae.to(device)
vae.eval()

batch_size = 2
buffer = []
file_buffer = []

for fp in tqdm(files):
    vid_path = os.path.join(processed_video_path, fp)
    video_frames, _, _ = read_video(vid_path)
    
    # 🔽 Rearrange and downsample to (T, C, 128, 128)
    video_frames = rearrange(video_frames, 't h w c -> t c h w')  # (T, C, H, W)
    video_frames = F.interpolate(video_frames, size=(128, 128), mode='bilinear', align_corners=False)

    # 영상 전처리
    video_frames = rearrange(video_frames, 't h w c -> 1 c t h w').tile(2, 1, 1, 1, 1)  # (1, C, T, H, W)
    video_frames = F.pad(video_frames, (0, 0, 0, 0, 0, 1))
    video_frames = video_frames.to(torch.float32) / 255.0 * 2 - 1
    video_frames = video_frames.to(device).to(torch.bfloat16)

    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                latent = vae.encode(video_frames).latent_dist.mode().float().cpu().numpy()  # (B, ...)
        np.save(os.path.join(ltx_latents_dir, fp[:-len("_224res_24fps.npy")] + "_128res_24fps.npy"), latent[0])
    except Exception as e:
        print(f"\n\n ERROR : {e} \n\n")
