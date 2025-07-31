import os
import numpy as np
import torch
from torchvision.io import read_video
from einops import rearrange
from tqdm import tqdm
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from torch.cuda.amp import autocast
import torch.nn.functional as F

device = 'cuda'
dir_path = '/workspace/AVE_Dataset/AVE_processed/'
pro_path = '/workspace/AVE_Dataset/AVE_processed_latents/'
files = sorted(os.listdir(dir_path))  # 정렬 권장 (일관된 순서)
os.makedirs(pro_path, exist_ok=True)
print("길이 : ", len(files))

ltxv_model_path = '/workspace/vts/ltxv-2b-0.9.8-distilled.safetensors'
vae = CausalVideoAutoencoder.from_pretrained(ltxv_model_path)
vae.to(device)
vae.eval()

batch_size = 2
buffer = []
file_buffer = []

for fp in tqdm(files):
    vid_path = os.path.join(dir_path, fp)
    video_frames, _, _ = read_video(vid_path)
    
    # 🔽 Rearrange and downsample to (T, C, 128, 128)
    video_frames = rearrange(video_frames, 't h w c -> t c h w')  # (T, C, H, W)
    video_frames = F.interpolate(video_frames, size=(128, 128), mode='bilinear', align_corners=False)
    video_frames = rearrange(video_frames, 't c h w -> t h w c')

    # 영상 전처리
    video_frames = rearrange(video_frames, 't h w c -> 1 c t h w').tile(2, 1, 1, 1, 1)  # (1, C, T, H, W)
    video_frames = F.pad(video_frames, (0, 0, 0, 0, 0, 1))
    video_frames = video_frames.to(torch.float32) / 255.0 * 2 - 1
    video_frames = video_frames.to(device).to(torch.bfloat16)

    try:
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                latent = vae.encode(video_frames).latent_dist.mode().float().cpu().numpy()  # (B, ...)
        np.save(os.path.join(pro_path, fp[:-4] + ".npy"), latent[0])
    except Exception as e:
        print(f"\n\n ERROR : {e} \n\n")
