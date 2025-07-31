original_video_path = '/workspace/AVE_Dataset/AVE/'
res = 224
fps = 24
seconds = 8

import subprocess
import os
from tqdm import tqdm
from multiprocessing import Pool

processed_video_path = f'{original_video_path[:-1]}_processed/'
os.makedirs(processed_video_path, exist_ok=True)

files = os.listdir(original_video_path)
print("Total video nums : ", len(files))


def process_video(input_path: str, output_path: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", f"{seconds}",
        "-vf", (
            "crop='min(in_w\\,in_h)':'min(in_w\\,in_h)':"
            "(in_w - min(in_w\\,in_h))/2:"
            "(in_h - min(in_w\\,in_h))/2,"
            f"scale={res}:{res},"
            f"fps={fps}"
        ),
        output_path
    ]

    import subprocess
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_video_wrapper(args):
    input_path, output_path = args
    try:
        process_video(input_path, output_path)
    except Exception as e:
        print("ERror : ", e)

def process_all_videos(files, dir_path, pro_path):
    args_list = [
        (os.path.join(dir_path, fp), os.path.join(pro_path, fp[:-4] + f"_{res}res_{fps}fps.mp4"))
        for fp in files
    ]

    with Pool(processes=4) as pool:
        list(tqdm(pool.imap_unordered(process_video_wrapper, args_list), total=len(args_list)))

print(f"Video processing 시작 : {res} resolution, {fps} fps, first {seconds} seconds")
# process_all_videos(files, original_video_path, processed_video_path)

import ffmpeg

audio_path = f'{original_video_path[:-1]}_audio/'
os.makedirs(audio_path, exist_ok=True)

def extract_audio(input_video_path, output_audio_path):
    (
        ffmpeg
        .input(input_video_path)
        .output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='44100')
        .run(quiet=True)
    )

def process_audio_wrapper(args):
    input_path, output_path = args
    try:
        extract_audio(input_path, output_path)
    except Exception as e:
        print("Error : ", e)

def process_all_audios(files, dir_path, pro_path):
    args_list = [
        (os.path.join(dir_path, fp), os.path.join(pro_path, fp[:-4] + ".wav"))
        for fp in files
    ]

    with Pool(processes=4) as pool:
        list(tqdm(pool.imap_unordered(process_audio_wrapper, args_list), total=len(args_list)))

print('\n\nAudio 추출 후 저장 시작\n\n')
# process_all_audios(files, original_video_path, audio_path)

# extract stable audio vae latent from audio and save, with same file name nut .npy, and directory + _latent
from vocos import get_voco

voco = get_voco("oobleck")
voco.to('cuda')

import librosa
from einops import rearrange
import re
import numpy as np
import torch

audio_latents_dir = f'{audio_path[:-1]}_latent/'
audio_files = os.listdir(audio_path)
os.makedirs(audio_latents_dir, exist_ok=True)

print("\n\nStableaudio codec latent 추출 후 저장 시작\n\n")
def extract_audio_latent():
    for af in tqdm(audio_files):
        audio, sr = librosa.load(audio_path + af, sr=voco.sampling_rate, mono=False)
        audio = torch.tensor(audio)
        if len(audio.shape) == 1:
            audio = torch.stack([audio, audio])
        audio = rearrange(audio, 'c n -> n c')
        
        lt = voco.encode(audio.unsqueeze(dim=0).to('cuda')).squeeze()[:int(21.5*seconds), :].cpu().float().numpy()
        save_path = re.sub("\.wav", ".npy", audio_latents_dir+af)
        np.save(save_path, lt)

# extract_audio_latent()

del voco
torch.cuda.empty_cache()

from synchformer import Synchformer
from einops import rearrange
import torch
import os
import numpy as np
import torch
from torchvision.io import read_video
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F

print("\n\nvideo synchformer output latent 추출 후 저장 시작\n\n")

device = 'cuda'

video_files = os.listdir(processed_video_path)
print(f"\nTotal videos nun : {len(video_files)}\n")
synchformer_latents_dir = f'{processed_video_path[:-1]}_syncs/'
os.makedirs(synchformer_latents_dir, exist_ok=True)

synchformer = Synchformer()
synchformer.eval()
synchformer.to('cuda')

ckpt_path = '/workspace/vts/synchformer_state_dict.pth'
synchformer.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location='cpu'))


@torch.inference_mode()
def encode_video_with_sync(x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    # partition the video
    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size:i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

    outputs = []
    if batch_size < 0:
        batch_size = b
    x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
    for i in range(0, b * num_segments, batch_size):
        outputs.append(synchformer(x[i:i + batch_size]))
    x = torch.cat(outputs, dim=0)
    x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
    return x

# for video_file in tqdm(video_files):
#     video_path = processed_video_path + video_file
#     video_frames, _, _ = read_video(video_path)

#     # 영상 전처리
#     video_frames = rearrange(video_frames, 't h w c -> 1 t c h w')  # (1, C, T, H, W)
#     video_frames = video_frames.to(torch.float32).to('cuda') / 255.0

#     try:
#         with torch.no_grad():
#             out = encode_video_with_sync(video_frames).float().cpu().numpy()
#         np.save(os.path.join(synchformer_latents_dir, video_file[:-4] + ".npy"), out[0])
#     except Exception as e:
#         print(f"\n\n ERROR : {e} \n\n")

video_batch_size = 16  # 한 번에 묶을 비디오 개수
for i in tqdm(range(0, len(video_files), video_batch_size)):
    batch_files = video_files[i:i+video_batch_size]
    vids = []

    # 1) 각 비디오 읽어서 (1, T, C, H, W) 텐서로 변환
    for vf in batch_files:
        path = os.path.join(processed_video_path, vf)
        frames, _, _ = read_video(path, pts_unit='sec')  # (T, H, W, C)
        v = rearrange(frames, 't h w c -> 1 t c h w')     # (1, T, C, H, W)
        v = v.to(device).float() / 255.0
        vids.append(v)

    # 2) 패딩: T_max에 맞춰 0으로 채우기
    T_max = max(v.shape[1] for v in vids)
    vids_padded = []
    for v in vids:
        B, T, C, H, W = v.shape
        if T < T_max:
            pad = torch.zeros((1, T_max - T, C, H, W), device=device, dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        vids_padded.append(v)

    # 3) 배치 생성 (B, T_max, C, H, W)
    batch = torch.cat(vids_padded, dim=0)
    print("batch : ", batch.shape)

    # 4) 배치 인퍼런스
    try:
        latents = encode_video_with_sync(batch)  # (B, L, D) on CPU
    except Exception as e:
        print(f"[ERROR] batch {i}-{i+len(batch_files)} : {e}")
        continue

    # 5) 개별 저장
    for vf, latent in zip(batch_files, latents):
        out_path = os.path.join(synchformer_latents_dir, vf[:-4] + '.npy')
        np.save(out_path, latent.cpu().numpy())
    
del synchformer
torch.cuda.empty_cache()

print("\n\n███         \
░░░███      \
  ░░░███    \
    ░░░███   \
     ███░    \
   ███░      \
 ███░         Video LTX-video vae encoder output latent 추출 후 저장 시작\n\n")

# ltx_latents_dir = f'{processed_video_path[:-1]}_latents/'
# os.makedirs(ltx_latents_dir, exist_ok=True)
# files = sorted(os.listdir(processed_video_path))  # 정렬 권장 (일관된 순서)
# print(f"\nTotal videos num : {len(files)}\n")

# ltxv_model_path = '/workspace/vts/ltxv-2b-0.9.8-distilled.safetensors'
# vae = CausalVideoAutoencoder.from_pretrained(ltxv_model_path)
# vae.to(device)
# vae.eval()

# batch_size = 2
# buffer = []
# file_buffer = []

# for fp in tqdm(files):
#     vid_path = os.path.join(processed_video_path, fp)
#     video_frames, _, _ = read_video(vid_path)
    
#     # 🔽 Rearrange and downsample to (T, C, 128, 128)
#     video_frames = rearrange(video_frames, 't h w c -> t c h w')  # (T, C, H, W)
#     video_frames = F.interpolate(video_frames, size=(128, 128), mode='bilinear', align_corners=False)

#     # 영상 전처리
#     video_frames = rearrange(video_frames, 't h w c -> 1 c t h w').tile(2, 1, 1, 1, 1)  # (1, C, T, H, W)
#     video_frames = F.pad(video_frames, (0, 0, 0, 0, 0, 1))
#     video_frames = video_frames.to(torch.float32) / 255.0 * 2 - 1
#     video_frames = video_frames.to(device).to(torch.bfloat16)

#     try:
#         with torch.no_grad():
#             with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#                 latent = vae.encode(video_frames).latent_dist.mode().float().cpu().numpy()  # (B, ...)
#         np.save(os.path.join(ltx_latents_dir, fp[:-len("_224res_24fps.npy")] + "_128res_24fps.npy"), latent[0])
#     except Exception as e:
#         print(f"\n\n ERROR : {e} \n\n")