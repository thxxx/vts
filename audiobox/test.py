# import numpy as np
# import pandas as pd
# import os
# import re
# from utils.utils import make_voice_cond

from model.module_voice import AudioBoxModule
from safetensors.torch import save_file

path = "/home/khj6051/alignment-v3/audiobox/checkpoints/2025-04-13_05-51-18/0055000-0.6663.ckpt"
model = AudioBoxModule.load_from_checkpoint(path)
save_file(model.audiobox.state_dict(), "audiobox_voice_0415.safetensors")

# import torch
# import torchaudio
# import matplotlib.pyplot as plt
# from pathlib import Path

# def visualize_and_save(audio_path: str, output_path: str = "voice_cond.png"):
#     # 1. Load audio
#     waveform, sr = torchaudio.load(audio_path)
#     SAMPLE_RATE = 44100
    
#     if sr != SAMPLE_RATE:
#         waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)

#     waveform = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform  # (1, C, T)

#     # 2. Generate voice condition tensor
#     voice_cond = make_voice_cond(waveform)  # shape: (B=1, T, D)
#     print(voice_cond.shape)
#     voice_cond_np = voice_cond[0].cpu().detach().numpy().T  # (D, T)
#     print(voice_cond_np.shape)

#     # 3. Plot
#     plt.figure(figsize=(12, 5))
#     plt.imshow(voice_cond_np, aspect='auto', origin='lower', interpolation='nearest')
#     plt.title("Voice Conditioning Feature")
#     plt.xlabel("Time")
#     plt.ylabel("Feature Dimension")
#     plt.colorbar(label='Normalized Value')

#     # 4. Save
#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

#     print(f"Saved visualization to {output_path}")

# audio_path = './voice_samples/beepbeep.m4a'
# visualize_and_save(audio_path)

# # df = pd.read_csv('total_0407.csv')
# # print(len(df))
# # print(len(df[df['duration']>60]))
# # print(len(df[df['duration']<50]))
# # print(len(df[df['duration']<36.8]))
# # print(len(df[df['duration']<18.6]))
# # print(len(df[df['duration']<0.5]))
# # df = df[df['duration']>0.4][df['duration']<18.605]
# # df.to_csv('total_voice_0409.csv')

# # df = pd.read_csv('total_0406.csv')
# # print(len(df))
# # print(df.keys())
# # print(df.head(3))
# # # print(len(df[df['rolloff_mean']>7500])) # 314만
# # # print(len(df[df['low_ratio_800']>0.2])) # 340만
# # # print(len(df[df['low_ratio_800']>0.2][df['rolloff_mean']>7500])) # 200만
# # # print(len(df[df['low_ratio_500']>0.5][df['rolloff_mean']<4000])) # 200만
# # # print(len(df[df['loudness_category'] == 'very soft'])) # 3만
# # # print(len(df[df['loudness_category'] == 'soft'])) # 80만
# # # print(len(df[df['loudness_category'] == 'loud'])) # 280만
# # # print(len(df[df['loudness_category'] == 'very loud'])) # 45만
# # # print(len(df[df['loudness'] > -18])) # 100만
# # # print(len(df[df['loudness'] > -19])) # 134만
# # print(len(df[df['loudness'] > -17])) # 84만
# # print(len(df[df['loudness'] < -32][df['loudness'] > -70])) # 83만
# # df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.4', 'Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1'])

# # def is_loud(x):
# #     if x>-17:
# #         return 'loud'
# #     if x>-32 and x>-70:
# #         return 'soft'
# #     return None

# # df['loudness_binary'] = df['loudness'].apply(lambda x: is_loud(x))

# # def pitchrange(data):
# #     if data['rolloff_mean']<5500 and data['low_ratio_500']>0.6 and data['duration']>0.3:
# #         return 'low'
# #     if data['rolloff_mean']>7500 and data['low_ratio_500']>0.25 and data['duration']>0.3:
# #         return 'rich'
# #     if data['rolloff_mean']>9000 and data['low_ratio_500']<0.15 and data['duration']>0.3:
# #         return 'high'
# #     return 'mid'

# # df['pitch_range'] = df.apply(lambda x: pitchrange(x), axis=1)

# # print("\n\n")
# # print(len(df[df['pitch_range'] == 'low']))
# # print(len(df[df['pitch_range'] == 'high']))
# # print(len(df[df['pitch_range'] == 'rich']))

# # df.to_csv('total_0407.csv')
# # df.head(3).to_csv('head.csv')

# # def make_caption(data):
# #     caption = data['fined_caption']

# # df['new_caption'] = df.apply(lambda x: make_caption(x))

# # df=pd.read_csv("version1_0406_exist.csv")
# # df2=pd.read_csv("version2_0406_exist.csv")
# # mdf = pd.concat([df, df2])
# # mdf.to_csv('total_0406.csv')
# # print(mdf.iloc[0])
# # print(mdf.iloc[-1])
# # print(len(mdf))

# # df = pd.read_csv('total_0406.csv')
# # print(len(df[df['fined_caption'].str.contains("unspecified content")]))
# # print(len(df[df['fined_caption'].str.contains("audio content")]))
# # print(df[df['fined_caption'].str.contains("unspecified content")])
# # print(df[df['fined_caption'].str.contains("audio content")])
# # df['fined_caption'] = df['fined_caption'].apply(lambda x: re.sub(', unspecified content.', '', x))
# # df = df[~df['fined_caption'].str.contains("unspecified content")]
# # df = df[~df['fined_caption'].str.contains("audio content")]
# # print(len(df))
# # df.to_csv('total_0406.csv')

# # # df['audio_path'] = df['audio_path'].apply(lambda x: '/home/khj6051/data/' + x)
# # # print(df.iloc[0])
# # # df.to_csv('version2_0406.csv')
# # print(df.iloc[0])
# # print(len(df[pd.isna(df['duration'])]))

# # import pandas as pd
# # import os

# # def convert_to_npy_path(path):
# #     base = os.path.splitext(path)[0]  # 확장자 제거
# #     return base + '.0000.npy'

# # # .0000.npy 경로로 변환
# # df['npy_path'] = df['audio_path'].apply(convert_to_npy_path)

# # # 존재 여부 확인
# # df['npy_exists'] = df['npy_path'].apply(os.path.exists)

# # # 존재하지 않는 파일 수
# # missing_count = (~df['npy_exists']).sum()

# # print(f"존재하지 않는 .0000.npy 파일 수: {missing_count}")
# # # 3. 존재하는 것만 필터링
# # df = df[df['npy_exists']].copy()
# # df = df.drop(columns=['npy_path', 'npy_exists'])
# # print(len(df))
# # df.to_csv('version1_0406_exist.csv')

# # # latent = np.load('./latent_pond_pond_5_01-faraming-sound-effect-257420503_nw_prev.0001.npy')
# # # print(latent.shape)