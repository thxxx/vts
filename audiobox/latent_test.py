from vocos import get_voco
import torchaudio
import librosa
import torch
from einops import rearrange, repeat
import numpy as np
from transformers import AutoTokenizer
from torch.nn import functional as F

voco = get_voco('oobleck').to('cuda')

audio, sr = torchaudio.load('./voice_samples/piung_devoiced.wav')
audio = rearrange(audio, 's t -> t s')
audio = repeat(audio, 't 1 -> t 2')
audio = audio.unsqueeze(dim=0).to('cuda')
print('audio : ', audio.shape, sr)

latent1 = voco.encode(audio)
print("WEFgefw ", latent1.shape)
latent_len = latent1.shape[1]
latent = np.array(latent1.squeeze().cpu())
latent = np.pad(latent, ((0, 400 - latent_len), (0, 0)))
np.save('./voice_samples/piung_devoiced_voice.npy', latent)
print('latent1 : ', latent1.shape, sr)

audio, sr = torchaudio.load('./voice_samples/piung.wav')
# audio, sr = torchaudio.load('./voice_samples/ss_science_fiction_page_1_512123-Energy_Weapon_Charging_01_-SCIWeap-energy-charging-weapon.mp3')
audio = rearrange(audio, 's t -> t s')
audio = audio.unsqueeze(dim=0).to('cuda')
print('audio : ', audio.shape, sr)

latent = voco.encode(audio)

def noise_audio(latent, corrupt=0.6):
    c = 1.0 - corrupt
    noised_enc = (latent * c) + torch.randn_like(latent) * (1 - (1 - 1e-4) * c)
    return noised_enc

# for i in [2, 4, 6, 8, 10, 12]:
#     voice_cond = latent.permute(0, 2, 1)            # (B, 64, 48)
#     voice_cond = F.avg_pool1d(voice_cond, kernel_size=i, stride=i)  # (B, 64, 24)
#     voice_cond = F.interpolate(voice_cond, scale_factor=i, mode='nearest')  # (B, 64, 48)
#     voice_cond = voice_cond.permute(0, 2, 1).squeeze()

#     voice_cond1 = noise_audio(voice_cond, 0.0).unsqueeze(dim=0)
#     audio = voco.decode(voice_cond1).squeeze()
#     audio = rearrange(audio, 't s -> s t')
#     torchaudio.save(f'sste00_tempoblur_{i}.wav', audio.cpu(), sample_rate=44100)

# voice_cond2 = noise_audio(voice_cond, 0.0).unsqueeze(dim=0)
# audio = voco.decode(voice_cond2).squeeze()
# audio = rearrange(audio, 't s -> s t')
# torchaudio.save('sste00_novoice2.wav', audio.cpu(), sample_rate=44100)

# voice_cond3 = noise_audio(voice_cond, 0.0).unsqueeze(dim=0)
# audio = voco.decode(voice_cond3).squeeze()
# audio = rearrange(audio, 't s -> s t')
# torchaudio.save('sste00_novoice3.wav', audio.cpu(), sample_rate=44100)

# latent = latent1[:, :latent2.shape[1], :] + latent2*0.1
# audio = voco.decode(latent).squeeze()
# audio = rearrange(audio, 't s -> s t')
# torchaudio.save('sste22.wav', audio.cpu(), sample_rate=44100)

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# tokenizer.padding_side = "right"

# desc = 'User interface sound with melodical alarm.'
# # desc = 'Scifi-cannon charging and shooting.'
# batch_encoding = tokenizer(
#     [desc + tokenizer.eos_token],
#     return_tensors="pt",
#     truncation="longest_first",
#     padding="max_length",
#     max_length=128,
#     add_special_tokens=False,
# )
# input_ids = batch_encoding.input_ids
# attention_mask = batch_encoding.attention_mask > 0
# print('input_ids ', input_ids.shape)
# print('attention_mask ', attention_mask.shape)

# audio, sr = torchaudio.load('./voice_samples/beepbeep.m4a')
# audio = rearrange(audio, 's t -> t s')
# audio = repeat(audio, 't 1 -> t 2')
# audio = audio.unsqueeze(dim=0).to('cuda')
# print('audio : ', audio.shape, sr)

# latent = voco.encode(audio)
# latent_len = latent.shape[1]
# latent = np.array(latent.squeeze().cpu())
# latent = np.pad(latent, ((0, 400 - latent_len), (0, 0)))
# print('latent : ', latent.shape)

# np.save('./voice_samples/beepbeep_voice.npy', latent)
# np.save('./voice_samples/beepbeep_token.npy', input_ids.squeeze())
# np.save('./voice_samples/beepbeep_token_mask.npy', attention_mask.squeeze())
