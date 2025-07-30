# default
import time
import concurrent
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import torch
from einops import repeat
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer
from vocos import get_voco
from model.module_voice import AudioBoxModule
from torchode.interface import solve_ivp
import torchaudio
from einops import rearrange
from make_dynamics2 import get_dynamic_paper

# model functions
class Infer:
    def __init__(self, path: Path):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = AudioBoxModule.load_from_checkpoint(path).to(self.device)
        self.model.eval()
        self.voco = get_voco(self.model.voco_type).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.tokenizer.padding_side = "right"

        self.steps = 64
        self.alpha = 3.0

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def encode_text(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        batch_encoding = self.tokenizer(
            [text + self.tokenizer.eos_token for text in texts],
            add_special_tokens=False,
            return_tensors="pt",
            max_length=127,
            truncation="longest_first",
            padding="max_length",
        )
        phoneme = batch_encoding.input_ids.to(self.device)
        phoneme_mask = batch_encoding.attention_mask.to(self.device) > 0
        phoneme_emb = self.model.t5(
            input_ids=phoneme, attention_mask=phoneme_mask
        ).last_hidden_state

        return phoneme_emb, phoneme_mask

    # @torch.no_grad()
    # @torch.autocast(device_type="cuda", enabled=False)
    # def clap_rank(self, audios: Tensor, texts: list[str]) -> Tensor:
    #     audios = audios[:, : self.clap_audio_len].mean(dim=-1)
    #     audios = audios.float()
    #     text_embed = self.clap.get_text_embeddings(texts)
    #     audio_embed = self.clap.clap.audio_encoder(audios)[0]

    #     similarity = F.cosine_similarity(text_embed, audio_embed)
    #     args = torch.argsort(similarity, dim=0, descending=True)
    #     return args

    @torch.no_grad()
    @torch.autocast(device_type="cuda")
    def generate(
        self, texts: list[str], dur: float, cfg=3.0, voice_enc=None
    ) -> list[np.ndarray]:
        phoneme_emb, phoneme_mask = self.encode_text(texts)
        batch_size = phoneme_emb.shape[0]

        target_len = round(self.model.sampling_rate * dur)
        latent_len = self.voco.encode_length(target_len)
        audio_mask = torch.ones(
            batch_size, latent_len, dtype=torch.bool, device=self.device
        )
        audio_context = torch.zeros(
            batch_size, latent_len, self.voco.latent_dim, device=self.device
        )

        if latent_len < 192:
            audio_mask = F.pad(audio_mask, (0, 192 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 192 - latent_len))
            voice_enc = voice_enc[:, :192, :]
        elif 192 < latent_len < 384:
            audio_mask = F.pad(audio_mask, (0, 384 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 384 - latent_len))
            voice_enc = voice_enc[:, :384, :]
        elif 384 < latent_len < 768:
            audio_mask = F.pad(audio_mask, (0, 768 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 768 - latent_len))
            voice_enc = F.pad(voice_enc, (0, 0, 0, 768 - 400))
        elif 768 < latent_len < 1536:
            audio_mask = F.pad(audio_mask, (0, 1536 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 1536 - latent_len))
            voice_enc = F.pad(voice_enc, (0, 0, 0, 1536 - 400))

        def fn(t: Tensor, y: Tensor):
            out = self.model.audiobox.cfg(
                w=y,
                context=audio_context,
                times=t,
                alpha=cfg,
                mask=audio_mask,
                phoneme_emb=phoneme_emb,
                phoneme_mask=phoneme_mask,
                voice_enc=voice_enc
            )
            return out

        y0 = torch.randn_like(audio_context)
        t = torch.linspace(0, 1, self.steps, device=self.device)

        t = repeat(t, "n -> b n", b=batch_size)
        sol = solve_ivp(
            torch.compile(fn, dynamic=False),
            y0,
            t,
            method_class=self.model.method, #self.model.torchode_method_klass,
        )
        sampled_audio = sol.ys[-1]

        sample = self.voco.decode(sampled_audio)
        sample = sample[:, :target_len]

        sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
        # args = self.clap_rank(sample, texts)
        # sample = sample[args]
        # sample = sample[:cutoff]
        sample = sample.detach().cpu().numpy().astype(np.float32)

        return [audio for audio in sample]

    @torch.no_grad()
    @torch.autocast(device_type="cuda")
    def variation(
        self, audios: list[np.ndarray], texts: list[str], dur: float, corrupt: float, sr: list[int], voice_enc: Tensor
    ) -> list[np.ndarray]:
        phoneme_emb, phoneme_mask = self.encode_text(texts)
        batch_size = phoneme_emb.shape[0]

        audios = [audio / np.iinfo(audio.dtype).max for audio in audios]
        audio_tensor = torch.from_numpy(np.stack(audios, axis=0)).to(self.device)
        audio_tensor = audio_tensor.float()
        ##
        audio_tensor = audio_tensor.transpose(1, 2)
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sr[0], new_freq=self.voco.sampling_rate)
        audio_tensor = audio_tensor.transpose(1, 2)
        if audio_tensor.shape[2] == 1:
            audio_tensor = audio_tensor.repeat(1, 1, 2)
        elif audio_tensor.shape[2] > 2:
            audio_tensor = audio_tensor[:, :, :2]
        target_len = audio_tensor.shape[1]
        latent_len = self.voco.encode_length(target_len)
        audio_enc = self.voco.encode(audio_tensor)
        audio_mask = torch.ones(
            batch_size, latent_len, dtype=torch.bool, device=self.device
        )
        audio_context = torch.zeros(
            batch_size, latent_len, self.voco.latent_dim, device=self.device
        )

        if latent_len < 192:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 192 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 192 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 192 - latent_len))
            voice_enc = voice_enc[:, :192, :]
        elif 192 < latent_len < 384:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 384 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 384 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 384 - latent_len))
            voice_enc = voice_enc[:, :384, :]
        elif 384 < latent_len < 768:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 768 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 768 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 768 - latent_len))
            voice_enc = F.pad(voice_enc, (0, 0, 0, 768 - 400))
        elif 768 < latent_len < 1536:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 1536 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 1536 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 1536 - latent_len))
            voice_enc = F.pad(voice_enc, (0, 0, 0, 1536 - 400))

        sigma = 1e-3
        c = 1.0 - corrupt
        noised_enc = (audio_enc * c) + torch.randn_like(audio_enc) * (1 - (1 - sigma) * c)
        corrupt_tensor = torch.tensor(1 - corrupt).to(self.device)
        # print("corrupt_tensor : ", corrupt_tensor)

        def forward(t: Tensor, y: Tensor):
            # print("times : ", t)
            out = self.model.audiobox.cfg(
                w=y,
                context=audio_context,
                # times=t,
                times=t,
                alpha=self.alpha,
                mask=audio_mask,
                phoneme_emb=phoneme_emb,
                phoneme_mask=phoneme_mask,
                voice_enc=voice_enc
            )
            return out

        # t = torch.linspace(c, 1, 64, device=self.device)
        t = torch.linspace(0, corrupt, self.steps, device=self.device)
        # print("T : ", t)
        # print("T : ", 1-t)

        t = repeat(t, "n -> b n", b=batch_size)
        sol = solve_ivp(
            torch.compile(forward, dynamic=False),
            # forward,
            noised_enc,
            t+corrupt_tensor,
            method_class=self.model.method #.torchode_method_klass,
        )
        sampled_audio = sol.ys[-1]

        sample = self.voco.decode(sampled_audio)
        new_target_len = round(self.model.sampling_rate * dur)
        sample = sample[:, :new_target_len]

        sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
        sample = sample.detach().cpu().numpy().astype(np.float32)

        return [audio for audio in sample]

def remove_non_ascii(s):
    return ''.join(i for i in s if ord(i)<128)

def convert_to_int16(audio_array):
    """
    오디오 배열을 int16 형식으로 변환.
    """
    # float형 오디오 배열을 int16 범위로 스케일링
    audio_array = np.clip(audio_array, -1.0, 1.0)  # -1.0 ~ 1.0 범위로 제한
    audio_int16 = (audio_array * 32767).astype(np.int16)
    return audio_int16


#runpod handler
def handler(event):
    it=time.time()
    # handle input data
    input_data=event['input']
    texts = input_data['descriptions'][0]
    duration = input_data['duration']
    file_paths = input_data['original_download_urls']
    temperature = input_data['temperature']
    
    # generate
    if file_paths == None:
        output_audios = infer.generate([texts] * 5, duration)
    else:
        merged_audios, sr = process_audio_urls(file_paths)
        # print(merged_audios[0])
        # model inference
        output_audios = infer.variation(merged_audios, [texts] * 5, duration, temperature, sr)
    mt = time.time()
    
    #upload them
    output_urls = [None for _ in range(len(output_audios))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        upload_futures = {executor.submit(upload_audio, audio): idx for idx, audio in enumerate(output_audios)}
        for future in concurrent.futures.as_completed(upload_futures):
            idx = upload_futures[future]
            try:
                res = future.result()
                output_urls[idx] = res
            except Exception as error:
                print('error:', error)
                
    # delete audios saved in the folder
    delete_audio_files()
    
    #prepare the API response.
    response_data={
        "output_download_urls": output_urls
    }
    ft=time.time()
    print("until generation time: ", mt-it)
    print("total time: ",ft-it)
    return json.dumps(response_data)

import numpy as np
import torch

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

if __name__ == "__main__":
    # prepare model
    infer = Infer(Path('/home/khj6051/alignment-v3/audiobox/checkpoints/2025-04-10_22-39-06/0040000-0.6909.ckpt'))

    infers = [
        {
            "duration": 3.7,
            "caption": 'Rochekt launching and exploding.',
            "audio_path": "./voice_samples/piung.wav",
            "voice_enc": torch.from_numpy(np.load(
                './voice_samples/piung_devoiced_voice.npy',
            )).unsqueeze(dim=0).to('cuda') # 1, 400, 64
        },
        {
            "duration": 3,
            "caption": 'User interface sound with melodical alarm.',
            "audio_path": "./voice_samples/beepbeep.m4a",
            "voice_enc": torch.from_numpy(np.load(
                './voice_samples/beepbeep_voice.npy',
            )).unsqueeze(dim=0).to('cuda') # 1, 400, 64
        },
        {
            "duration": 4,
            "caption": 'Sci-fi cannon charging and shooting.',
            "audio_path": "./voice_samples/charging.m4a",
            "voice_enc": torch.from_numpy(np.load(
                './voice_samples/charging_voice.npy',
            )).unsqueeze(dim=0).to('cuda') # 1, 400, 64
        },
    ]
    
    def noise_audio(latent, corrupt=0.6):
        c = 1.0 - corrupt
        noised_enc = (latent * c) + torch.randn_like(latent) * (1 - (1 - 1e-4) * c)
        return noised_enc

    for i in range(3):
        d = infers[i]
        duration = d['duration']
        caption = d['caption'] + " & loud loudness & rich pitch range"
        print("generate : ", caption)
        voice_enc = d['voice_enc']
        audio_path = d['audio_path']
        
        waveform, sr = torchaudio.load(audio_path)
        SAMPLE_RATE = 44100
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
        # print(' waveform : ', waveform.shape)
        waveform_mono = waveform.mean(dim=0, keepdim=True).to('cuda')
        dynamic_context = get_dynamic_paper(waveform_mono, sr=SAMPLE_RATE, device='cuda', median_filter_size=4)
        dynamic_context = rearrange(dynamic_context, 's t -> t s').unsqueeze(dim=0).to('cuda')

        # output_audios = infer.generate([
        #     caption,
        # ], duration, 3.0, voice_enc=dynamic_context)
        print('waveform ', waveform.shape)
        merged_audios = convert_to_int16(np.array(rearrange(waveform[0], 't -> () t ()')))
        print('merged_audios ', merged_audios.shape)
        output_audios = infer.variation(
            merged_audios, 
            [caption],
            duration,
            0.7,
            [sr],
            voice_enc=dynamic_context
        )
        
        oa = torch.tensor(output_audios)
        oa = rearrange(oa, "b n c -> b c n")

        # os.path.mkdir("./outputs", exist_ok=True)
        for idx, audio in enumerate(oa):
            fp = f'{caption}_voice'
            torchaudio.save(f'./outputs/{fp}_{i}.wav', audio, sample_rate=44100)


