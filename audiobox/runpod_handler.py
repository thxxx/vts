# default
import runpod
import time
import os
import glob
import uuid
import concurrent
import concurrent.futures
import requests
import json
import datetime
import soundfile as sf
import io
# optional
from pathlib import Path
import numpy as np
import torch
from einops import repeat
from msclap import CLAP
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer
from vocos import get_voco
from model.module import AudioBoxModule
from torchode.interface import solve_ivp
import torchaudio


#initiate supabase
from supabase import create_client, Client
SUPABASE_URL="https://hpxjdveijpuehyuykkos.supabase.co"
SUPABASE_ANON="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhweGpkdmVpanB1ZWh5dXlra29zIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwNDUyNzgxMSwiZXhwIjoyMDIwMTAzODExfQ.zo_ddufJU0SGR9ijLhzPFGZGJ6a46x7oByroj_qTkY8"

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

        self.clap = CLAP(version="2023", use_cuda=torch.cuda.is_available())

        self.steps = 64
        self.alpha = 3.0
        self.clap_audio_len = 7 * self.model.sampling_rate

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

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def clap_rank(self, audios: Tensor, texts: list[str]) -> Tensor:
        audios = audios[:, : self.clap_audio_len].mean(dim=-1)
        audios = audios.float()
        text_embed = self.clap.get_text_embeddings(texts)
        audio_embed = self.clap.clap.audio_encoder(audios)[0]

        similarity = F.cosine_similarity(text_embed, audio_embed)
        args = torch.argsort(similarity, dim=0, descending=True)
        return args

    @torch.no_grad()
    @torch.autocast(device_type="cuda")
    def generate(
        self, texts: list[str], dur: float, cutoff: int = 5
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
        elif 192 < latent_len < 384:
            audio_mask = F.pad(audio_mask, (0, 384 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 384 - latent_len))
        elif 384 < latent_len < 768:
            audio_mask = F.pad(audio_mask, (0, 768 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 768 - latent_len))
        elif 768 < latent_len < 1536:
            audio_mask = F.pad(audio_mask, (0, 1536 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 1536 - latent_len))

        def fn(t: Tensor, y: Tensor):
            out = self.model.audiobox.cfg(
                w=y,
                context=audio_context,
                times=t,
                alpha=self.alpha,
                mask=audio_mask,
                phoneme_emb=phoneme_emb,
                phoneme_mask=phoneme_mask,
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
        args = self.clap_rank(sample, texts)
        sample = sample[args]
        sample = sample[:cutoff]
        sample = sample.detach().cpu().numpy().astype(np.float32)

        return [audio for audio in sample]

    @torch.no_grad()
    @torch.autocast(device_type="cuda")
    def variation(
        self, audios: list[np.ndarray], texts: list[str], dur: float, corrupt: float, sr: list[int],
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
        elif 192 < latent_len < 384:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 384 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 384 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 384 - latent_len))
        elif 384 < latent_len < 768:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 768 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 768 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 768 - latent_len))
        elif 768 < latent_len < 1536:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 1536 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 1536 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 1536 - latent_len))

        sigma = 1e-3
        c = 1.0 - corrupt
        noised_enc = (audio_enc * c) + torch.randn_like(audio_enc) * (1 - (1 - sigma) * c)
        corrupt_tensor = torch.tensor(1 - corrupt).to(self.device)

        def forward(t: Tensor, y: Tensor):
            out = self.model.audiobox.cfg(
                w=y,
                context=audio_context,
                times=t + corrupt_tensor,
                alpha=self.alpha,
                mask=audio_mask,
                phoneme_emb=phoneme_emb,
                phoneme_mask=phoneme_mask,
            )
            return out

        t = torch.linspace(0, corrupt, self.steps, device=self.device)

        t = repeat(t, "n -> b n", b=batch_size)
        sol = solve_ivp(
            torch.compile(forward, dynamic=False),
            noised_enc,
            t,
            method_class=self.model.method #.torchode_method_klass,
        )
        sampled_audio = sol.ys[-1]

        sample = self.voco.decode(sampled_audio)
        new_target_len = round(self.model.sampling_rate * dur)
        sample = sample[:, :new_target_len]

        sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
        sample = sample.detach().cpu().numpy().astype(np.float32)

        return [audio for audio in sample]

#functions
def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open file in binary write mode and save the content to the file
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def remove_non_ascii(s):
    return ''.join(i for i in s if ord(i)<128)

def upload_to_supabase(local_filepath, bucket_name, bucket_path, content_type):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON)
    #upload
    st = time.time()
    with open(local_filepath, 'rb') as f:
        supabase.storage.from_(bucket_name).upload(file=f, path=bucket_path, file_options={"content-type": content_type})
    print("upload supabase time: ", time.time() - st)
    #get download URL
    res = supabase.storage.from_(bucket_name).get_public_url(bucket_path)
    return res

def upload_audio(audio_array):
    local_audio_id =str(uuid.uuid4())
    local_filename = "{:s}.wav".format(local_audio_id)
    local_savepath = "audiobox/temp_audio_folder/{:s}".format(local_filename)
    sf.write(local_savepath, audio_array, 44100)
    
    now=str(datetime.datetime.now().date())
    audio_id=str(uuid.uuid4())
    filename="{:s}.wav".format(audio_id)
    
    try:
        res = upload_to_supabase(local_savepath, "v2", remove_non_ascii("{:s}/{:s}".format(now, filename)), "audio/wav")
        if res is None:
            raise ValueError("in function upload audio: Upload returned None")
        return res
    except Exception as error:
        print("in function upload audio: ", error)
        return None

# def download_and_save_audio(input_url):
#     try:
#         temp_filename =os.path.join("audiobox/temp_audio_folder", f'{uuid.uuid4()}.wav')
#         response = requests.get(input_url)
#         if response.status_code == 200:
#             with open(temp_filename, 'wb') as f:
#                 f.write(response.content)
#                 return temp_filename
#         else:
#             print(f'failed to download {input_url}. status code: {response.staues_code}')
#             return None
#     except Exception as e:
#         print(f'error downloading {input_url}: {e}')
#         return None

def delete_audio_files():
    files = glob.glob(os.path.join("audiobox/temp_audio_folder/", '*'))
    for file in files:
        try:
            os.remove(file)  # 파일 삭제
            # print(f"Deleted: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

def download_audio_as_array(url):
    """
    주어진 URL에서 오디오 파일을 다운로드하여 넘파이 배열과 샘플링 레이트를 반환하는 함수.
    """
    # URL에서 바이너리 데이터 가져오기
    response = requests.get(url)
    response.raise_for_status()  # 요청 실패 시 예외 발생

    # BytesIO로 감싸서 soundfile로 읽기
    data, samplerate = sf.read(io.BytesIO(response.content), always_2d=True)
    return data, samplerate

def convert_to_int16(audio_array):
    """
    오디오 배열을 int16 형식으로 변환.
    """
    # float형 오디오 배열을 int16 범위로 스케일링
    audio_array = np.clip(audio_array, -1.0, 1.0)  # -1.0 ~ 1.0 범위로 제한
    audio_int16 = (audio_array * 32767).astype(np.int16)
    return audio_int16

def process_audio_urls(url_list):
    """
    URL 리스트를 받아 int16 타입의 오디오 배열로 변환.
    반환: [n, length, channels] 형태의 3D 배열
    """
    audio_arrays = []
    samplerates = []
    max_length = 0

    # 각 URL에서 오디오 다운로드 및 변환
    for url in url_list:
        data, samplerate = download_audio_as_array(url)

        samplerates.append(samplerate)  
        data_int16 = convert_to_int16(data)
        
        # 길이 업데이트
        max_length = max(max_length, data_int16.shape[0])
        audio_arrays.append(data_int16)

    # 모든 오디오 데이터를 동일한 길이로 패딩
    padded_audios = []
    for audio in audio_arrays:
        padding = ((0, max_length - audio.shape[0]), (0, 0))  # 시간축 패딩 추가
        padded_audio = np.pad(audio, padding, mode='constant', constant_values=0)
        padded_audios.append(padded_audio)

    # [n, length, channels] 형태의 3D 배열로 병합
    result = np.stack(padded_audios, axis=0)
    return result, samplerates

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

# prepare model
infer = Infer(Path("audiobox/new-stage-2.ckpt"))

# testing in local
# if __name__ == "__main__":
#     event = {
#         "input": {
#             "descriptions": ["angry dog barking"]*5,
#             "duration": 1.0,
#             "original_download_urls": ["https://hpxjdveijpuehyuykkos.supabase.co/storage/v1/object/voices/2024-12-08/90b30e38-86d0-4e85-b205-107cace5421a.wav"] * 5,
#             "temperature": 0.8 #작으면 덜바뀜
#         }
#     }
    # event = {
    #     "input": {
    #         "descriptions": ["cat meow","cat meow","cat meow","cat meow","cat meow"],
    #         "duration": 1.0,
    #         "original_download_urls":None,
    #         "temperature": None
    #     }
    # }
    # for i in range (10):
    #     result = handler(event)
    # print('final handler response', result)

# real
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })