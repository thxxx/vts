# 모델 불러오기 : synchformer, ltx-video-vae, vae, t5, audiobox

from vocos import get_voco
import librosa
from einops import rearrange
import re
import numpy as np
import torch
import subprocess
import os
from tqdm import tqdm
from multiprocessing import Pool
from synchformer import Synchformer
from einops import rearrange
import torch
from torchvision.io import read_video
import torch.nn.functional as F


voco = get_voco("oobleck")
voco.to('cuda')

synchformer = Synchformer()
synchformer.eval()
synchformer.to('cuda')

ckpt_path = '/workspace/vts/synchformer_state_dict.pth'
synchformer.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location='cpu'))

