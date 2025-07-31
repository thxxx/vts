apt-get update
apt-get install -y ffmpeg
pip install -e .
pip install av torchvision imageio imageio[ffmpeg]

git config --global user.email zxcv05999@naver.com
git config --global user.name thxxx

# wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-distilled-04-25.safetensors?download=true