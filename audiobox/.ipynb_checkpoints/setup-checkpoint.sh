apt-get update
apt-get install -y ffmpeg
pip install -r requirements.txt
pip install git+https://github.com/descriptinc/audiotools
pip install --ignore-installed mlflow
pip install moviepy==1.0.3
pip uninstall -y torchaudio
pip install torchaudio --no-cache-dir
pip uninstall -y torchvision
pip install torchvision --no-cache-dir
pip install ffmpeg-python
pip install numpy==2.0.0