apt-get update
apt-get install -y ffmpeg
python3 -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/descriptinc/audiotools
python3 -m pip install --ignore-installed mlflow
pip uninstall -y torchaudio
pip install torchaudio --no-cache-dir
pip uninstall -y torchvision
pip install torchvision --no-cache-dir
pip install ffmpeg-python
pip install numpy==2.0.0