sudo apt-get update
sudo apt-get install -y ffmpeg
python3 -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/descriptinc/audiotools
python3 -m pip install --ignore-installed mlflow