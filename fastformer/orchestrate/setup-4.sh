
## Install python 3.8 kernel
source ~/.zshrc
conda update -y -n base -c defaults conda
conda create -y -n py38 python=3.8 anaconda
conda activate py38

pip install --upgrade pip
conda install -y -c anaconda ipykernel ipython
python -m ipykernel install --user --name=py38
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html > /dev/null
pip install -U numpy pandas scikit-learn tqdm pillow opencv-python bidict matplotlib decord katna gpustat  open_clip_torch > /dev/null
pip install --ignore-installed -U 'threadpoolctl>=3' > /dev/null
pip install transformers tokenizers datasets accelerate > /dev/null
pip install moviepy imageio-ffmpeg > /dev/null
pip install setuptools==59.5.0 > /dev/null
pip install -U mmh3 pandarallel zstandard > /dev/null
pip install -U tqdm wandb sacrebleu sacremoses sentencepiece dataclasses dataclasses-json dill nltk more-itertools scipy> /dev/null
pip install -U gpustat nlpaug pyarrow snakeviz torch-optimizer attrdict  pytorch-fast-transformers pytorch-ranger> /dev/null
pip install -U opencv-python transformers datasets tokenizers  accelerate evaluate lm-eval appdirs fsspec joblib scikit-learn opencv-python einops open_clip_torch > /dev/null
pip install --pre timm > /dev/null
pip install -U mmh3 pandarallel zstandard apache_beam mwparserfromhell py7zr> /dev/null
pip install --upgrade nvitop nvidia-ml-py > /dev/null

pip install --ignore-installed -U tbb
pip uninstall tensorflow
pip uninstall tensorflow-gpu
pip install tensorflow-gpu

pip install deepface -—no-deps
pip install retina-face
pip install lru-dict
pip install more-itertools


conda install -y libgcc
pip install -U detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install deepface
pip install retina-face

## Installed python 3.8 kernel