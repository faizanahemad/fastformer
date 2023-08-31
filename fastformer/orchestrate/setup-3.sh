# https://serverfault.com/questions/72955/how-to-change-default-tmp-to-home-user-tmp
# P4d specifics
# Add `export PATH="$PATH:/home/$USER/.local/bin"` to your ~/.zshrc
# use --no-cache-dir --user for pip
# wget https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/nvidia-fabric-manager-510.39.01-1.x86_64.rpm
# sudo yum localinstall nvidia-fabric-manager-510.39.01-1.x86_64.rpm
source ~/.zshrc
sudo mkdir -p /home/ahemf/processed_datasets
mkdir -p $HOME/tmp
export TMPDIR=$HOME/tmp

pip install --cache-dir=$HOME/cache nvidia-cusolver-cu11==11.4.0.1 nvidia-curand-cu11==10.2.10.91  nvidia-cufft-cu11==10.9.0.58 nvidia-cublas-cu11==11.10.3.66 nvidia-cudnn-cu11==8.5.0.96 nvidia-cuda-cupti-cu11==11.7.101 nvidia-cuda-runtime-cu11==11.7.99 nvidia-nccl-cu11==2.14.3 nvidia-nvtx-cu11==11.7.91 triton==2.0.0 
pip install --cache-dir=$HOME/cache torch torchvision torchaudio

# yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
# --no-cache-dir --force-reinstall --ignore-installed

pip install --upgrade pip setuptools wheel
pip install --cache-dir=$HOME/cache -U numpy pandas scikit-learn tqdm wandb sacrebleu sacremoses sentencepiece dataclasses dataclasses-json dill nltk more-itertools scipy> /dev/null
pip install --cache-dir=$HOME/cache -U nlpaug pyarrow snakeviz torch-optimizer attrdict pytorch-ranger> /dev/null
pip install --cache-dir=$HOME/cache -U typing_extensions matplotlib py7z deepspeed opencv-python transformers datasets tokenizers  accelerate evaluate lm-eval appdirs fsspec joblib scikit-learn opencv-python einops open_clip_torch bidict> /dev/null
pip install --cache-dir=$HOME/cache --upgrade nvitop nvidia-ml-py regex pyparsing kiwisolver fonttools cycler contourpy sympy appdirs threadpoolctl> /dev/null
pip install --cache-dir=$HOME/cache -U mmh3 pandarallel zstandard apache_beam mwparserfromhell > /dev/null

pip install -U --cache-dir=$HOME/cache --force-reinstall --ignore-installed pynvml
pip uninstall -y pynvml
pip install -U --cache-dir=$HOME/cache --force-reinstall --ignore-installed nvidia-ml-py
pip install -U --cache-dir=$HOME/cache --force-reinstall --ignore-installed gpustat xformers bitsandbytes nvitop
pip install -U --cache-dir=$HOME/cache --force-reinstall --ignore-installed lmql guidance langchain
pip install --pre timm > /dev/null

# https://pytorch.org/get-started/previous-versions
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 > /dev/null
# --use-deprecated=legacy-resolver 
# https://github.com/pypa/pip/issues/9187
# --use-feature=fast-deps --use-deprecated=legacy-resolver
pip install jupyterlab > /dev/null
python -c "import nltk;nltk.download('all');"

# https://jupyter-server.readthedocs.io/en/latest/users/configuration.html
jupyter notebook --generate-config
jupyter server --generate-config
cat >> /home/$USER/.jupyter/jupyter_server_config.py << EOF

c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.allow_remote_access =True
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 8888
c.NotebookApp.password_required = True
c.NotebookApp.password = 'sha1:ab4969df6701:c5a55bd723e8a5e18db1b37db9a8021ffafd39e0'

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.allow_remote_access =True
c.ServerApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.ServerApp.port = 8888
c.ServerApp.password_required = True
c.ServerApp.password = 'sha1:ab4969df6701:c5a55bd723e8a5e18db1b37db9a8021ffafd39e0'
c.ContentsManager.allow_hidden=True

EOF

cp /home/$USER/.jupyter/jupyter_server_config.py /home/$USER/.jupyter/jupyter_notebook_config.py


pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
jupyter contrib nbextension install --user

jupyter nbextension enable codefolding/main
jupyter nbextension enable execute_time/ExecuteTime
jupyter nbextension enable addbefore/main
jupyter nbextension enable hinterland/hinterland
jupyter nbextension enable autosavetime/main
jupyter nbextension enable collapsible_headings/main
jupyter nbextension enable scratchpad/main
jupyter nbextension enable table_beautifier/main
jupyter nbextension enable toc2/main
jupyter nbextension enable toggle_all_line_numbers/main
jupyter nbextension enable code_prettify/code_prettify

cd ~

# https://jupyterlab.readthedocs.io/en/latest/user/extensions.html
# conda install -y nodejs > /dev/null
# conda install -y -c conda-forge nodejs > /dev/null
# https://stackoverflow.com/questions/62325068/cannot-install-latest-nodejs-using-conda-on-mac
# --repodata-fn=repodata.json is needed to install 12.0 + node which is needed for build
# conda install -y nodejs -c conda-forge --repodata-fn=repodata.json > /dev/null
# conda update -y nodejs

# https://github.com/nodesource/distributions
# curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash - &&\
# sudo apt-get install -y nodejs
curl -fsSL https://rpm.nodesource.com/setup_14.x | sudo sudo bash -
sudo yum install nodejs-14.21.3

pip install jupyterlab-topbar
pip install jupyterlab-system-monitor
pip install jupyterlab_execute_time
# pip install jupyterlab-lsp
# pip install 'python-lsp-server[all]'
# pip install nbdime
# pip install lckr-jupyterlab-variableinspector
# pip install jupyterlab-git

# pip install jupyterlab-hide-code
# pip install jupyterlab-material-night-eighties
# pip install jupyterlab-filesystem-access
# pip install jupyterlab-night

# pip uninstall nbdime lckr-jupyterlab-variableinspector jupyterlab-git jupyterlab-hide-code jupyterlab-material-night-eighties jupyterlab-filesystem-access jupyterlab-night jupyterlab-lsp 'python-lsp-server[all]'
jupyter lab build

cat >> jupyter_start.sh << "EOF"

#!/bin/bash
# source ~/.bashrc
export PATH="/home/$USER/anaconda3/bin:$PATH"
jupyter_str=`ps -ef | grep "/home/$USER/anaconda3/bin/jupyter-notebook" | grep -v grep`
HOST=`hostname -f`

if [ ! -z "$jupyter_str" -a "$jupyter_str" != " " ]; then
  echo "jupyter server already running"
  echo "Access at http://$HOST:8888/"
  echo "$jupyter_str"
fi

if [ -z "$jupyter_str" -o "$jupyter_str" == "" ]; then
    echo "Starting Server..."
    nohup jupyter notebook &
    echo "Access at http://$HOST:8888/"
fi

EOF

cat >> jupyter_lab_start.sh << "EOF"

#!/bin/bash
# source ~/.bashrc
export PATH="/home/$USER/anaconda3/bin:$PATH"
jupyter_str=`ps -ef | grep "/home/$USER/anaconda3/bin/jupyter-lab" | grep -v grep`
HOST=`hostname -f`

if [ ! -z "$jupyter_str" -a "$jupyter_str" != " " ]; then
  echo "jupyter lab server already running"
  echo "Access at http://$HOST:8888/lab"
  echo "$jupyter_str"
fi

if [ -z "$jupyter_str" -o "$jupyter_str" == "" ]; then
    echo "Starting lab Server..."
    nohup jupyter lab &
    echo "Access at http://$HOST:8888/lab"
fi

EOF

cat >> jupyter_stop.sh << "EOF"
#!/bin/bash
# source ~/.bashrc
jupyter_str=`/bin/ps -fu $USER| grep "jupyter-notebook" | grep -v "grep" | awk '{print $2}'`
echo $jupyter_str
if [ ! -z "$jupyter_str" -a "$jupyter_str" != " " ]; then
    echo "killing jupyter server at $jupyter_str"
    kill -9 $jupyter_str
fi

if [ -z "$jupyter_str" -o "$jupyter_str" == "" ]; then
    echo "No server running..."
fi

EOF

cat >> jupyter_lab_stop.sh << "EOF"
#!/bin/bash
# source ~/.bashrc
jupyter_str=`/bin/ps -fu $USER| grep "jupyter-lab" | grep -v "grep" | awk '{print $2}'`
echo $jupyter_str
if [ ! -z "$jupyter_str" -a "$jupyter_str" != " " ]; then
    echo "killing jupyter lab server at $jupyter_str"
    kill -9 $jupyter_str
fi

if [ -z "$jupyter_str" -o "$jupyter_str" == "" ]; then
    echo "No server running..."
fi

EOF

chmod 777 jupyter_stop.sh
chmod 777 jupyter_start.sh

chmod 777 jupyter_lab_stop.sh
chmod 777 jupyter_lab_start.sh
cd ~

yes | ssh-keygen -N "" -f "/home/$USER/.ssh/id_rsa" -t rsa -b 4096 -C "$USER@amazon.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
echo "Your public key for use:"
cat ~/.ssh/id_rsa.pub

pip install --use-feature=fast-deps --use-deprecated=legacy-resolver -U dataclasses-json einops nlpaug attrdict tabulate tqdm wandb dataclasses dataclasses-json transformers datasets tokenizers dill nltk more-itertools nlpaug pyarrow pytorch-ranger sacrebleu sacremoses sentencepiece snakeviz torch-optimizer jsonlines imgaug albumentations attrdict uniplot > /dev/null
conda install -y -c anaconda ipykernel ipython

cd ~ && mkdir mygit && cd mygit && git clone https://github.com/faizanahemad/offsite-tuning.git
cd ~/mygit/offsite-tuning && pip install -e .
git clone https://github.com/faizanahemad/accelerate.git && cd accelerate && pip install -e .
# python -c "import wandb; wandb.login(\"never\", \"7fcf597a1a07dcb2e98622a96838a7eb41b1245d\")"


conda activate base
pip uninstall -y jupyterlab
pip install jupyterlab
sudo yum -y install mlocate
sudo updatedb
curl -fsSL https://code-server.dev/install.sh | sh
sudo systemctl enable --now code-server@$USER
sed -i.bak 's/auth: password/auth: none/' ~/.config/code-server/config.yaml

# https://coder.com/docs/code-server/latest/guide#using-a-self-signed-certificate
sed -i.bak 's/cert: false/cert: true/' ~/.config/code-server/config.yaml
sed -i.bak 's/bind-addr: 127.0.0.1:8080/bind-addr: 0.0.0.0:8080/' ~/.config/code-server/config.yaml
sudo setcap cap_net_bind_service=+ep /usr/lib/code-server/lib/node

sudo systemctl restart code-server@$USER
~/jupyter_lab_start.sh
sudo mount -t nfs -o nolock dev-dsk-ahemf-i3-16x-2566f711.us-west-2.amazon.com:/local/home/ahemf/processed_datasets /home/ahemf/processed_datasets

# From local machine do
# ssh -N -L 8080:127.0.0.1:8080 [user]@<instance-ip>
# goto http://127.0.0.1:8080
# See https://coder.com/docs/code-server/latest/guide#port-forwarding-via-ssh and https://github.com/coder/code-server 

# ssh -i spohio.pem -N -L 8080:127.0.0.1:8080 ubuntu@ec2-3-16-36-106.us-east-2.compute.amazonaws.com # code-server
# ssh -i spohio.pem -N -L 8890:127.0.0.1:8890 ubuntu@ec2-3-16-36-106.us-east-2.compute.amazonaws.com # jupyter-lab

# sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 172.31.19.252:/ video_ads