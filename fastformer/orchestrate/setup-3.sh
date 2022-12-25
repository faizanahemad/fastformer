source ~/.zshrc
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 > /dev/null
pip install --use-deprecated=legacy-resolver -U tqdm wandb> /dev/null
pip install --use-deprecated=legacy-resolver -U sacrebleu sacremoses sentencepiece  > /dev/null
pip install --use-deprecated=legacy-resolver -U dataclasses dataclasses-json  > /dev/null
pip install --use-deprecated=legacy-resolver -U dill nltk more-itertools  > /dev/null
pip install --use-deprecated=legacy-resolver -U gpustat nlpaug pyarrow  > /dev/null
pip install --use-deprecated=legacy-resolver -U snakeviz torch-optimizer attrdict  > /dev/null
pip install --use-deprecated=legacy-resolver -U pytorch-fast-transformers pytorch-ranger  > /dev/null
pip install --use-deprecated=legacy-resolver -U transformers datasets tokenizers  > /dev/null
pip install --use-deprecated=legacy-resolver -U joblib scikit-learn opencv-python einops timm > /dev/null
pip install opencv-python > /dev/null
# https://github.com/pypa/pip/issues/9187
# --use-feature=fast-deps --use-deprecated=legacy-resolver

python -c "import nltk;nltk.download('all');"

jupyter notebook --generate-config

cat >> /home/$USER/.jupyter/jupyter_notebook_config.py << EOF

c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.allow_remote_access =True
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 8888
c.NotebookApp.password_required = True
c.NotebookApp.password = 'sha1:ab4969df6701:c5a55bd723e8a5e18db1b37db9a8021ffafd39e0'

EOF

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

cat >> jupyter_stop.sh << "EOF"
#!/bin/bash
# source ~/.bashrc
jupyter_str=`/bin/ps -fu $USER| grep "jupyter-notebook" | grep -v "grep" | awk '{print $2}'`
echo $jupyter_str
if [ ! -z "$jupyter_str" -a "$jupyter_str" != " " ]; then
    echo "killing jupyter server at $jupyter_str"
    kill $jupyter_str
fi

if [ -z "$jupyter_str" -o "$jupyter_str" == "" ]; then
    echo "No server running..."
fi

EOF

chmod 777 jupyter_stop.sh
chmod 777 jupyter_start.sh
cd ~

yes | ssh-keygen -N "" -f "/home/$USER/.ssh/id_rsa" -t rsa -b 4096 -C "$USER@amazon.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
echo "Your public key for use:"
cat ~/.ssh/id_rsa.pub
~/jupyter_start.sh

pip install --use-feature=fast-deps --use-deprecated=legacy-resolver -U tqdm wandb dataclasses dataclasses-json transformers datasets tokenizers dill nltk more-itertools gpustat nlpaug pyarrow pytorch-fast-transformers pytorch-ranger sacrebleu sacremoses sentencepiece snakeviz torch-optimizer jsonlines imgaug albumentations attrdict uniplot > /dev/null
pip install --use-feature=fast-deps --use-deprecated=legacy-resolver -U dataclasses-json einops nlpaug pytorch-fast-transformers timm attrdict tabulate > /dev/null
pip install --use-feature=fast-deps --use-deprecated=legacy-resolver -U unidecode einops nlpaug pyarrow pytorch-fast-transformers transformers datasets wandb tqdm tokenizers pytorch-ranger torch-optimizer dataclasses-json > /dev/null

cd ~ && mkdir mygit && cd mygit && git clone https://github.com/faizanahemad/fastformer.git && cd fastformer && pip install -e .
python -c "import wandb; wandb.login(\"never\", \"7fcf597a1a07dcb2e98622a96838a7eb41b1245d\")"

#cd ~/mygit && git clone https://github.com/NVIDIA/apex
#cd ~/mygit/apex
#pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#DS_BUILD_OPS=1 pip install deepspeed

#pip install --use-feature=fast-deps --use-deprecated=legacy-resolver -U fairscale > /dev/null
#
#cd ~/mygit
#
#cd ~/mygit && git clone https://github.com/pytorch/fairseq.git
#cd ~/mygit/fairseq
#cd fairseq/modules/lightconv_layer
#python cuda_function_gen.py > /dev/null
#python setup.py install > /dev/null
#cd ~/mygit/fairseq
#cd fairseq/modules/dynamicconv_layer
#python cuda_function_gen.py > /dev/null
#python setup.py install > /dev/null
#cd ~/mygit/fairseq
#pip install -e .
#cd ~/mygit/fairseq
#cd fairseq/modules/lightconv_layer
#python cuda_function_gen.py > /dev/null
#python setup.py install > /dev/null
#cd ~/mygit/fairseq
#cd fairseq/modules/dynamicconv_layer
#python cuda_function_gen.py > /dev/null
#python setup.py install > /dev/null
#cd ~/mygit/fastformer
#
#pip freeze | grep dynamicconv-layer
#pip freeze | grep fairscale
#pip freeze | grep torch

sudo yum -y install mlocate
sudo updatedb


exit
