sudo yum -y remove nvidia-dkms nvidia > /dev/null
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run > /dev/null
chmod +x cuda_11.2.2_460.32.03_linux.run
sudo sudo sh cuda_11.2.2_460.32.03_linux.run --silent --driver --toolkit

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html > /dev/null
pip install fairscale > /dev/null


cd ~/mygit && git clone https://github.com/pytorch/fairseq.git
cd ~/mygit/fairseq
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py > /dev/null
python setup.py install > /dev/null
cd ~/mygit/fairseq
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py > /dev/null
python setup.py install > /dev/null
cd ~/mygit/fairseq
pip install -e .
cd ~/mygit/fairseq
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py > /dev/null
python setup.py install > /dev/null
cd ~/mygit/fairseq
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py > /dev/null
python setup.py install > /dev/null
cd ~/mygit/fastformer

pip freeze | grep dynamicconv-layer
pip freeze | grep fairscale

