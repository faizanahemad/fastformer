source ~/.zshrc
cd ~/mygit/fairseq
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py
python setup.py install
cd ~/mygit/fairseq
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py
python setup.py install
cd ~/mygit/fairseq
pip install -e .
cd ~/mygit/fairseq
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py
python setup.py install
cd ~/mygit/fairseq
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py
python setup.py install
cd ~/mygit/fastformer