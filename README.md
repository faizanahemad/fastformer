# fastformer
fastformer model, data and training code

For SDCONV with cuda see here: 

On Mac
```
brew install libssh
brew reinstall openssl@1.1
export PATH="/usr/local/opt/openssl@1.1/bin:$PATH" >> ~/.zshrc
export LDFLAGS="-L/usr/local/opt/openssl@1.1/lib"
export CPPFLAGS="-I/usr/local/opt/openssl@1.1/include"
export LIBS=-L/usr/local/opt/openssl@1.1/lib
pip install parallel-ssh
```
