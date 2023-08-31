# Need to change PATH and LD_LIBRARY_PATH if you install Cuda at a different location
mkdir -p /home/$USER/processed_datasets
cat > ~/.zshrc << "EOF"

export ZSH=/home/$USER/.oh-my-zsh

ZSH_THEME="robbyrussell"

ENABLE_CORRECTION="true"
export AUTO_TITLE_SCREENS="NO"

plugins=(git git-extras history history-substring-search extract python yum)


source $ZSH/oh-my-zsh.sh

bindkey "^[[A" history-substring-search-up

bindkey "^[[B" history-substring-search-down

export AUTO_TITLE_SCREENS="NO"
alias grep='nocorrect grep --color=auto'
alias nvkill="nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9"
alias mpkill="kill $(ps aux | grep multiprocessing | grep -v grep | awk '{print $2}')"
setopt HIST_VERIFY
export PATH="$PATH:/usr/local/cuda-11.6/bin"
export PATH="$PATH:/home/$USER/nvidia/toolkit/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/nvidia/toolkit/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib"
export PATH="$PATH:/usr/lib64"
export HISTSIZE=10000
export SAVEHIST=10000
setopt HIST_IGNORE_ALL_DUPS
setopt autocd nomatch
setopt HIST_EXPIRE_DUPS_FIRST
setopt COMPLETE_IN_WORD
DISABLE_AUTO_UPDATE="true"
# export PATH="/home/$USER/anaconda3/bin:$PATH"  # commented out by conda initialize

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
conda_path="/home/$USER/anaconda3"
conda_str="$conda_path/bin/conda"
__conda_setup="$($conda_str 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
        . "$conda_path/etc/profile.d/conda.sh"
    else
        export PATH="$conda_path/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

mkdir -p $HOME/tmp
mkdir -p $HOME/cache
export TMPDIR=$HOME/tmp
export TMP=$HOME/tmp

EOF

cat > ~/.vimrc << "EOF"
syntax enable
set tabstop=4
set softtabstop=4
set expandtab
set cursorline
set number
set showmatch
set incsearch
set hlsearch
EOF

wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh > /dev/null
sh Anaconda3-2023.03-Linux-x86_64.sh -b
. ~/.zshrc

# Setup docker location
# https://linuxconfig.org/how-to-move-docker-s-default-var-lib-docker-to-another-directory-on-ubuntu-debian-linux
# sudo systemctl daemon-reload
# sudo systemctl restart docker
