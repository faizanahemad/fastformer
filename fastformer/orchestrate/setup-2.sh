mkdir -p /home/ahemf/processed_datasets
mkdir -p /home/ahemf/model_save_dir
mkdir -p /home/ahemf/torch_distributed_init
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
export PATH="$PATH":~/bin
export PATH="$PATH:/usr/local/cuda-11.2/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib"
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
__conda_setup="$('/home/ahemf/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ahemf/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ahemf/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ahemf/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

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

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh > /dev/null
sh Anaconda3-2021.11-Linux-x86_64.sh -b
. ~/.zshrc
