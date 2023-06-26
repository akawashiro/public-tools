#! /bin/bash -ex

pip install --upgrade pip
cd $HOME
mkdir -p ${HOME}/.config
ln -sf $HOME/public-tools/.tmux.conf
ln -sf $HOME/public-tools/.zshrc
ln -sf $HOME/public-tools/nvim $HOME/.config/nvim
touch .zenlog.toml

if [ ! -d ${HOME}/.zplug ]
then
    curl -sL --proto-redir -all,https https://raw.githubusercontent.com/zplug/installer/master/installer.zsh | zsh
fi

if [ ! -d ${HOME}/.tmux/plugins/tpm ]
then
    git clone https://github.com/tmux-plugins/tpm ${HOME}/.tmux/plugins/tpm
fi

# ========== python3 package ==========
cd $HOME
pip3 install --upgrade pip --user
pip3 install neovim flake8 ydiff msgpack --user

# ========== Go package ===========
cd $HOME
go install github.com/x-motemen/ghq@latest
