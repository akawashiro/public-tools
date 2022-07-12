#! /bin/bash -ex

pip install --upgrade pip
cd $HOME
ln -sf $HOME/public-tools/.tmux.conf
ln -sf $HOME/public-tools/.zshrc
ln -sf $HOME/public-tools/nvim $HOME/.config/nvim
curl -sL --proto-redir -all,https https://raw.githubusercontent.com/zplug/installer/master/installer.zsh | zsh
pip install --user powerline-status
