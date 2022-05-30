#! /bin/bash -ex

cd $HOME
ln -sf $HOME/public-tools/.tmux.conf
ln -sf $HOME/public-tools/.zshrc
ln -sf $HOME/public-tools/nvim $HOME/.config/nvim
curl -sL --proto-redir -all,https https://raw.githubusercontent.com/zplug/installer/master/installer.zsh | zsh
pip3 install --user powerline-status
