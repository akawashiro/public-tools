#! /bin/bash

set -ux

# ========== Fundamental packages ==========
sudo sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
sudo add-apt-repository -y ppa:neovim-ppa/stable
sudo apt update
sudo apt build-dep -y linux
sudo apt install -y libtool libtool-bin gnustep-gui-runtime automake lua-system lua-term lua-dkjson lua-mediator lua-filesystem curl git zsh tmux openssh-server curl python3-dev python3-pip ncurses-dev lua5.2 luajit ruby-dev libperl-dev software-properties-common build-essential silversearcher-ag htop pwgen m4 cmake tig xclip gettext neovim libclang-dev fakeroot libncurses-dev iftop git-email neomutt golang-go locales-all pass

$HOME/public-tools/setup-without-root.sh
