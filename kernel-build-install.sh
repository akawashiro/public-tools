#! /bin/bash -eux

LINUX_DIR=${HOME}/linux
BRANCH_NAME=fix-load_addr-v2

cd ${LINUX_DIR}
git fetch --all
git checkout origin/${BRANCH_NAME}
cp /boot/config-$(uname -r) .config
LOCALVERSION=-dev-${BRANCH_NAME} make CC="ccache gcc" -j4
sudo make modules_install
sudo make install

read -p "reboot? (y/N): " yn
case "$yn" in
  [yY]*) sudo reboot now;;
esac
