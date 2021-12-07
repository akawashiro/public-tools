#! /bin/bash -eux

# sudo apt-get -y build-dep linux linux-image-$(uname -r)
sudo apt-get -y install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf git ccache dwarves

LINUX_DIR=${HOME}/linux
BRANCH_NAME=fix-load_addr-patch-v4
USE_CPUS=$(nproc --all)
USE_CPUS=$((USE_CPUS-2))

if [[ ! -d "${LINUX_DIR}" ]]
then
    git clone https://github.com/akawashiro/linux.git "${LINUX_DIR}"
fi

cd ${LINUX_DIR}
git fetch --all
git checkout origin/${BRANCH_NAME}
cp /boot/config-$(uname -r) ${LINUX_DIR}/.config
cat ${LINUX_DIR}/.config | sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" > ${LINUX_DIR}/.config.bak
cp ${LINUX_DIR}/.config.bak ${LINUX_DIR}/.config
make localmodconfig
LOCALVERSION=-dev-${BRANCH_NAME} make CC="ccache gcc" -j ${USE_CPUS}
sudo make modules_install
sudo make install

read -p "reboot? (y/N): " yn
case "$yn" in
  [yY]*) sudo reboot now;;
esac
