#! /bin/bash -eux

# sudo apt-get -y build-dep linux linux-image-$(uname -r)
# sudo apt-get -y install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf git ccache dwarves

LINUX_DIR=${HOME}/linux
BUILD_DIR=${HOME}/linux-build
BRANCH_NAME=fix-load_addr-patch-v4
USE_CPUS=$(nproc --all)
USE_CPUS=$((USE_CPUS-2))

# We should not install kernels with the same name
pushd /boot
if [ $(ls -1 | grep linux | wc -l) -ne 0 ]; then
    ls -1 | grep ${BRANCH_NAME} | xargs sudo rm
fi
popd
pushd /lib/modules
if [ $(ls -1 | grep linux | wc -l) -ne 0 ]; then
    ls -1 | grep ${BRANCH_NAME} | xargs sudo rm -r
fi
popd
sudo update-grub

if [[ ! -d "${LINUX_DIR}" ]]
then
    git clone https://github.com/akawashiro/linux.git "${LINUX_DIR}"
fi

cd ${LINUX_DIR}
git fetch --all
git checkout origin/${BRANCH_NAME}

make clean

# TODO: Want to use out-of-tree build
# make CC="ccache gcc" O=${BUILD_DIR} olddefconfig 

cat .config | \
    sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | \
    sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" \
    > .config.bak
cp .config.bak .config
make LOCALVERSION=-dev-${BRANCH_NAME} CC="ccache gcc" olddefconfig
make LOCALVERSION=-dev-${BRANCH_NAME} CC="ccache gcc" -j ${USE_CPUS} 2>&1 | tee ${BUILD_DIR}/build.log
sudo make modules_install
sudo make install

read -p "reboot? (y/N): " yn
case "$yn" in
  [yY]*) sudo reboot now;;
esac
