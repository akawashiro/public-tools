#! /bin/bash -eux
#
# Usage
# - `./linux-install.sh` just build Linux kernel
# - `DELETE_SAME_NAME_KERNELS=yes INSTALL_BUILT_KERNEL=yes REBOOT_AFTER_INSTALL=yes ./linux-install.sh` 
#   to build and install Linux kernel

# Constant settings
LINUX_DIR=${HOME}/linux
BUILD_DIR=${HOME}/linux-build
USE_CPUS=$(nproc --all)
USE_CPUS=$((USE_CPUS-2))
CONFIG_PATH=/boot/config-$(uname -r)

# Variable settings
BRANCH_NAME=${BRANCH_NAME:-device-file-experiment}
INSTALL_PACKAGES=${INSTALL_PACKAGES:-no}
DELETE_SAME_NAME_KERNELS=${DELETE_SAME_NAME_KERNELS:-no}
INSTALL_BUILT_KERNEL=${INSTALL_BUILT_KERNEL:-no}
REBOOT_AFTER_INSTALL=${REBOOT_AFTER_INSTALL:-no}

# Check inputs
if [[ "${INSTALL_PACKAGES}" != yes ]] && [[ "${INSTALL_PACKAGES}" != no ]]
then
    echo INSTALL_PACKAGES is not valid: "${INSTALL_PACKAGES}"
fi
if [[ "${DELETE_SAME_NAME_KERNELS}" != yes ]] && [[ "${DELETE_SAME_NAME_KERNELS}" != no ]]
then
    echo DELETE_SAME_NAME_KERNELS is not valid: "${DELETE_SAME_NAME_KERNELS}"
fi
if [[ "${INSTALL_BUILT_KERNEL}" != yes ]] && [[ "${INSTALL_BUILT_KERNEL}" != no ]]
then
    echo INSTALL_BUILT_KERNEL is not valid: "${INSTALL_BUILT_KERNEL}"
fi
if [[ "${REBOOT_AFTER_INSTALL}" != yes ]] && [[ "${REBOOT_AFTER_INSTALL}" != no ]]
then
    echo REBOOT_AFTER_INSTALL is not valid: "${REBOOT_AFTER_INSTALL}"
fi

# Install packages
if [[ "${INSTALL_PACKAGES}" == yes ]]
then
    sudo apt-get -y build-dep linux linux-image-$(uname -r)
    sudo apt-get -y install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf git ccache dwarves cmake
fi

# We should not install kernels with the same name
if [[ "${DELETE_SAME_NAME_KERNELS}" == yes ]]
then
    pushd /boot
    if [ $(ls -1 | grep linux | wc -l) -ne 0 ]; then
        ls -1 | grep ${BRANCH_NAME} | xargs sudo rm
    fi
    popd
    pushd /lib/modules
    if [ $(ls -1 | grep linux | wc -l) -ne 0 ]; then
        ls -1 | grep ${BRANCH_NAME} | xargs sudo rm -r
    fi
    sudo update-grub
    popd
fi

# Prepare source code
if [[ ! -d "${LINUX_DIR}" ]]
then
    git clone https://github.com/akawashiro/linux.git "${LINUX_DIR}"
fi
pushd ${LINUX_DIR}
git fetch --all --depth=1
git checkout origin/${BRANCH_NAME}
popd

# Prepare build dir
mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}"

cp "${CONFIG_PATH}" "${BUILD_DIR}"/.config
cat .config | \
    sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | \
    sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" \
    > .config.bak
cp .config.bak .config
popd

# Build Linux kernel
pushd "${LINUX_DIR}"
bear -- make olddefconfig LOCALVERSION=-dev-${BRANCH_NAME} CC="ccache gcc" O="${BUILD_DIR}"
bear -- make LOCALVERSION=-dev-${BRANCH_NAME} CC="ccache gcc" -j ${USE_CPUS} O="${BUILD_DIR}" 2>&1 | tee "${BUILD_DIR}"/build-$(date +%s).log
popd

# Install Linux kernel
if [[ "${INSTALL_BUILT_KERNEL}" == yes ]]
then
    sudo make modules_install
    sudo make install

    if [[ "${REBOOT_AFTER_INSTALL}" == yes ]]
    then
        sudo reboot now
    fi
fi
