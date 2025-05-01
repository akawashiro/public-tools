#! /bin/bash -eux
#
# Usage
# - `./linux-install.sh` just build Linux kernel
# - `DELETE_SAME_NAME_KERNELS=yes INSTALL_BUILT_KERNEL=yes REBOOT_AFTER_INSTALL=yes ./linux-install.sh` 
#   to build and install Linux kernel

# Constant settings
USE_CPUS=$(nproc --all)
USE_CPUS=$((USE_CPUS-2))

# Variable settings
LINUX_REPOSITORY=${LINUX_REPOSITORY:-https://github.com/akawashiro/linux}
BRANCH_NAME=${BRANCH_NAME:-v6.11}
INSTALL_PACKAGES=${INSTALL_PACKAGES:-no}
DELETE_SAME_NAME_KERNELS=${DELETE_SAME_NAME_KERNELS:-no}
INSTALL_BUILT_KERNEL=${INSTALL_BUILT_KERNEL:-no}
REBOOT_AFTER_INSTALL=${REBOOT_AFTER_INSTALL:-no}
CONFIG_PATH=${CONFIG_PATH:-/boot/config-$(uname -r)}
LINUX_DIR=${LINUX_DIR:-$(ghq root)/github.com/akawashiro/linux}
BUILD_DIR=${BUILD_DIR:-${HOME}/tmp/linux-build}
LOCALVERSION=-built-by-$(whoami)-${BRANCH_NAME}

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
    sudo apt-get -y build-dep linux
    sudo apt-get -y install \
        libncurses-dev \
        gawk \
        flex \
        bison \
        openssl \
        libssl-dev \
        dkms \
        libelf-dev \
        libudev-dev \
        libpci-dev \
        libiberty-dev \
        autoconf \
        git \
        ccache \
        dwarves \
        cmake \
        bear \
        clang
fi

# Sometimes you must delete all obsoluted kernels before install new one.
# pushd /boot
# ls -1 | grep "device-file-experiment" | xargs sudo rm
# popd
# pushd /lib/modules
# ls -1 | grep "device-file-experiment" | xargs sudo rm -r
# sudo update-grub
# popd

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
    git clone "${LINUX_REPOSITORY}" "${LINUX_DIR}"
fi
pushd ${LINUX_DIR}
git fetch --all --depth=1
git checkout ${BRANCH_NAME}
popd

# Prepare build dir
mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}"

# cp "${CONFIG_PATH}" "${BUILD_DIR}"/.config
# cat .config | \
#     sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | \
#     sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" \
#     > .config.bak
# cp .config.bak .config
popd

# Build Linux kernel
pushd "${LINUX_DIR}"
make defconfig LOCALVERSION=${LOCALVERSION} CC="ccache clang" O="${BUILD_DIR}"
make kvm_guest.config LOCALVERSION=${LOCALVERSION} CC="ccache clang" O="${BUILD_DIR}"
make LOCALVERSION=${LOCALVERSION} CC="ccache clang" -j ${USE_CPUS} O="${BUILD_DIR}"
./scripts/clang-tools/gen_compile_commands.py --directory "${BUILD_DIR}"

# Install Linux kernel
if [[ "${INSTALL_BUILT_KERNEL}" == yes ]]
then
    sudo make modules_install O="${BUILD_DIR}"
    sudo make install O="${BUILD_DIR}"

    if [[ "${REBOOT_AFTER_INSTALL}" == yes ]]
    then
        sudo reboot now
    fi
fi
popd
