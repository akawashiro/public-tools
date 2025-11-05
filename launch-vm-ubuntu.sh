#! /bin/bash -eux

HOST_NAME=rausu3
DISK_PATH=${HOME}/${HOST_NAME}.qcow2
DISK_SIZE_GB=80G
MEMORY_SIZE_MB=2048
PORT_FORWARD=5555
UBUNTU_DISK_PATH=~/Downloads/ubuntu-22.04.1-desktop-amd64.iso
INSTALL=${INSTALL:-1}

CDROM_OPTIONS=""
if [ "${INSTALL}" -eq "1" ]; then
    qemu-img create -f qcow2 ${DISK_PATH} ${DISK_SIZE_GB}
    CDROM_OPTIONS="-cdrom ${UBUNTU_DISK_PATH} -boot d"
fi

if [ ! -f "${DISK_PATH}" ]; then
    echo "Disk image not found at ${DISK_PATH}. Please set INSTALL=1 to create a new disk and install the OS."
    exit 1
fi

qemu-system-x86_64 \
    -hda ${DISK_PATH} \
    -m ${MEMORY_SIZE_MB} \
    --enable-kvm \
    -usb \
    -serial none \
    -parallel none \
    -smp $(nproc) \
    -device e1000,netdev=net0 \
    -netdev user,id=net0,hostfwd=tcp::${PORT_FORWARD}-:22 \
    ${CDROM_OPTIONS}

# Do Ubuntu install here after installing from the ISO image, set up ssh.
#
# sudo apt install openssh-server git neovim
#
# Set up ssh using following command in another terminal
#
# ssh localhost -p ${PORT_FORWARD} "mkdir /home/akira/.ssh/"
# scp -P ${PORT_FORWARD} ~/.ssh/id_rsa.pub localhost:~/.ssh/authorized_keys
# sudo shutdown now
