#! /bin/bash -eux

HOST_NAME=rausu3
DISK_PATH=${HOME}/${HOST_NAME}.qcow3
DISK_SIZE_GB=80G
MEMORY_SIZE_MB=2048
PORT_FORWARD=5555
UBUNTU_DISK_PATH=~/Downloads/ubuntu-22.04.1-desktop-amd64.iso

qemu-img create -f qcow2 ${DISK_PATH} ${DISK_SIZE_GB}
qemu-system-x86_64 -hda ${DISK_PATH} -m ${MEMORY_SIZE_MB} -cdrom ${UBUNTU_DISK_PATH} -boot d --enable-kvm -usb -serial none -parallel none

# Do Ubuntu install here
# sudo apt install openssh-server git neovim

qemu-system-x86_64 -hda ${DISK_PATH} -m ${MEMORY_SIZE_MB} -boot d --enable-kvm -usb -serial none -parallel none -smp $(nproc) -device e1000,netdev=net0 -netdev user,id=net0,hostfwd=tcp::${PORT_FORWARD}-:22

# Set up ssh using following command in another terminal
# ssh localhost -p ${PORT_FORWARD} "mkdir /home/akira/.ssh/"
# scp -P ${PORT_FORWARD} ~/.ssh/id_rsa.pub localhost:~/.ssh/authorized_keys
# sudo shutdown now
