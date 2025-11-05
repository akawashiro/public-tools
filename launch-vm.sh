#! /bin/bash -eux

HOST_NAME=rausu3
DISK_PATH=${HOME}/${HOST_NAME}.qcow3
MEMORY_SIZE_MB=2048
PORT_FORWARD=5555

qemu-system-x86_64 \
    -hda ${DISK_PATH} \
    -m ${MEMORY_SIZE_MB} \
    -boot d \
    --enable-kvm \
    -usb \
    -serial none \
    -parallel none \
    -smp $(nproc) \
    -device e1000,netdev=net0 \
    -netdev user,id=net0,hostfwd=tcp::${PORT_FORWARD}-:22
