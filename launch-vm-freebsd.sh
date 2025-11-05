#! /bin/bash -eux

HOST_NAME=chausu
DISK_PATH=${HOME}/${HOST_NAME}.qcow2
DISK_SIZE_GB=80G
MEMORY_SIZE_MB=2048
PORT_FORWARD=5555
INSTALL_DISK_PATH=${HOME}/Downloads/FreeBSD-14.3-RELEASE-amd64-dvd1.iso
INSTALL=${INSTALL:-0}

CDROM_OPTIONS=""
if [ "${INSTALL}" -eq "1" ]; then
    if [ ! -f "${INSTALL_DISK_PATH}" ]; then
        echo "FreeBSD ISO not found at ${INSTALL_DISK_PATH}. Please download it first."
        exit 1
    fi
    if [ -f "${DISK_PATH}" ]; then
        echo "Disk image already exists at ${DISK_PATH}. Please remove it or set INSTALL=0 to use the existing disk."
        exit 1
    fi
    qemu-img create -f qcow2 ${DISK_PATH} ${DISK_SIZE_GB}
    CDROM_OPTIONS="-cdrom ${INSTALL_DISK_PATH} -boot d"
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

