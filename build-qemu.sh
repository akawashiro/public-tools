#! /bin/bash

set -eux -o pipefail

QEMU_PATH=$(ghq root)/github.com/akawashiro/qemu
BUILD_PATH=${HOME}/tmp/qemu-build

if [[ ! -d ${QEMU_PATH} ]]; then
  echo "QEMU_PATH not found: ${QEMU_PATH}"
  exit 1
fi

CAPSTONE_INSTALLED=$(apt list --installed | grep libcapstone-dev | wc -l)
if [[ ${CAPSTONE_INSTALLED} -eq 0 ]]; then
  echo Please run sudo apt install -y libcapstone-dev
fi

mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}
${QEMU_PATH}/configure --prefix=~/tmp/qemu-install --enable-debug --enable-capstone
bear -- make -j$(nproc)
ln -sf ~/tmp/qemu-build/compile_commands.json ${QEMU_PATH}/compile_commands.json
make install
