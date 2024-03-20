#! /bin/bash

set -eux -o pipefail

QEMU_PATH=$(ghq root)/github.com/akawashiro/qemu
BUILD_PATH=${HOME}/tmp/qemu-build

if [[ ! -d ${QEMU_PATH} ]]; then
  echo "QEMU_PATH not found: ${QEMU_PATH}"
  exit 1
fi

mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}
${QEMU_PATH}/configure --prefix=~/tmp/qemu-install --enable-debug
bear -- make -j$(nproc)
ln -sf ~/tmp/qemu-build/compile_commands.json ${QEMU_PATH}/compile_commands.json
make install
