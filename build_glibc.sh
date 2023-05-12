#! /bin/bash -eux

SOURCE_DIR=$(ghq root)/github.com/akawashiro/glibc
BUILD_DIR=${HOME}/tmp/glibc-build
INSTALL_DIR=${HOME}/tmp/glibc-install

rm -rf ${BUILD_DIR} ${INSTALL_DIR}
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}

cd ${BUILD_DIR}
LD_LIBRARY_PATH="" ${SOURCE_DIR}/configure CC="ccache gcc" --prefix=${INSTALL_DIR}
LD_LIBRARY_PATH="" bear -- make all -j 20
LD_LIBRARY_PATH="" bear -- make install -j 20
ln -sf ${BUILD_DIR}/compile_commands.json ${SOURCE_DIR}/compile_commands.json
