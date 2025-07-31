#! /bin/bash -eux

SOURCE_DIR=$(ghq list -p | grep pytorch | head -n 1)
BUILD_DIR=${HOME}/tmp/libtorch-build
INSTALL_DIR=${HOME}/tmp/libtorch-install
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}

pushd ${SOURCE_DIR}
git submodule update --init --recursive
popd

cmake -S ${SOURCE_DIR} -B ${BUILD_DIR} -G Ninja \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DPYTHON_EXECUTABLE:PATH=`which python3` \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DUSE_CUDA=OFF \
    -DUSE_CUDNN=OFF \
    -DUSE_ROCM=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_NNPACK=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}
cmake --build ${BUILD_DIR} -- -j install
