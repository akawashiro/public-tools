#! /bin/bash -eux

SOURCE_DIR=$(ghq list -p | grep pytorch | head -n 1)

BUILD_DIR=${HOME}/tmp/pytorch-build
INSTALL_DIR=${HOME}/tmp/pytorch-install
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}

pushd ${SOURCE_DIR}
rm -rf build
ln -sf ${BUILD_DIR} build

export CMAKE_EXPORT_COMPILE_COMMANDS=1
export DEBUG=1
export MAX_JOBS=16
export USE_CUDA=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_DISTRIBUTED=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_DISTRIBUTED=0
export USE_TENSORPIPE=0
export USE_GLOO=0
export USE_MPI=0
export USE_SYSTEM_NCCL=0
export BUILD_CAFFE2_OPS=0
export BUILD_CAFFE2=0
export USE_IBVERBS=0
export USE_OPENCV=0
export USE_OPENMP=0
export USE_FFMPEG=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_LEVELDB=0
export _GLIBCXX_USE_CXX11_ABI=1

python3 setup.py build --build-base ${BUILD_DIR} --debug
python3 setup.py install --prefix ${INSTALL_DIR}
