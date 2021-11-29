#! /bin/bash -eux

LINUX_DIR=${HOME}/linux-build-test
BRANCH_NAME=fix-load_addr-v3-patch

rm -rf ${LINUX_DIR}
git clone --depth=1 --branch ${BRANCH_NAME} --single-branch git@github.com:akawashiro/linux.git ${LINUX_DIR}

cd ${LINUX_DIR}

for c_compiler in gcc clang
do
    make clean
    ccache -C
    make KCFLAGS="-W" CC="ccache ${c_compiler}" -k olddefconfig
    cat ${LINUX_DIR}/.config | sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" > ${LINUX_DIR}/.config.bak
    cp ${LINUX_DIR}/.config.bak ${LINUX_DIR}/.config
    LOCALVERSION=-dev-${BRANCH_NAME} make KCFLAGS="-W" CC="ccache ${c_compiler}" -k -j4 2>&1 | tee ${LINUX_DIR}/build.${c_compiler}.log
done
