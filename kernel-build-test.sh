#! /bin/bash -eux

LINUX_DIR=${HOME}/linux-build-test
BRANCH_NAME=fix-load_addr-v2-patch

if [[ ! -d "${LINUX_DIR}" ]]
then
    git clone git@github.com:akawashiro/linux.git "${LINUX_DIR}"
fi

cd ${LINUX_DIR}
git fetch --all
git checkout origin/${BRANCH_NAME}

for c_compiler in gcc clang
do
    make clean
    ccache -C
    make KCFLAGS="-W" CC="ccache ${c_compiler}" -k olddefconfig
    cat ${LINUX_DIR}/.config | sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" > ${LINUX_DIR}/.config.bak
    cp ${LINUX_DIR}/.config.bak ${LINUX_DIR}/.config
    LOCALVERSION=-dev-${BRANCH_NAME} make KCFLAGS="-W" CC="ccache ${c_compiler}" -k -j4 2>&1 | tee ${LINUX_DIR}/build.${c_compiler}.log
done
