#! /bin/bash -eux

# sudo apt install clang-11

LINUX_DIR=${HOME}/linux-build-test
BRANCH_NAME=fix-load_addr-v3-patch
USE_CPUS=$(nproc --all)
USE_CPUS=$((USE_CPUS-2))

rm -rf ${LINUX_DIR}
git clone --depth=1 --branch ${BRANCH_NAME} --single-branch https://github.com/akawashiro/linux.git ${LINUX_DIR}

cd ${LINUX_DIR}

for c_compiler in "clang-11 --analyze" "gcc"
do
    make clean
    ccache -C
    make KCFLAGS="-W" CC="ccache ${c_compiler}" -k olddefconfig
    cat ${LINUX_DIR}/.config | \
        sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | \
        sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" \
        > ${LINUX_DIR}/.config.bak
    cp ${LINUX_DIR}/.config.bak ${LINUX_DIR}/.config
    log=$(echo "${LINUX_DIR}/build.${c_compiler}.log" | tr " " "_")
    LOCALVERSION=-dev-${BRANCH_NAME} make KCFLAGS="-W" CC="ccache ${c_compiler}" -k -j ${USE_CPUS} 2>&1 | tee ${log}
done
