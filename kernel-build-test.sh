#! /bin/bash -eux

LINUX_DIR=${HOME}/linux-build-test
BRANCH_NAME=fix-load_addr-patch-v4
USE_CPUS=$(nproc --all)
USE_CPUS=$((USE_CPUS-2))

if [[ ! -d "${LINUX_DIR}" ]]
then
    git clone https://github.com/akawashiro/linux.git "${LINUX_DIR}"
fi

cd ${LINUX_DIR}
git fetch --all
git checkout origin/${BRANCH_NAME}

for c_compiler in "clang-13" "gcc"
do
	BUILD_DIR=${HOME}/linux-build-test-build-${c_compiler}
	mkdir ${BUILD_DIR}
    make clean
    make KCFLAGS="-W" CC="ccache ${c_compiler}" olddefconfig O=${BUILD_DIR}
	cd ${BUILD_DIR}
    cat ${BUILD_DIR}/.config | \
		sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | \
		sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" \
		> ${BUILD_DIR}/.config.bak
    cp ${BUILD_DIR}/.config.bak ${BUILD_DIR}/.config
    log=$(echo "${BUILD_DIR}/build.${c_compiler}.log" | tr " " "_")
    LOCALVERSION=-dev-${BRANCH_NAME} make KCFLAGS="-W" CC="ccache ${c_compiler}" -j ${USE_CPUS} 2>&1 | tee ${log}
done
