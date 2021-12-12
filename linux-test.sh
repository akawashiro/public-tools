#! /bin/bash -eux

# You should run this script on the host.

LINUX_DIR=${HOME}/linux-test
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

for c_compiler in "clang-13"
do
	BUILD_DIR=$(echo ${LINUX_DIR}-build-${c_compiler} | tr " " "_")
	rm -rf ${BUILD_DIR}
	mkdir -p ${BUILD_DIR}
    make KCFLAGS="-W" CC="${c_compiler}" olddefconfig O=${BUILD_DIR}
	cd ${BUILD_DIR}
    cat ${BUILD_DIR}/.config | \
		sed -e "s/CONFIG_SYSTEM_TRUSTED_KEYS=".*"/CONFIG_SYSTEM_TRUSTED_KEYS=\"\"/g" | \
		sed -e "s/CONFIG_SYSTEM_REVOCATION_KEYS=".*"/CONFIG_SYSTEM_REVOCATION_KEYS=\"\"/g" \
		> ${BUILD_DIR}/.config.bak
    cp ${BUILD_DIR}/.config.bak ${BUILD_DIR}/.config
    compile_log=$(echo "${BUILD_DIR}/build.${c_compiler}.log" | tr " " "_")
    LOCALVERSION=-dev-${BRANCH_NAME} bear -- make KCFLAGS="-W" CC="ccache ${c_compiler}" -j ${USE_CPUS} 2>&1 | tee ${compile_log}
	CodeChecker analyze ${BUILD_DIR}/compile_commands.json --enable sensitive -o ${BUILD_DIR}/codechecker-reports --jobs ${USE_CPUS} --file ${LINUX_DIR}/fs/binfmt_elf.c
	CodeChecker parse ./codechecker-reports 2>&1 | tee codechecker-reports.txt
	CodeChecker parse ./codechecker-reports -e html -o ./codechecker-reports-html
    # analyzer_log=$(echo "${BUILD_DIR}/analyzer.${c_compiler}.log" | tr " " "_")
	# ${LINUX_DIR}/scripts/clang-tools/run-clang-tools.py clang-analyzer ${BUILD_DIR}/compile_commands.json 2>&1 | tee ${analyzer_log}
    # LOCALVERSION=-dev-${BRANCH_NAME} make KCFLAGS="-W" CC="${c_compiler}" -j ${USE_CPUS} 2>&1 | tee ${log}
done
