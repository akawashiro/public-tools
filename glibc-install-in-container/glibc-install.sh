#! /bin/bash

N_CPUS=$(cat /proc/cpuinfo | grep processor | wc -l)

cd /root/
if [ ! -d "/root/glibc" ]; then
    git clone https://github.com/akawashiro/glibc.git glibc
fi
pushd glibc
git checkout glibc-2.31
popd

mkdir -p /root/glibc-build
cd /root/glibc-build
pwd
../glibc/configure --prefix=/root/glibc-install
# ../glibc/configure --disable-sanity-checks
make -j ${N_CPUS}
make install
