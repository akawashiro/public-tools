#! /bin/bash

apt update
apt install build-essential

cd /root/
if [ -d "/root/glibc" ]; then
    git clone https://github.com/bminor/glibc.git glibc
fi

mkdir -p /root/glibc-build
cd /root/glibc-build
../glibc/configure --prefix=/root/glibc-install
make -j10
make install
