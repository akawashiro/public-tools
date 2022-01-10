#! /bin/bash

cd /root/
git clone https://github.com/bminor/glibc.git
mkdir -p /root/glibc-build
cd /root/glibc-build
../glibc/configure --prefix=/root/glibc-install
make -j10
make install
