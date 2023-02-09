#!/bin/bash -eux

if [ $# -ne 1 ]; then
    echo "You must specify the target directory."
    exit 1
fi

git clone git@github.com:akawashiro/glibc.git $1
cd $1
git remote add upstream  https://sourceware.org/git/glibc.git
git remote set-url upstream --push no-push
