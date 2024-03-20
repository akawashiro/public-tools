#!/bin/bash -eux

if [ $# -ne 1 ]; then
    echo "You must specify the target directory."
    exit 1
fi

git clone git@github.com:akawashiro/pytorch.git $1
cd $1
git remote add upstream git@github.com:pytorch/pytorch.git
git remote set-url upstream --push no-push
