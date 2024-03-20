#!/bin/bash -eux

ghq get git@github.com:akawashiro/qemu.git
cd $(ghq root)/github.com/akawashiro/qemu
git remote add upstream https://github.com/qemu/qemu
git remote set-url --push upstream no_push
