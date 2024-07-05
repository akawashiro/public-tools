#!/bin/bash -eux

ghq get git@github.com:akawashiro/linux.git
cd $(ghq root)/github.com/akawashiro/linux
git remote add torvalds git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
git remote add linux-next https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git
git remote set-url torvalds --push no-push
git remote set-url linux-next --push no-push
