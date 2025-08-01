#! /bin/bash -ux

ghq get git@github.com:akawashiro/misc.git

ghq get git@github.com:akawashiro/sold.git

ghq get git@github.com:akawashiro/ros3fs.git

ghq get git@github.com:akawashiro/jendeley.git

ghq get git@github.com:akawashiro/sloader.git

ghq get git@github.com:akawashiro/z3shape.git

ghq get git@github.com:akawashiro/xla2onnx.git

ghq get git@github.com:akawashiro/akawashiro.github.io.git

ghq get https://github.com/rizsotto/Bear.git

ghq get https://github.com/Rust-GCC/gccrs.git

ghq get https://github.com/awslabs/mountpoint-s3.git

ghq get https://github.com/python/cpython.git

ghq get https://github.com/facebookresearch/llama.git

ghq get git@github.com:akawashiro/ELF2.jl.git

ghq get git@github.com:akawashiro/tensorflow.git
cd $(ghq root)/github.com/akawashiro/tensorflow
git remote add upstream https://github.com/tensorflow/tensorflow
git remote set-url upstream --push no-push

ghq get git@github.com:akawashiro/glibc.git
cd $(ghq root)/github.com/akawashiro/glibc
git remote add upstream  https://sourceware.org/git/glibc.git
git remote set-url upstream --push no-push

ghq get git@github.com:akawashiro/rtld-only-glibc.git

ghq get git@github.com:akawashiro/linux.git
cd $(ghq root)/github.com/akawashiro/linux
git remote add torvalds git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
git remote add focal git://git.launchpad.net/~ubuntu-kernel/ubuntu/+source/linux/+git/focal
git remote add linux-next https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git
git remote set-url torvalds --push no-push
git remote set-url focal --push no-push
git remote set-url linux-next --push no-push

ghq get git@github.com:akawashiro/pytorch.git
cd $(ghq root)/github.com/akawashiro/pytorch
git remote add upstream git@github.com:pytorch/pytorch.git
git remote set-url upstream --push no-push

ghq get git@github.com:akawashiro/ruby.git
cd $(ghq root)/github.com/akawashiro/ruby
git remote add upstream git@github.com:ruby/ruby.git
git remote set-url upstream --push no-push

ghq get https://github.com/llvm/llvm-project.git

ghq get git@github.com:akawashiro/qemu.git
cd $(ghq root)/github.com/akawashiro/qemu
git remote add upstream https://github.com/qemu/qemu
git remote set-url --push upstream no_push

ghq get git@github.com:akawashiro/julia.git
cd $(ghq root)/github.com/akawashiro/julia
git remote add upstream git@github.com:JuliaLang/julia.git
git remote set-url upstream --push no-push

ghq get git@github.com:akawashiro/optuna.git
cd $(ghq root)/github.com/akawashiro/optuna
git remote add upstream git@github.com:optuna/optuna.git
git remote set-url upstream --push no-push

ghq get git@github.com:akawashiro/lmbench.git
cd $(ghq root)/github.com/akawashiro/lmbench
git remote add upstream https://github.com/intel/lmbench.git
git remote set-url upstream --push no-push

ghq get git@github.com:akawashiro/akbench.git
