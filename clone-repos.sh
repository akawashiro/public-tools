#! /bin/bash -ux

ghq get https://github.com/akawashiro/misc.git

ghq get https://github.com/akawashiro/sold.git

ghq get https://github.com/akawashiro/ros3fs.git

ghq get https://github.com/akawashiro/jendeley.git

ghq get https://github.com/akawashiro/sloader.git

ghq get https://github.com/akawashiro/z3shape.git

ghq get https://github.com/akawashiro/xla2onnx.git

ghq get https://github.com/akawashiro/akawashiro.github.io.git

ghq get https://github.com/rizsotto/Bear.git

ghq get https://github.com/Rust-GCC/gccrs.git

ghq get https://github.com/awslabs/mountpoint-s3.git

ghq get https://github.com/python/cpython.git

ghq get https://github.com/facebookresearch/llama.git

ghq get https://github.com/akawashiro/tensorflow.git
cd $(ghq root)/github.com/akawashiro/tensorflow
git remote add upstream https://github.com/tensorflow/tensorflow
git remote set-url upstream --push no-push

ghq get https://github.com/akawashiro/glibc.git
cd $(ghq root)/github.com/akawashiro/glibc
git remote add upstream  https://sourceware.org/git/glibc.git
git remote set-url upstream --push no-push

ghq get git@github.com:akawashiro/rtld-only-glibc.git

ghq get https://github.com/akawashiro/linux.git
cd $(ghq root)/github.com/akawashiro/linux
git remote add torvalds git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
git remote add focal git://git.launchpad.net/~ubuntu-kernel/ubuntu/+source/linux/+git/focal
git remote add linux-next https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git
git remote set-url torvalds --push no-push
git remote set-url focal --push no-push
git remote set-url linux-next --push no-push

ghq get https://github.com/akawashiro/pytorch.git
cd $(ghq root)/github.com/akawashiro/pytorch
git remote add upstream git@github.com:pytorch/pytorch.git
git remote set-url upstream --push no-push

ghq get https://github.com/akawashiro/ruby.git
cd $(ghq root)/github.com/akawashiro/ruby
git remote add upstream git@github.com:ruby/ruby.git
git remote set-url upstream --push no-push

ghq get https://github.com/llvm/llvm-project.git
