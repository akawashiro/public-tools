#! /bin/bash -ux

FILTER="$1"

repos=(
  "git@github.com:akawashiro/misc.git"
  "git@github.com:akawashiro/sold.git"
  "git@github.com:akawashiro/ros3fs.git"
  "git@github.com:akawashiro/jendeley.git"
  "git@github.com:akawashiro/sloader.git"
  "git@github.com:akawashiro/z3shape.git"
  "git@github.com:akawashiro/xla2onnx.git"
  "git@github.com:akawashiro/akawashiro.github.io.git"
  "https://github.com/rizsotto/Bear.git"
  "https://github.com/Rust-GCC/gccrs.git"
  "https://github.com/awslabs/mountpoint-s3.git"
  "https://github.com/python/cpython.git"
  "https://github.com/facebookresearch/llama.git"
  "git@github.com:akawashiro/ELF2.jl.git"
  "git@github.com:akawashiro/tensorflow.git"
  "git@github.com:akawashiro/glibc.git"
  "git@github.com:akawashiro/rtld-only-glibc.git"
  "git@github.com:akawashiro/linux.git"
  "git@github.com:akawashiro/pytorch.git"
  "git@github.com:akawashiro/ruby.git"
  "https://github.com/llvm/llvm-project.git"
  "git@github.com:akawashiro/qemu.git"
  "git@github.com:akawashiro/julia.git"
  "git@github.com:akawashiro/optuna.git"
  "git@github.com:akawashiro/lmbench.git"
  "git@github.com:akawashiro/akbench.git"
  "git@github.com:akawashiro/renlog.git"
  "https://github.com/huggingface/transformers.git"
  "git@github.com:akawashiro/nano-vllm.git"
)

repo_match() {
  url="$1"
  # git@github.com:akawashiro/foo.git → foo
  # https://github.com/foo/bar.git → bar
  echo "$url" | sed -E 's/.*\/([^/]+)\.git$/\1/'
}

for repo in "${repos[@]}"; do
  name=$(repo_match "$repo")
  if [[ -z "$FILTER" || "$name" == *"$FILTER"* ]]; then
    ghq get "$repo"

    case "$name" in
      tensorflow)
        cd "$(ghq root)/github.com/akawashiro/tensorflow"
        git remote add upstream https://github.com/tensorflow/tensorflow || true
        git remote set-url upstream --push no-push
        ;;
      glibc)
        cd "$(ghq root)/github.com/akawashiro/glibc"
        git remote add upstream https://sourceware.org/git/glibc.git || true
        git remote set-url upstream --push no-push
        ;;
      linux)
        cd "$(ghq root)/github.com/akawashiro/linux"
        git remote add torvalds git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git || true
        git remote add focal git://git.launchpad.net/~ubuntu-kernel/ubuntu/+source/linux/+git/focal || true
        git remote add linux-next https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git || true
        git remote set-url torvalds --push no-push
        git remote set-url focal --push no-push
        git remote set-url linux-next --push no-push
        ;;
      pytorch)
        cd "$(ghq root)/github.com/akawashiro/pytorch"
        git remote add upstream git@github.com:pytorch/pytorch.git || true
        git remote set-url upstream --push no-push
        ;;
      ruby)
        cd "$(ghq root)/github.com/akawashiro/ruby"
        git remote add upstream git@github.com:ruby/ruby.git || true
        git remote set-url upstream --push no-push
        ;;
      qemu)
        cd "$(ghq root)/github.com/akawashiro/qemu"
        git remote add upstream https://github.com/qemu/qemu || true
        git remote set-url --push upstream no_push
        ;;
      julia)
        cd "$(ghq root)/github.com/akawashiro/julia"
        git remote add upstream git@github.com:JuliaLang/julia.git || true
        git remote set-url upstream --push no-push
        ;;
      optuna)
        cd "$(ghq root)/github.com/akawashiro/optuna"
        git remote add upstream git@github.com:optuna/optuna.git || true
        git remote set-url upstream --push no-push
        ;;
      lmbench)
        cd "$(ghq root)/github.com/akawashiro/lmbench"
        git remote add upstream https://github.com/intel/lmbench.git || true
        git remote set-url upstream --push no-push
        ;;
      nano-vllm)
        cd "$(ghq root)/github.com/akawashiro/nano-vllm"
        git remote add upstream https://github.com/GeeeekExplorer/nano-vllm.git || true
        git remote set-url upstream --push no-push
        ;;
    esac

    cd "$OLDPWD" || true
  fi
done
