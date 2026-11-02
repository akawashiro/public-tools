#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
fetch() {
    local name="$1" checksum="$2" url="$3"
    if [[ -f "models/$name" ]] && printf '%s  %s\n' "$checksum" "models/$name" | sha256sum --check --status; then
        return
    fi
    local temporary
    temporary=$(mktemp "models/$name.XXXXXX")
    if curl --fail --location --retry 3 --output "$temporary" "$url" &&
       printf '%s  %s\n' "$checksum" "$temporary" | sha256sum --check; then
        mv "$temporary" "models/$name"
    else
        rm -f "$temporary"
        return 1
    fi
}
fetch stories15M.bin cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a \
    https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
fetch tokenizer.bin 50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361 \
    https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
