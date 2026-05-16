#! /bin/bash

set -eux -o pipefail

script_dir=$(realpath $(dirname $0))
python3 -m venv /tmp/photo_sort_venv
source /tmp/photo_sort_venv/bin/activate

pip install pillow
python3 ${script_dir}/photo_sort.py "$@"
