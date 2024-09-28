#! /bin/bash
set -eux -o pipefail

SCRIPT_DIR=$(cd $(dirname $0); pwd)
export PYTHONPATH=${HOME}/tmp/pytorch-install/lib/python3.10/site-packages
python3 ${SCRIPT_DIR}/victim.py
