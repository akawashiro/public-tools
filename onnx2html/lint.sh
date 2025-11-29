#! /bin/bash

set -eux -o pipefail

VENV_DIR=/tmp/onnx2html-lint-env
SCRIPT_DIR=$(dirname "$0")
SOURCE_FILE=${SCRIPT_DIR}/onnx2html.py
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip3 install black isort mypy onnx

black $SOURCE_FILE
isort $SOURCE_FILE
mypy --strict $SOURCE_FILE
