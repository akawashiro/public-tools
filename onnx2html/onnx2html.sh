#! /bin/bash

VENV_DIR=/tmp/onnx2html-venv
SCRIPT_DIR=$(dirname "$0")
SOURCE_FILE=${SCRIPT_DIR}/onnx2html.py
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip3 install onnx
python3 $SOURCE_FILE "$@"
