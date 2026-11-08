#!/usr/bin/env bash
# usage: bash scripts/setup_env.sh   (from project root)
set -e
python3 -m pip install -r requirements.txt
python3 scripts/check_env.py
