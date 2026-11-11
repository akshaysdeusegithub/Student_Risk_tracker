#!/usr/bin/env bash
# One-click launcher for macOS / Linux
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
