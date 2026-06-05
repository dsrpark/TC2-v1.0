#!/bin/bash
# Run TC2 detection for CMIP6 models.
# Thin wrapper — delegates to run_detect.py with CMIP6 config.
#
# IMPORTANT: run_detect.py runs with tracker Python.
# Classifier subprocess uses glibc/pyenv from config.json internally.
#
# Usage:
#   PYTHONUNBUFFERED=1 nohup bash run_all.sh > run_all.log 2>&1 &

PYTHON=/data4/TOOL/environments/python/current/bin/python3
RUNNER=/data4/TOOL/event_detection/TC/TC2/release/tracker/run_detect.py
CONFIG=/data3/DERIVED/ASCII/TC/CMIP6/_config/TC2/config.json

echo "Starting all CMIP6 TC2 detection $(date)"
$PYTHON $RUNNER --config $CONFIG --all --all-scenarios
echo "Complete $(date)"
