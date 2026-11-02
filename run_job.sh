#!/bin/bash
mkdir -p results logs
python3 neup_zld/core/htc_worker.py "$@"
