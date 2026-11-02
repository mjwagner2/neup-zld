#!/bin/bash
mkdir -p results logs
python3 neup_zld/htc_worker.py "$@"
