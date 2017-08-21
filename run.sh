#!/bin/bash

set -x
set -e

LOG="logs/log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
time $*

set +x

echo -e "\a"
