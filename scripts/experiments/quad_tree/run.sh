#!/bin/bash

# BASE="/gpfs/milgram/scratch60/yildirim/meb266/tmp"
BASE="/gpfs/radev/project/yildirim/meb266/GranularScenes/env.d/tmp"

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
  TDIR="${BASE}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "Using slurm array info at $TDIR"
  mkdir "$TDIR"
else
  TDIR="$(mktemp -d -p ${BASE})"
  echo "Using random tmp dir at $TDIR"
fi

export HOME="$TDIR"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
julia "${SCRIPT_DIR}/run.jl" "$@"
