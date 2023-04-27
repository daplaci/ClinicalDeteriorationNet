#! /usr/bin/env bash

spec=("qsub -l nodes=1:ppn={threads},"
      "mem={resources.vmem}mb,"
      "walltime={resources.tmin}:00"
      " -j eo -e .snakemake_log/")

call=$(printf "%s" "${spec[@]}")

snakemake $@ -p --jobs 20 --notemp --verbose \
    --cluster "$call" --latency-wait 35 --cluster-status "python qsub-status.py"
