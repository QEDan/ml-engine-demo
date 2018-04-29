#!/bin/bash

export JOBNAME=TEST4
export JOBDIR=local-job-dir
export TRAINFILE=data/random_linear.csv
export CONFIGFILE=hp-tuning-config.yaml


gcloud ml-engine local train \
    --module-name model.regression_model \
    --package-path ./model \
    -- --train-file $TRAINFILE \
    --export-dir ../saved_models/
