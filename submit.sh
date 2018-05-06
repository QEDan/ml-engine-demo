#!/bin/bash

export JOBNAME=TEST015
export JOBBUCKET=gs://productionml-mlengine-demo
export JOBDIR=${JOBBUCKET}/job-dir/${JOBNAME}/
export JOBREGION=us-central1
export TRAINFILE=${JOBBUCKET}/data/random_linear.csv
export CONFIGFILE=hp-tuning-config.yaml


gcloud ml-engine jobs submit training $JOBNAME \
    --job-dir $JOBDIR \
    --module-name model.regression_model \
    --package-path ./model \
    --region $JOBREGION \
    --runtime-version 1.6 \
    --config ${CONFIGFILE}\
    -- --train-file $TRAINFILE \
    --export-dir ${JOBBUCKET}/saved_models/${JOBNAME}/
