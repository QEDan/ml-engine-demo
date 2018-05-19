#!/bin/bash

export TRAINFILE=data/random_linear.csv

gcloud ml-engine local train \
    --module-name model.regression_model \
    --package-path ./model \
    -- --train-file ${TRAINFILE} \
    --export-dir ../saved_models/
