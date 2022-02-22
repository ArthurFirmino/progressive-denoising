#!/bin/bash

# Create hardlinks of data/raw images in data/interim, and split 5% of the dataset for validation
##################################################################################################
ls data/raw/training | grep -E "[0-9]{3}0[0-4]_" | \
  xargs -I {} ln -f data/raw/training/{} data/interim/rt_valid/{}
ls data/raw/training | grep -Ev "[0-9]{3}0[0-4]_" | \
  xargs -I {} ln -f data/raw/training/{} data/interim/rt_train/{}

# Preprocess the dataset
##################################################################################################
python3 oidn/training/preprocess.py hdr alb nrm var -f RT \
  -D data/interim -P data/preproc -t rt_train -v rt_valid \
  -X '{"hdr":"log", "alb":"log", "var":"atan"}'

# Train the denoisers
##################################################################################################
feature_sets=("hdr" "hdr var" "hdr alb nrm" "hdr alb nrm var")
for feature_set in "${feature_sets[@]}"; do
  python3 oidn/training/train.py \
    $feature_set -f RT -P data/preproc -R models -t rt_train -v rt_valid -r rt_${feature_set// /_} \
    --epochs 2500 --save_epochs 500 --max_lr 5e-5 --bs 12 -l smape \
    -X '{"hdr":"log", "alb":"log", "var":"atan"}'
done