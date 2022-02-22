#!/bin/bash

feature_sets=("hdr" "hdr var" "hdr alb nrm" "hdr alb nrm var")
for feature_set in "${feature_sets[@]}"; do

  # Create hardlinks of data/sure images in data/interim
  #####################################
  ls data/sure/rt_valid | grep \.rt_${feature_set// /_}\. | cut -d "." -f 1 | xargs -I {} sh -c \
    "ln -f data/sure/rt_valid/{}.rt_${feature_set// /_}.hdr.exr data/interim/rt_valid/{}.den.exr &&
     ln -f data/sure/rt_valid/{}.rt_${feature_set// /_}.sure.exr data/interim/rt_valid/{}.sure.exr"
  ls data/sure/rt_train | grep \.rt_${feature_set// /_}\. | cut -d "." -f 1 | xargs -I {} sh -c \
    "ln -f data/sure/rt_train/{}.rt_${feature_set// /_}.hdr.exr data/interim/rt_train/{}.den.exr &&
     ln -f data/sure/rt_train/{}.rt_${feature_set// /_}.sure.exr data/interim/rt_train/{}.sure.exr"

  # Preprocess the dataset
  #####################################
  rm -rf data/preproc/*
  python3 oidn/training/preprocess.py hdr den var sure -f RT \
    -D data/interim -P data/preproc -t rt_train -v rt_valid \
    -X '{"hdr":"linear", "den":"linear", "var":"atan", "sure":"atan"}'

  # Train the progressive denoiser
  #####################################
    python3 oidn/training/train.py \
    hdr den var sure -f RT -P data/preproc -R models -t rt_train -v rt_valid -r pd_${feature_set// /_} \
    --epochs 2500 --save_epochs 500 --max_lr 2e-5 --lr_warmup 0.05 --bs 12 -l smape --model pdnet \
    -X '{"hdr":"linear", "den":"linear", "var":"atan", "sure":"atan"}'

done