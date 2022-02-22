#!/bin/bash

# Create hardlinks of data/raw images in data/interim
##################################################################################################
ls data/raw/testing | \
  xargs -I {} ln -f data/raw/testing/{} data/interim/rt_test/{}

# Infer the testing and training datasets, and compute SURE
##################################################################################################
feature_sets=("hdr alb nrm var")
for feature_set in "${feature_sets[@]}"; do
  python3 oidn/training/infer.py \
    -D data/interim -i rt_valid -R models -r rt_${feature_set// /_} \
    -M mse -F exr -O data/sure --compute-sure 1 --sure-mc-iterations 4

  python3 oidn/training/infer.py \
    -D data/interim -i rt_train -R models -r rt_${feature_set// /_} \
    -M mse -F exr -O data/sure --compute-sure 1 --sure-mc-iterations 4

  python3 oidn/training/infer.py \
    -D data/interim -i rt_test -R models -r rt_${feature_set// /_} \
    -M mse -F exr -O data/sure --compute-sure 1 --sure-mc-iterations 4

  python3 oidn/training/infer.py \
    -D data/interim -i rt_test -R models -r rt_${feature_set// /_} \
    -M mse -F exr -O output
done


