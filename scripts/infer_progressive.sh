#!/bin/bash

# Infer the testing dataset
##################################################################################################
feature_sets=("hdr alb nrm var")
for feature_set in "${feature_sets[@]}"; do

  # Create hardlinks of data/sure images
  ######################################
  ls data/sure/rt_test | grep \.rt_${feature_set// /_}\. | cut -d "." -f 1 | xargs -I {} sh -c \
    "ln -f data/sure/rt_test/{}.rt_${feature_set// /_}.hdr.exr data/interim/rt_test/{}.den.exr &&
     ln -f data/sure/rt_test/{}.rt_${feature_set// /_}.sure.exr data/interim/rt_test/{}.sure.exr"


  python3 oidn/training/infer.py \
    -D data/interim -i rt_test -R models -r pd_${feature_set// /_} \
    -M mse -F exr -O output
done


