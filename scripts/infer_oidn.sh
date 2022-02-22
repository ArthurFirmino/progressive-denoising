#!/bin/bash

./oidn/build/oidnSure --input-dir data/interim/rt_test --output-dir data/sure/rt_test
./oidn/build/oidnSure --input-dir data/interim/rt_test --output-dir data/sure/rt_test --use-aov 1

./oidn/build/oidnSure --input-dir data/interim/rt_train --output-dir data/sure/rt_train
./oidn/build/oidnSure --input-dir data/interim/rt_train --output-dir data/sure/rt_train --use-aov 1

./oidn/build/oidnSure --input-dir data/interim/rt_valid --output-dir data/sure/rt_valid
./oidn/build/oidnSure --input-dir data/interim/rt_valid --output-dir data/sure/rt_valid --use-aov 1
