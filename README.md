# Progressive Denoising of Monte Carlo Rendered Images

## (To be presented in Computer Graphics Forum, EG 2022)

---

### Abstract
Image denoising based on deep learning has become a powerful tool to accelerate Monte Carlo rendering. Deep learning techniques can produce smooth images using a low sample count. Unfortunately, existing deep learning methods are biased and do not converge to the correct solution as the number of samples increase. In this paper, we propose a progressive denoising technique that aims to use denoising only when it is beneficial and to reduce its impact at high sample counts. We use Stein's unbiased risk estimate (SURE) to estimate the error in the denoised image, and we combine this with a neural network to infer a per-pixel mixing parameter. We further augment this network with confidence intervals based on classical statistics to ensure consistency and convergence of the final denoised image. Our results demonstrate that our method is consistent and that it improves existing denoising techniques. Furthermore, it can be used in combination with existing high quality denoisers to ensure consistency. In addition to being asymptotically unbiased, progressive denoising is particularly good at preserving fine details that would otherwise be lost with existing denoisers.

---

### Links

* Paper: [pending]
* Supplement: [pending]
* Results Viewer: [pending]

---

### Dependencies (Tested only on Debian 11)

+ python (>=3.7, tested 3.9.2)
+ numpy (>=1.8, tested 1.21.0)
+ pytorch (tested 1.10.1+cu113)
+ tqdm (tested 4.61.1)
+ python3-openimageio (>=2.1, tested 2.2.10)

### Data

---

The following is made available on [Google Drive](https://drive.google.com/drive/folders/1-8sBlpvkKVkwrZsZqnPQbU1ozrUXSxOo):

+ 3 pre-trained models, used to run the example (`/models/*.zip`)
+ Sample test input data, used to run the example (`/data/\[interim|sure\]/rt_test.zip`)
+ The entire training dataset (`/data/raw/training.zip`, ~3GB)
+ The entire testing dataset (`/data/raw/testing.zip`, ~3GB)

`.zip` files must be extracted to their respective folders.

---

### Code & Scripts

The code builds upon that of [OpenImageDenoise](https://github.com/OpenImageDenoise/oidn), and all changes are contained in the `oidn_patch/pd.patch` file, which can be applied as follows:

```
# Clone this repo with --recurse-submodules switch
git clone --recurse-submodules https://github.com/ArthurFirmino/progressive-denoising
cd progressive-denoising

# Apply the patch file to oidn
cd oidn
git apply ../oidn_patch/pd.patch 
cd ..
```

To run the example with the sample test input data, run the following commands after downloading and extracting the sample test input data and pre-trained models above:

```
# Denoise and apply Progressive Denoising
bash scripts/infer_denoiser.sh
bash scripts/infer_progressive.sh

python3 scripts/analysis.py --models rt_hdr_alb_nrm_var pd_hdr_alb_nrm_var input --metrics rmse smape

# Apply Progressive Denoising to OIDN
bash scripts/infer_oidn_progressive.sh

python3 scripts/analysis.py --models oidn_alb_nrm pd_oidn_alb_nrm input --metrics rmse smape
```

The remaining scripts are self-documenting, and intended to be run in the following order: `train_denoiser.sh`; `infer_denoiser.sh`; `train_progressive.sh`; `infer_progressive.sh`.

The directory structure is as follows:

* `data/raw/[testing|training]`, the raw rendered images (as multi-channel `.exr` image files).
* `data/interim/[rt_train|rt_valid|rt_test]`, links to the image files and additional channels not present in the raw images (`.den.exr` and `.sure.exr` for the denoised and SURE images respectively).
* `data/sure/[rt_train|rt_valid|rt_test]`, the inferred output and computed SURE of the denoisers.
* `data/preproc`, pre-processed data for training.
* `models`, the directory containing trained models.
* `output/rt_test`, the directories containing the output test data.


**Note:** To avoid making unnecessary copies of image files, the scripts here make extensive usage of hard links between files, using the `ln` command.

**Note**: Running `infer_oidn.sh` requires building `oidn` after applying the patch file, where `oidn/apps/oidnSure` is a small program used to denoise and compute SURE with the OIDN denoiser.

**Note**: To use the [FLIP](https://github.com/NVlabs/flip) error metric in the analysis, set `PATH_TO_FLIP_CUDA` in `scripts/analysis.py` to the path to the build `flip-cuda` program.

---

### Citation

[pending]
