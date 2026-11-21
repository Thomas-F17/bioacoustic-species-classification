# Testing on SSW60

## Overview

In this README.md file, we document how we evaluated our four iNatSounds-trained models 
on the SSW60 audio test set. SSW60 contains 60 bird species with strongly verified 
labels, all of which appear in iNatSounds, this makes it a useful benchmark for measuring 
how our iNatSounds-trained models generalise to a strongly labelled dataset.

We follow the same protocol as Chasmai et al. (2024):

- No fine-tuning: our models are not fine-tuned on SSW60, we directly use them with the 
iNatSounds fine-tuning we conducted.
- Class mask: as per Chasmai et al., logits for the 5,509 species absent from SSW60 are 
set to minus infinity, effectively masking species that are not present in the SSW60 
dataset.
- Sliding-window: the same sliding window methodology that we used for the iNatSounds 
dataset is applied here (3 second windows, 1.5 second stride, softmax averaged across 
windows per recording).
- Reported Metrics: we simply show Top-1 and Top-5 accuracy at the sample-level (exactly 
as Chasmai et al.).

We evaluate four out of our five iNatSounds-trained models because we lost the ConvNeXt 
100% run weights and were not able to re-run the training in time. The four models we 
evaluate here are MobileNet-V3-Large, EfficientNet-B0, ConvNeXt (25% subsample run), and 
Swin-V2-B.

Reference:

Chasmai, M., Shepard, A., Maji, S., & Van Horn, G. (2024).
**The iNaturalist Sounds Dataset.**
*NeurIPS 2024 Datasets and Benchmarks Track.*
[Paper](https://openreview.net/forum?id=QCY01LvyKm) ·
[Code](https://github.com/cvl-umass/iNatSounds)

## Folder Structure

This folder contains only the notebook, the results, and this README. The larger files needed to run the notebook (model weights, spectrograms, and metadata) can be shared if appropriate and authorised:

```
.
├── README.md
├── ssw60_eval.ipynb
└── ssw60_results.json
```

The whole evaluation was small enough to run locally on a personal laptop, so we did not 
need a Thunder Compute instance for this step. The eval loop runs all four models 
sequentially over the 1,264 SSW60 test recordings.

## Importing the Required Files

The four model weights, the SSW60 metadata, the SSW60 test set spectrograms, and the iNatSounds `train.json` can be shared together if appropriate and authorised.

The shared files would include:

- `convnext_epoch49_25pct_run.pt` (ConvNeXt-Base weights, 25% subsample run, epoch 49)
- `mobilenetv3_50epoch_weights.pt` (MobileNet-V3-Large weights, epoch 50)
- `swin_v2_b_epoch050_weights.pt` (Swin-V2-B weights, epoch 50)
- `efficientnet-b0_50epochs_weights.pt` (EfficientNet-B0 weights, epoch 50)
- `ssw60/audio_ml.csv` (SSW60 metadata, including the split and label columns)
- `ssw60/taxa.csv` (SSW60 taxa, used to map SSW60 labels to iNatSounds taxon ids)
- `ssw60_test_spectrograms/` (folder of .npy spectrograms for the 1,264 SSW60 test 
recordings, generated using the pipeline documented in
`Dataset creation/SSW60/`)
- `train.json` (iNatSounds `train.json`, used to map iNatSounds taxon ids to the 5,569 
class indices that the models were trained on)

To run the notebook, download these files and place them in the same directory as 
`ssw60_eval.ipynb`, keeping the directory structure intact (i.e. the `ssw60/` folder and 
the `ssw60_test_spectrograms/` folder must remain as folders alongside the four `.pt` 
files and `train.json`). The expected layout the notebook reads from specified below:

```
.
├── README.md
├── ssw60_eval.ipynb
├── convnext_epoch49_25pct_run.pt
├── mobilenetv3_50epoch_weights.pt
├── swin_v2_b_epoch050_weights.pt
├── efficientnet-b0_50epochs_weights.pt
├── train.json
├── ssw60/
│   ├── audio_ml.csv
│   └── taxa.csv
└── ssw60_test_spectrograms/
    ├── {asset_id}.npy
    ├── {asset_id}.npy
    └── [...]
```

If different paths are preferred, the path constants at the top of `ssw60_eval.ipynb` 
(in the first code cell) can be edited.

## Running the Notebook

Once the files are in place, the workflow is:

```bash
pip install torch torchvision timm tqdm scikit-learn pandas numpy
jupyter notebook ssw60_eval.ipynb
```

Then execute the cells top to bottom. The notebook will:

1. Load the path constants and model configurations.
2. Build the SSW60 class mask by joining `audio_ml.csv` and `taxa.csv` to `train.json` 
so that the 60 SSW60 species map to the correct iNatSounds class indices, with all other 
5,509 indices masked out.
3. Define the `SSW60Dataset`, the sliding-window evaluation function, and `build_model()` 
(which reproduces the exact architectures used during iNatSounds training: torchvision 
for MobileNet and EfficientNet, `timm` for ConvNeXt and Swin-V2-B).
4. Run a quick weight-loading test to confirm each .pt file loads cleanly into their 
corresponding architecture before kicking off the full eval.
5. Loop over the four models, running the sliding-window evaluation on the SSW60 test 
set for each one, and print a final summary table.
6. Save all the results (per-model top-1 and top-5 accuracy plus the configuration that 
produced them) to `ssw60_results.json`.

## Results

Our results on the SSW60 audio test set, evaluated using the protocol described above 
(i.e. sliding-window, masked to the 60 SSW60 species, no fine-tuning, and no test-time 
augmentation):

| Model | Params | Top-1 | Top-5 |
|-------|--------|-------|-------|
| MobileNet-V3-Large | 5.48M | 73.18% | 87.10% |
| EfficientNet-B0 | 5.29M | 74.68% | 89.16% |
| ConvNeXt-Base (25% run) | 88.60M | 70.25% | 85.84% |
| **Swin-V2-B** | **87.92M** | **79.83%** | **92.17%** |

Some key observations:
- Swin-V2-B is the strongest model on SSW60, which is consistent with its iNatSounds 
test set performance (because it also beat all the other models we trained as well as 
the paper baselines).
- ConvNeXt under-performs the two smaller models here (more details in our report)
- The two smaller models, MobileNet-V3-Large and EfficientNet-B0, generalise surprisingly 
well to SSW60 despite their much lower parameter count

## Output

As mentioned, the notebook outputs a single file `ssw60_results.json` at the end of the 
run, which contains the per-model top-1 and top-5 accuracy on the SSW60 test set, as well 
as the the configuration.

## Notes

- We use the exact same sliding-window protocol as for the iNatSounds test evaluation to 
maintain comparability with the iNatSounds paper's methodology.
- The ConvNeXt result here is from the 25% subsample run because, as mentioned, we lost 
the 100% run weights, this potentially means our ConvNeXt result is a slight 
underestimate of what a full ConvNeXt run would have achieved.
