# Improving Bioacoustic Species Classification on the iNaturalist Sounds Dataset with Modern CNN and Transformer Architectures

This repository contains our group project for ST456 Deep Learning (LSE, WT 2026). We 
extend the iNaturalist Sounds Dataset paper (Chasmai et al., 2024) by benchmarking four 
additional backbone architectures on the iNatSounds species classification task.

## Background

The iNatSounds dataset (NeurIPS 2024, Datasets and Benchmarks track) provides 232,237 
audio recordings from 5,569 species, sourced from the iNaturalist citizen science 
platform. The original paper converts audio to mel spectrograms and classifies species 
using three ImageNet-1k pretrained vision backbones: MobileNet V3 Large (6.24M 
parameters), ResNet 50 (26.77M) and ViT B 16 (87.02M). Their best model without geographic 
priors achieves 53.6% class averaged Top 1 accuracy on the test set.

We investigate whether modern convolutional and transformer architectures can improve on 
these baselines. All models follow the same audio pre-processing pipeline using 
spectrograms and training protocol described in the original paper to ensure fair 
comparison.

## Our Architectures

| Model | Parameters* | Pretraining | Key Architectural Feature |
|-------|-----------|-------------|--------------------------|
| MobileNet V3 Large (ours) | 5.48M | ImageNet 1K | Inverted residuals, h swish, NAS optimised |
| EfficientNet B0 | 5.29M | ImageNet 1K | Compound scaling, squeeze and excitation |
| ConvNeXt Base | 88.59M | ImageNet 22K to 1K | Modernised CNN with 7x7 depthwise, LayerNorm, GELU |
| Swin V2 B | 87.92M | ImageNet 22K to 1K | Shifted window attention, scaled cosine attention |

**note: we report model parameter counts with the ImageNet-1k classification for comparability to the original paper*

## Results

### iNatSounds Test Set (sound avg, class averaged, no geo priors)

| Model | Params | Data | mAP | mF1 | Top 1 | Top 5 |
|-------|--------|------|-----|-----|-------|-------|
| MobileNet V3 (paper) | 6.24M | 100% | 60.00 | 63.20 | 49.10 | 69.50 |
| ResNet 50 (paper) | 26.77M | 100% | 62.80 | 65.50 | 52.60 | 74.30 |
| ViT B 16 (paper) | 87.02M | 100% | 64.40 | 66.90 | 53.60 | 74.30 |
| MobileNet V3 (ours) | 5.48M | 100% | 56.03 | 60.58 | 49.42 | 69.70 |
| EfficientNet B0 | 5.29M | 100% | 66.74 | 68.60 | 55.28 | 76.56 |
| Swin V2 B | 87.92M | 25%* | 70.30 | 71.50 | 60.90 | 81.40 |
| ConvNeXt Base | 88.59M | 25%* | 69.50 | 71.00 | 58.50 | 79.00 |
| **ConvNeXt Base** | **88.59M** | **100%** | **74.35** | **74.93** | **62.79** | **81.94** |

*25% denotes a fresh random 25% subset of training recordings drawn each epoch. All other 
models saw 100% of the data in each epoch.

### SSW60 Downstream Transfer (sample level accuracy, no fine tuning)

| Model | iNat Data | Top 1 | Top 5 |
|-------|-----------|-------|-------|
| MobileNet V3 (paper) | 100% | 66.9 | 83.9 |
| ResNet 50 (paper) | 100% | 68.1 | 83.2 |
| ViT B 16 (paper) | 100% | 70.9 | 85.8 |
| ConvNeXt Base | 25%* | 70.3 | 85.8 |
| MobileNet V3 (ours) | 100% | 73.2 | 87.1 |
| EfficientNet B0 | 100% | 74.7 | 89.2 |
| **Swin V2 B** | **25%*** | **79.8** | **92.2** |

### Key Findings

ConvNeXt Base trained on 100% of the data achieves the highest in distribution performance 
across all metrics: 74.35% mAP and 62.79% Top 1, surpassing the paper's best baseline 
(ViT B 16) by +9.95 mAP and +9.19 Top 1 at a comparable parameter count.

EfficientNet B0 delivers the strongest performance per parameter ratio on both benchmarks, 
with 66.74% mAP on iNatSounds and 74.7% Top 1 on SSW60, using only 5.3M parameters.

On the SSW60 downstream transfer task, the ranking diverges from iNatSounds. Swin V2 B 
dominates with 79.8% Top 1 despite training on only 25% of the data per epoch, outperforming 
every model trained on 100% by at least 5 points. This confirms that in distribution 
accuracy alone is insufficient for model selection, as the original paper warned.

## Repository Structure

```
.
├── README.md                       # This file
├── ST456_AI_Use_Statement.pdf
├── Dataset Creation/
│   ├── README.md                   # Overview of both data pipelines
│   ├── generate_spectrograms.py    # Shared WAV to mel spectrogram pipeline
│   ├── iNatSounds/                 # iNatSounds spectrogram generation pipeline
│   └── SSW60/                      # SSW60 spectrogram generation pipeline
├── Models/
│   ├── README.md                   # Overview of model training runs
│   ├── parameter_counts.ipynb      # Notebook computing parameter counts for each model
│   ├── MobileNetV3-Large/          # MobileNet V3 training run, weights, results
│   ├── EfficientNet-B0/            # EfficientNet B0 training run, weights, results
│   ├── ConvNeXt/                   # ConvNeXt Base 100% and 25% training runs
│   ├── Swin-V2-B/                  # Swin V2 B training run, weights, results
│   └── Testing on SSW60/           # SSW60 downstream evaluation for all models
└── Report/
    └── ST456_Visionaries_Final_Report.pdf
```

Each model folder contains a README with full instructions for reproducing the training 
run, including exact commands, hyperparameters, and links to pretrained weights.

## Reproducing Our Results

The project is organised into two stages:

1. **Data preparation.** The `Dataset Creation/` folder contains the pipeline that 
   converts raw WAV recordings into mel spectrograms (the format expected by all our 
   models). The same pipeline is applied to both iNatSounds and SSW60 to ensure that 
   models trained on iNatSounds transfer cleanly to SSW60 without any retraining. See 
   `Dataset Creation/README.md` for details.

2. **Model training and evaluation.** The `Models/` folder contains one subfolder per 
   architecture, each with the training notebook (or training script and command), 
   pretrained weights, and per-model results. The `Testing on SSW60/` subfolder contains 
   the joint downstream evaluation that takes the four iNatSounds-trained checkpoints 
   and evaluates them on the SSW60 test set without any fine-tuning. See `Models/README.md` 
   for an overview, then the individual model READMEs for setup commands.

For a quick start without re-running training, the model weights are linked from each 
model's README, and the SSW60 evaluation can be reproduced from those checkpoints alone.

## Training Infrastructure

Training was performed on Thunder Compute instances:
- MobileNet V3, EfficientNet B0, and ConvNeXt Base were trained on a single NVIDIA A100 (80GB).
- Swin V2 B was trained on a single NVIDIA RTX A6000 (48GB), with batch size 64 due to the 
  48GB memory limit at 256 by 256 input resolution.
- The SSW60 downstream evaluation was small enough to run locally on a laptop.

Training times ranged from approximately 8 hours (Swin V2 B, 50 epochs) to approximately 
48 hours (ConvNeXt Base 100%, 50 epochs) depending on model size and the data fraction 
used per epoch.

## Limitations

All of our results are from single seed runs; the original paper reports mean and standard 
deviation across three seeds. Geographical prior evaluation was not performed for any of 
our models due to the need for an additional model (the paper trained a SINR model for 
this task). Swin V2 B and ConvNeXt Base (25%) were trained on a random 25% subset per 
epoch rather than the full dataset, due to computational constraints. We additionally 
trained ConvNeXt Base on the full dataset and report its iNatSounds results, however 
the weights were corrupted before SSW60 evaluation, so this run does not appear in 
the SSW60 table.

## Key Reference

1. Chasmai, M., Shepard, A., Maji, S., and Van Horn, G. "The iNaturalist Sounds Dataset." *NeurIPS Datasets and Benchmarks Track*, 2024. [arXiv:2506.00343](https://arxiv.org/abs/2506.00343)

## Team

ST456 Deep Learning, LSE, Winter Term 2026, Group: Visionaries


## Licence

This repository uses different licences for different types of material:

- **Code:** the Python, shell and notebook source code is licensed under the **MIT License**; see `LICENSE`.
- **Report and documentation:** the project report, README files and supporting explanatory documents are licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**; see `LICENSE-CC-BY-NC-4.0.txt`.
- **Results and plots:** plots, tables and result text are also covered by CC BY-NC 4.0 unless otherwise stated.

These licences apply only to material created by the project team. They do **not** grant any rights to third-party software, pretrained model weights, datasets, media, annotations or derived dataset artefacts.

## Dataset Terms and Restrictions

No iNatSounds or SSW60 recordings, spectrogram archives, metadata, split JSON files or model checkpoints trained from those data are included in this repository. The dataset and media remain subject to their original terms:

- iNatSounds use is restricted to non-commercial research and educational purposes, recordings must not be distributed, and users must abide by the iNaturalist Terms of Service.
- SSW60 media are to be used for computer-vision research purposes and are not to be redistributed or used for other non-research purposes.

Anyone obtaining iNatSounds or SSW60 data or generated artefacts must obtain them through the applicable official process and comply with the current terms supplied by the data owners. Any model weights, datasets or spectrograms described as available elsewhere are subject to separate authorization and must not be treated as licensed by this repository.

The iNaturalist terms are available at <https://www.inaturalist.org/pages/terms>. A set of SSW60 Terms of Use could not be located separately; users should use the terms supplied with the SSW60 dataset and contact the dataset providers when in doubt.
