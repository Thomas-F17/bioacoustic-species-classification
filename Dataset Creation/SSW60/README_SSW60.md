# SSW60 Spectrogram Generation Pipeline

## Overview

In this README.md file, we document the pipeline that we used to prepare the
Sapsucker Woods 60 (SSW60) dataset as a downstream evaluation set for
models trained on iNatSounds. Indeed, SSW60 contains 60 bird species with strongly
verified labels, all of which appear in iNatSounds. This makes it a natural
benchmark for measuring how iNatSounds-trained models generalise to a strongly labelled 
dataset.

We only use test split of SSW60 this is because, as per the iNatSounds paper, the same 
protocol as iNatSounds can be used for evaluation on SSW60 i.e. 
predictions over 3-second windows (strided by 1.5 seconds) from each test
sample to arrive at file-level scores.

The preprocessing recipe is also identical to the one we use for iNatSounds
(see the iNatSounds spectrogram README). We have done this intentionally because within 
the iNatSounds paper, they apply the same audio-to-image conversion to SSW60 so that  
models trained on the iNatSounds dataset only transfer cleanly without need for 
retraining.

Our References:

Chasmai, M., Shepard, A., Maji, S., & Van Horn, G. (2024).
**The iNaturalist Sounds Dataset.**
*NeurIPS 2024 Datasets and Benchmarks Track.*
[Paper](https://openreview.net/forum?id=QCY01LvyKm) ·
[Code](https://github.com/cvl-umass/iNatSounds)

Van Horn, G., Qian, R., Wilber, K., Adam, H., Mac Aodha, O., & Belongie, S. (2022).
**Exploring Fine-Grained Audiovisual Categorization with the SSW60 Dataset.**
*ECCV 2022.*
[Paper](https://arxiv.org/abs/2207.10664) ·
[Data](https://github.com/visipedia/ssw60)

## Files

| File | Purpose |
|---|---|
| `ssw60_creation.ipynb` | Google Colab notebook that runs the full pipeline |
| `generate_spectrograms.py` | Converts WAV -> Mel-spectrogram `.npy` (reused unchanged from iNatSounds pipeline) |
| `README_SSW60.md` | This file |

## Usage

We ran the pipeline on Google Colab Free. We then transferred the data to Google Drive 
once the spectrogram generation is done.

The whole pipeline runs from `ssw60_creation.ipynb`, this can be opened in Colab and 
then run from start to finish (provided that `generate_spectrograms.py` is present) to 
create the spectrograms. The notebook does the following:
downloads the SSW60 archive (c.25 min on Colab's connection), extracts only the audio 
files, checks the file count and split distribution, filters to the 1264 test-split 
files, and finally, generates the spectrograms.

### Output structure

In this case, contrary to iNatSounds (where spectrograms are nested under `train/`, `val/`,
`test/` and further by category), SSW60 spectrograms simply have one folder:

```
ssw60_spectrograms/
    {asset_id}.npy
    {asset_id}.npy
    [...]
```

### Compatibility iNatSounds Models

We reuse `generate_spectrograms.py` from our iNatSounds pipeline.
This ensures that SSW60 spectrograms are consistent with what our iNatSounds trained 
models expect in terms of inputs.
