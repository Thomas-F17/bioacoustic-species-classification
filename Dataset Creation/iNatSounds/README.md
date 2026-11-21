# iNatSounds Spectrogram Generation Pipeline

## Overview

In this README.md file, we document the initial pipeline that we created to 
have a pre-processed dataset ready for our models to use.
This pipeline converts the raw iNaturalist Sounds Dataset (iNatSounds) WAV
recordings into Mel-spectrogram .npy files, this reproduces the exact 
preprocessing described in:

Chasmai, M., Shepard, A., Maji, S., & Van Horn, G. (2024).
**The iNaturalist Sounds Dataset.**
*NeurIPS 2024 Datasets and Benchmarks Track.*
[Paper](https://openreview.net/forum?id=QCY01LvyKm) ·
[Code](https://github.com/cvl-umass/iNatSounds) ·
[Data](https://github.com/visipedia/inat_sounds/tree/main/2024)

## Files

| File | Purpose |
|---|---|
| `run_pipeline.sh` | Main entry point - runs everything end-to-end |
| `generate_spectrograms.py` | Converts WAV -> Mel-spectrogram `.npy` |
| `validate_spectrograms.py` | Verifies output correctness (counts, shapes, dtypes) |
| `README.md` | This file - documentation and justification |

## Usage

We ran this on a EC2 instance on AWS (documentation available here: 
[Get started with Amazon EC2](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)).
After importing the files within this repository folder, the below 
steps describe the process we went through to create the dataset.

On the EC2 (c5.4xlarge Spot Instance in us-east-1 Ubuntu) instance's ubuntu terminal we 
ran:
```bash
bash run_pipeline.sh
```

The script then handles everything: it installs dependencies, then downloads data,
verifies the checksums to make sure we have the right data, extracts the data, 
converts the data, and validates it.

## Justifying our pre-processing steps: From WAV Audio Files to Spectrogram Images 

The below table maps every preprocessing parameter from Section 4 of the
paper ("Model inputs: converting audio to an image") to our implementation
in the `generate_spectrograms.py` file. We also cross-reference against the official
training code at `cvl-umass/iNatSounds` to confirm compatibility.

### The Paper -> Our Code

| Parameter | Paper Quotes (Section 4) | Our Code |
|---|---|---|
| Sample rate | "sample rate of 22.05kHz" | SR = 22050 |
| Window type | "Hann window" | window='hann' |
| Window size | "Hann window size of 512" | WIN_LENGTH = 512 | 
| Stride | "stride length of 128" | HOP_LENGTH = 128 | 
| FFT length | "1024 FFT length" | N_FFT = 1024 | 
| Mel bins | "128 logarithmically spaced mel bins" | N_MELS = 128 |
| Min frequency | "frequencies in the range [50Hz" | FMIN = 50 |
| Max frequency | "11.025kHz]" | FMAX = 11025 |
| Amplitude scale | "magnitude values are converted to decibels" | librosa.power_to_db() |
| Output range | "rescaling the decibel values to span [0, 255]" | min-max then multiply by 255 | 
| Output type | "save the matrix as an uint8 image" | .astype(np.uint8) |

### Additional Details (when working with the paper's official training pipeline)

Since we also use the paper's training pipeline for our CNNs, we detail below some 
key points as to how our pre-processing corresponds to their pipeline code. 

The official training code can be found at `cvl-umass/iNatSounds/src/dataset.py` 
ingests the already generated spectrograms (the .npy files).

Firstly, the `dataset.py` expects numpy arrays as inputs (the .npy format). Further, it 
performs channel duplication so that the ingested grayscale spectrograms (.npy files) 
become RGB images, thus, it expects a two dimensional array (128, T) which is indeed 
what we have generated. Finally, the gray scale spectrograms are resized to 224x224 
images at this stage as well which means our (128, T) dimension size where T is depends 
on the audio length makes sense.
