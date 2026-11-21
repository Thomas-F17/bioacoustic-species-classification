# Dataset creation

Here we have the scripts and notebooks that turn the raw audio data into the 
Mel-spectrogram .npy files that our models train and evaluate on.

## Layout

```
Dataset creation/
    generate_spectrograms.py # shared file between pipelines which converts WAV to (128, T) uint8 .npy
    README.md # this file
    iNatSounds/ # iNatSounds Dataset pipeline (train/val/test)
        README.md
        run_pipeline.sh
        validate_spectrograms.py
    SSW60/ # SSW60 Dataset pipeline (test)
        README_SSW60.md
        ssw60_creation.ipynb
```

`generate_spectrograms.py` is shared between the two pipelines because 
both datasets use the same preprocessing recipe (Section 4 of the iNatSounds
paper).

## Per-dataset documentation

- **iNatSounds:** see [`iNatSounds/README.md`](iNatSounds/README.md)
- **SSW60:** see [`SSW60/README_SSW60.md`](SSW60/README_SSW60.md)

## Reference to Original Datasets

- [**iNatSounds**](https://github.com/visipedia/inat_sounds/tree/main/2024)
- [**SSW60**](https://github.com/visipedia/ssw60)
