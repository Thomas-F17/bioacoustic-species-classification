# MobileNet-V3 Baseline: iNatSounds

This documents the MobileNet-V3 baseline training run on the
iNaturalist Sounds Dataset (iNatSounds, NeurIPS 2024).

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/rubyroobs/iNatSounds
cd iNatSounds
pip install torch torchvision torchaudio numpy scikit-learn tqdm matplotlib pandas Pillow opencv-python
sudo apt-get install -y libgl1
```

## Modification Made

In `main.py`, both GeoModel instantiations (in `run_train` and `run_eval`)
were replaced with:

```python
geo_model = None
```

This disables geographic prior filtering, which requires separate pretrained
weights not needed for the baseline. No other files were modified.

## Training Command

```bash
python3 main.py \
    --model mobilenet \
    --pretrained \
    --spectrogram_dir ~/data/spectrograms \
    --json_dir ~/data \
    --batch_size 128 \
    --exp_name mobilenet_baseline
```

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Model | MobileNet-V3-Large (5.48M parameters)|
| Pretrained weights | ImageNet-1K V1 (torchvision default) |
| Epochs | 50 |
| Batch size | 128 |
| Optimiser | Nesterov SGD |
| Learning rate | 0.05 |
| Weight decay | 1e-5 |
| LR schedule | 5-epoch linear warmup + cosine annealing |

## Output

Weights and logs are saved to:

```
logs/mobilenet_baseline/pretrained/mobilenet/nesterov_b64_lr0.05_wd1e-05/train/0/
```

Checkpoints are saved every 10 epochs to the `checkpoints/` subfolder.

## Final Weights

The final weights can be shared if appropriate and authorised.

