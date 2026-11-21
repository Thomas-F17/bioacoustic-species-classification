"""
generate_spectrograms.py: iNatSounds Mel-Spectrogram Generation

We developed this python script to create the same audio pre-processing pipeline
as Chasmai et al. (NeurIPS 2024)'s iNaturalist Sounds Dataset. Since this dataset
creation pipepline was not available in the paper's github we had to develop it 
ourselves using the same exact conversion metrics as detailed in the 5th page of
Chasmai et al.'s paper.

Thus, exactly as per Chasmai et al.'s process: this script converts WAV audio 
files to "Mel-spectrogram" .npy files. 

The exact preprocessing recipe is detailed in Chasmai et al. 
"The iNaturalist Sounds Dataset" (NeurIPS 2024)
Section 4: "Model inputs: converting audio to an image"

Paper quote (Section 4):
"Assuming single-channel audio with a sample rate of 22.05kHz, we use
a Hann window size of 512, with a stride length of 128, and 1024 FFT
length for the STFT. We convert the linear spaced frequencies to
mel-scale [...] Our mel-scale maps frequencies in the range
[50Hz, 11.025kHz] to 128 logarithmically spaced mel bins. The
resulting magnitude values are converted to decibels. We convert to
a traditional gray scale image by rescaling the decibel values to
span [0, 255] and save the matrix as an uint8 image."

---

Details below:

---

Parameters from the paper -> our code:
Sample rate: 22.05 kHz -> SR = 22050
Hann window: 512 -> WIN_LENGTH = 512
Stride: 128 -> HOP_LENGTH = 128
FFT length: 1024 -> N_FFT = 1024
Mel bins: 128 -> N_MELS = 128
Freq range: 50-11025 Hz -> FMIN = 50, FMAX = SR // 2
Output: uint8 [0,255] -> .astype(np.uint8)

Output format:
.npy files with shape (128, T), dtype uint8, values in [0, 255]

"""

import os
import argparse
import numpy as np
# note: we use the librosa package to compute the mel-spectrograms easily
# (so we avoid doing it from scratch)
import librosa
from multiprocessing import Pool
from functools import partial
import time

# ===========================================================================
# Step 1: We set hyperparameters to match Section 4 of the paper
# ===========================================================================
SR = 22050 # "sample rate of 22.05kHz"
N_FFT = 1024 # "1024 FFT length"
HOP_LENGTH = 128 # "stride length of 128"
WIN_LENGTH = 512 # "Hann window size of 512"
N_MELS = 128 # "128 logarithmically spaced mel bins"
FMIN = 50 # "frequencies in the range [50Hz, ...]"
FMAX = SR // 2 # "11.025kHz" = 22050 / 2 = 11025


# ===========================================================================
# Step 2: We translate the .wav audio files into .npy spectrogram images
# ===========================================================================
def wav_to_spectrogram(wav_path, output_path):
    """Convert a single WAV file to a dB-scaled Mel-spectrogram .npy file."""
    try:
        # Load audio - dataset WAVs are already 22.05kHz mono
        y, _ = librosa.load(wav_path, sr=SR, mono=True)

        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window='hann',
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
        )

        # Convert power to decibels (ref=np.max -> loudest point = 0 dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Rescale to [0, 255] (as per the paper)
        mel_spec_db = mel_spec_db - mel_spec_db.min()
        if mel_spec_db.max() > 0:
            mel_spec_db = (mel_spec_db / mel_spec_db.max()) * 255.0
        mel_spec_db = mel_spec_db.astype(np.uint8)

        # We save the image
        # note: shape will be (128, T) where T depends on audio duration
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, mel_spec_db)
        return True

    except Exception as e:
        print(f"ERROR processing {wav_path}: {e}")
        return False


# ===========================================================================
# Step 3: We create a function to process multiple .wav files to go faster
# ===========================================================================
def process_file(rel_path, input_dir, output_dir):
    """Worker function for multiprocessing."""
    wav_path = os.path.join(input_dir, rel_path)
    npy_path = os.path.join(output_dir, rel_path.replace('.wav', '.npy'))
    if os.path.exists(npy_path):
        return True  # Skip already processed files
    return wav_to_spectrogram(wav_path, npy_path)


# ===========================================================================
# Step 4: Create the main pipeline to process all the iNatSounds .wav files
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Mel-spectrograms from iNatSounds WAV files."
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory with WAV files (e.g., ./train)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for output .npy files')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    args = parser.parse_args()

    # Collect all WAV files
    wav_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(
                    os.path.relpath(os.path.join(root, f), args.input_dir)
                )

    print(f"We found {len(wav_files)} WAV files in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.num_workers}")
    # Parameters as per the paper:
    print(f"Parameters: SR={SR}, N_FFT={N_FFT}, HOP={HOP_LENGTH}, "
          f"WIN={WIN_LENGTH}, MELS={N_MELS}, FMIN={FMIN}, FMAX={FMAX}")

    worker_fn = partial(
        process_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    start_time = time.time()
    completed, failed = 0, 0

    with Pool(args.num_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(worker_fn, wav_files, chunksize=64)
        ):
            if result:
                completed += 1
            else:
                failed += 1
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(wav_files) - i - 1) / rate
                print(f"  Progress: {i+1}/{len(wav_files)} "
                      f"(done={completed}, fail={failed}, "
                      f"rate={rate:.0f} files/s, ETA={eta/60:.0f}min)")

    elapsed = time.time() - start_time
    print(f"\nComplete in {elapsed/60:.1f} minutes. "
          f"Done: {completed}, Failed: {failed}")


if __name__ == '__main__':
    main()
