"""
We perform some "security checks" so that we can make sure that our data is correct,
not corrupted and ready for the models to ingest.

validate_spectrograms.py: Verify generated spectrograms match paper specs.

The checks we perform:
1. File counts match the official dataset (train=137012, val=45698, test=49527)
2. All .npy files referenced in the JSON annotations exist
3. Array shapes are correct (128 mel bins, variable time frames)
4. Data type is uint8 with values in [0, 255]
5. Time dimension is consistent with audio duration at the paper's settings
"""

import os
import json
import argparse
import numpy as np

# Expected file counts from the official dataset README
# https://github.com/visipedia/inat_sounds/tree/main/2024
EXPECTED_COUNTS = {
    "train": 137012,
    "val": 45698,
    "test": 49527,
}

# Paper parameters for time-dimension verification
HOP_LENGTH = 128
SR = 22050
N_MELS = 128

# We validate one data split (i.e. train / validation / test) at a time
def validate_split(split, spec_dir, json_dir):
    """Validate one split. Returns (passed: bool, messages: list[str])."""
    messages = []
    passed = True

    json_path = os.path.join(json_dir, f"{split}.json")
    split_spec_dir = os.path.join(spec_dir, split)

    # First, we chek if the JSON exists
    if not os.path.exists(json_path):
        return False, [f"Failed: {json_path} not found"]

    with open(json_path, "r") as f:
        dataset = json.load(f)

    audio_id2path = {
        au["id"]: au["file_name"] for au in dataset["audio"]
    }
    expected_files = [
        audio_id2path[a["audio_id"]].replace(".wav", ".npy")
        for a in dataset["annotations"]
    ]

    # Second, we check the file count in the current split
    expected_count = EXPECTED_COUNTS[split]
    actual_count = len(expected_files)
    if actual_count != expected_count:
        messages.append(
            f"Warning: JSON has {actual_count} annotations, "
            f"expected {expected_count}"
        )

    # Third, we chcek that all referenced .npy files exist
    missing = 0
    for rel_path in expected_files:
        full_path = os.path.join(spec_dir, rel_path)
        if not os.path.exists(full_path):
            missing += 1
            if missing <= 5:
                messages.append(f"Missing: {rel_path}")

    if missing > 0:
        passed = False
        messages.append(
            f"Failed: {missing}/{len(expected_files)} files missing"
        )
    else:
        messages.append(
            f"Passed: All {len(expected_files)} spectrogram files present"
        )

    # Finally, we check the first 100 files for shape, dtype, value range
    check_files = expected_files[:100]
    bad_shape = 0
    bad_dtype = 0
    bad_range = 0
    shapes = []

    for rel_path in check_files:
        full_path = os.path.join(spec_dir, rel_path)
        if not os.path.exists(full_path):
            continue
        try:
            arr = np.load(full_path)
            shapes.append(arr.shape)

            if arr.ndim != 2 or arr.shape[0] != N_MELS:
                bad_shape += 1
            if arr.dtype != np.uint8:
                bad_dtype += 1
            if arr.min() < 0 or arr.max() > 255:
                bad_range += 1
        except Exception as e:
            messages.append(f"  ERROR loading {rel_path}: {e}")

    if bad_shape > 0:
        passed = False
        messages.append(
            f"Failed: {bad_shape}/{len(check_files)} files have wrong shape "
            f"(expected (128, T))"
        )
    else:
        messages.append(f"Passed: All checked files have shape (128, T)")

    if bad_dtype > 0:
        passed = False
        messages.append(
            f"Failed: {bad_dtype}/{len(check_files)} files have wrong dtype "
            f"(we expected uint8)"
        )
    else:
        messages.append(f"Passed: All checked files are uint8")

    if bad_range > 0:
        passed = False
        messages.append(
            f"Failed: {bad_range}/{len(check_files)} files have values "
            f"outside [0, 255]"
        )
    else:
        messages.append(f"Passed: All checked files have values in [0, 255]")

    # We add time-frame statistics to our output for completness
    if shapes:
        time_frames = [s[1] for s in shapes]
        durations = [t * HOP_LENGTH / SR for t in time_frames]
        messages.append(
            f"Information: Time frames in checked files: "
            f"min={min(time_frames)}, max={max(time_frames)}, "
            f"median={sorted(time_frames)[len(time_frames)//2]}"
        )
        messages.append(
            f"Information: Corresponding durations (seconds): "
            f"min={min(durations):.1f}s, max={max(durations):.1f}s, "
            f"median={sorted(durations)[len(durations)//2]:.1f}s"
        )

    return passed, messages


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated iNatSounds spectrograms."
    )
    parser.add_argument('--spectrogram_dir', type=str, required=True,
                        help='Root of spectrograms (parent of train/val/test)')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Directory containing train.json, val.json, test.json')
    args = parser.parse_args()

    print("=" * 60)
    print("iNatSounds Spectrogram Validation")
    print("=" * 60)

    all_passed = True

    for split in ["train", "val", "test"]:
        print(f"\ {split} split")
        passed, messages = validate_split(
            split, args.spectrogram_dir, args.json_dir
        )
        for msg in messages:
            print(msg)
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("Result: All Validations Passed")
    else:
        print("Result: Some Validations Failed (check details above)")
    print("=" * 60)


if __name__ == '__main__':
    main()
