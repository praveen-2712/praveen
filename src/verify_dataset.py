"""
verify_dataset.py — Neurologix Pro V3
=======================================
Scratch script to manually verify that natural sorting is working and that
2.5D stacks are visually consecutive.

Usage:
    python src/verify_dataset.py --data_dir ./data/classification/Training
    python src/verify_dataset.py --data_dir ./data/classification/Training --save_grids
"""

import argparse
import os
import sys
import re
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MRI25DDataset, _natural_sort_key


def verify_sort(data_dir: str, n_classes: int = 4):
    """
    Print the first 12 filenames per class under both sort strategies so you
    can visually confirm that natural sort produces the correct sequence.
    """
    print("=" * 70)
    print("SORT VERIFICATION")
    print("=" * 70)

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        all_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        alpha_sorted   = sorted(all_files)[:12]
        natural_sorted = sorted(all_files, key=_natural_sort_key)[:12]

        print(f"\nClass: {class_name}")
        print(f"  {'Alphabetic (WRONG)':35s}  {'Natural (CORRECT)'}")
        print(f"  {'-'*35}  {'-'*35}")
        for a, n in zip(alpha_sorted, natural_sorted):
            marker = "  " if a == n else "<- DIFF"
            print(f"  {a:35s}  {n:35s}  {marker}")


def verify_stacking(data_dir: str, class_name: str = None,
                    n_samples: int = 5, save_grids: bool = False):
    """
    Load n_samples items from the dataset and check that the three channels
    (t-1, t, t+1) are consecutive frames.  Optionally saves side-by-side grids.
    """
    print("\n" + "=" * 70)
    print("2.5D STACK VERIFICATION")
    print("=" * 70)

    ds = MRI25DDataset(data_dir, transform=None, preprocess_logic=None, img_size=224)

    # Optionally filter to a specific class
    if class_name:
        indices = [i for i, s in enumerate(ds.samples)
                   if s["label"] == class_name.lower()]
    else:
        indices = list(range(len(ds.samples)))

    checked = 0
    for idx in indices:
        if checked >= n_samples:
            break

        sample_meta = ds.samples[idx]
        neighbors   = sample_meta["neighbors"]
        i           = sample_meta["index"]

        prev_name = neighbors[max(0, i - 1)]
        curr_name = neighbors[i]
        next_name = neighbors[min(len(neighbors) - 1, i + 1)]

        # Extract embedded numbers to check consecutiveness
        def nums(fname):
            return [int(x) for x in re.findall(r'\d+', fname)]

        is_consecutive = (nums(prev_name)[-1] + 1 == nums(curr_name)[-1] or
                          prev_name == curr_name)  # boundary pad is fine

        status = "[OK] CONSECUTIVE" if is_consecutive else "[FAIL] NON-CONSECUTIVE (BUG)"

        print(f"\n  [{idx:04d}] class={sample_meta['label']:12s}  {status}")
        print(f"         t-1 : {prev_name}")
        print(f"         t   : {curr_name}")
        print(f"         t+1 : {next_name}")

        if save_grids:
            batch = ds[idx]
            # batch['image'] is a (3, H, W) float tensor in [0,1]
            tensor = batch["image"].numpy()  # (3, H, W)
            channels = [(tensor[c] * 255).astype(np.uint8) for c in range(3)]

            # Tile: [t-1 | t | t+1]
            grid = np.concatenate(channels, axis=1)
            out_name = f"stack_{idx:04d}_{sample_meta['label']}.png"
            cv2.imwrite(out_name, grid)
            print(f"         Saved → {out_name}")

        checked += 1

    print(f"\n[verify] Inspected {checked} samples from {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify natural sort + 2.5D stacking")
    parser.add_argument("--data_dir",   type=str,
                        default="./data/classification/Training")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Filter to a specific class, e.g. glioma")
    parser.add_argument("--n_samples",  type=int, default=5,
                        help="Number of stacks to inspect")
    parser.add_argument("--save_grids", action="store_true",
                        help="Save side-by-side (t-1 | t | t+1) images to disk")
    args = parser.parse_args()

    verify_sort(args.data_dir)
    verify_stacking(args.data_dir, args.class_name, args.n_samples, args.save_grids)
