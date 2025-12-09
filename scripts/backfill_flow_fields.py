"""
Backfill flow-field (distance + greedy action) artifacts into existing maze datasets.
Usage:
    python scripts/backfill_flow_fields.py --data-dir src/data/3131

It will:
  - compute dist/action maps for each maze (if missing)
  - save them as <base>_dist.npy / <base>_actions.npy in the same directory
  - add/overwrite metadata columns: dist_file, action_file, path_len
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.maze_generator import MazeGenerator


def backfill(data_dir: str) -> None:
    meta_path = os.path.join(data_dir, "maze_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    df = pd.read_csv(meta_path)
    if "filename" not in df.columns:
        raise ValueError("Expected 'filename' column in metadata.")

    # Ensure columns exist
    if "dist_file" not in df.columns:
        df["dist_file"] = np.nan
    if "action_file" not in df.columns:
        df["action_file"] = np.nan
    if "path_len" not in df.columns:
        df["path_len"] = df.get("optimal_steps", np.nan)

    # Ensure proper dtypes for string columns to avoid pandas casting warnings
    df["dist_file"] = df["dist_file"].astype(object)
    df["action_file"] = df["action_file"].astype(object)

    gen = MazeGenerator(include_flow_fields=True)
    total = len(df)
    for idx, row in df.iterrows():
        maze_file = row["filename"]
        base, _ = os.path.splitext(maze_file)
        dist_file = f"{base}_dist.npy"
        action_file = f"{base}_actions.npy"

        dist_path = os.path.join(data_dir, dist_file)
        action_path = os.path.join(data_dir, action_file)

        if os.path.exists(dist_path) and os.path.exists(action_path):
            df.at[idx, "dist_file"] = dist_file
            df.at[idx, "action_file"] = action_file
            if pd.isna(df.at[idx, "path_len"]) and "optimal_steps" in df.columns:
                df.at[idx, "path_len"] = row.get("optimal_steps", np.nan)
            continue

        maze_path = os.path.join(data_dir, maze_file)
        maze = np.load(maze_path)
        dist_map, action_map = gen.compute_dist_action_maps(maze)
        np.save(dist_path, dist_map)
        np.save(action_path, action_map)

        df.at[idx, "dist_file"] = dist_file
        df.at[idx, "action_file"] = action_file
        if pd.isna(df.at[idx, "path_len"]) and "optimal_steps" in df.columns:
            df.at[idx, "path_len"] = row.get("optimal_steps", np.nan)

        if (idx + 1) % 250 == 0:
            print(f"Processed {idx + 1}/{total}")

    df.to_csv(meta_path, index=False)
    print(f"Backfill complete. Updated metadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset folder containing maze_metadata.csv (e.g., src/data/3131)",
    )
    args = parser.parse_args()
    backfill(args.data_dir)


if __name__ == "__main__":
    main()
