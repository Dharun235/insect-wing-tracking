#!/usr/bin/env python3
"""Create optical-flow motion videos from numbered wing frames.

For every sequence in ``data/frames``, this script computes dense DIS optical
flow between consecutive grayscale frames, keeps motion components of at least
300 pixels, thresholds the motion, and writes a binary motion video to
``output/wings/flow/mask_videos``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT = Path("data/frames")
DEFAULT_OUTPUT = Path("output/wings/flow/mask_videos")


def numbered_frames(directory: Path) -> list[Path]:
    """Return JPG frames in numeric filename order."""
    return sorted(directory.glob("*.jpg"), key=lambda path: int(path.stem))


def motion_mask(flow: cv2.DISOpticalFlow, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Calculate a cleaned binary motion mask between two grayscale frames."""
    vectors = flow.calc(previous, current, None)
    magnitude = cv2.magnitude(vectors[..., 0], vectors[..., 1])
    motion = np.clip((magnitude - 3.0) / 17.0 * 255, 0, 255).astype(np.uint8)
    candidate = cv2.morphologyEx(
        (motion > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros_like(candidate)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 300:
            keep[labels == label] = 1
    return np.where(motion * keep > 180, 255, 0).astype(np.uint8)


def process_sequence(frames_dir: Path, output_dir: Path, overwrite: bool) -> None:
    """Write a motion video for one frame sequence."""
    paths = numbered_frames(frames_dir)
    if len(paths) < 2:
        return

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", frames_dir.name).strip("_")
    output_path = output_dir / f"{name}_flow.mp4"
    if output_path.exists() and not overwrite:
        print(f"Skip existing {output_path}")
        return

    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Could not read {paths[0]}")
    height, width = first.shape
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height), False
    )
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    previous = cv2.GaussianBlur(first, (9, 9), 0)
    try:
        for path in paths[1:]:
            frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise RuntimeError(f"Could not read {path}")
            current = cv2.GaussianBlur(frame, (9, 9), 0)
            writer.write(motion_mask(flow, previous, current))
            previous = current
    finally:
        writer.release()
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, dest="input_root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, dest="output_root")
    parser.add_argument("--overwrite", action="store_true", help="replace existing flow videos")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for frames_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        process_sequence(frames_dir, args.output_root, args.overwrite)


if __name__ == "__main__":
    main()
