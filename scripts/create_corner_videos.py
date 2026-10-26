"""Create MP4 review videos from annotated corner-analysis JPEG frames."""

import argparse
import re
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def frame_number(path):
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else -1


def create_video(frames_dir, output_path, fps):
    frames = sorted(frames_dir.glob("*.jpg"), key=frame_number)
    if not frames:
        return 0
    first = cv2.imread(str(frames[0]))
    if first is None:
        return 0
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")
    written = 0
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None or frame.shape[:2] != (height, width):
                continue
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output" / "wing_corner_videos")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sources = [
        (ROOT / "output" / "wing_corner_analysis", "sam"),
        (ROOT / "output" / "wing_flow_corner_analysis", "flow"),
    ]
    total = 0
    for source_root, prefix in sources:
        if not source_root.is_dir():
            continue
        for sequence_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
            frames_dir = sequence_dir / "frames"
            output_path = args.output_root / f"{prefix}_{sequence_dir.name}.mp4"
            count = create_video(frames_dir, output_path, args.fps)
            if count:
                print(f"{output_path.name}: {count} frames")
                total += 1
    print(f"DONE: {total} videos -> {args.output_root}")


if __name__ == "__main__":
    main()
