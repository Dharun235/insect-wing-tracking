#!/usr/bin/env python3
"""Create color-coded flow and SAM2Long wing-mask previews for one sequence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def frame_count(video: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def make_overlay(
    frames: Path,
    mask_input: str,
    output: Path,
    mask_filter: str,
    color: str,
    count: int,
    source_select: str = "",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    select = f"select='{source_select}'," if source_select else ""
    band = "red" if color.startswith("r=") else "green"
    graph = (
        f"[0:v]{select}setpts=PTS-STARTPTS,format=rgb24[original];"
        f"[1:v]{mask_filter}split[alpha][color];"
        f"[color]format=rgb24,lutrgb={color}[colored];"
        "[colored][alpha]alphamerge,setpts=PTS-STARTPTS[mask];"
        f"[original][mask]overlay,drawbox=x=0:y=0:w=iw:h=10:color={band}:t=fill,"
        "format=yuv420p[video]"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
        "-start_number", "0", "-i", str(frames / "%05d.jpg"),
        "-i", mask_input, "-filter_complex", graph, "-map", "[video]",
        "-frames:v", str(count), "-r", "30", "-c:v", "libx264",
        "-crf", "20", "-preset", "ultrafast", str(output),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    frames = root / "data" / "frames" / args.video
    flow = root / "output" / "results_flow" / args.video / "motion.mp4"
    masks = root / "output" / "results_sam2long" / args.video / "masks"
    output = root / "output" / "wing_label_previews" / args.video

    original_count = len(list(frames.glob("*.jpg")))
    flow_count = min(original_count - 1, frame_count(flow))
    sam_count = len(list(masks.glob("*.png")))

    make_overlay(
        frames, str(flow), output / "flow_wing_labels.mp4",
        "format=gray,median=radius=2,lut=y='if(lt(val,4),0,min(255,(val-4)*12))',",
        "r='val':g=0:b=0", flow_count, "gte(n,1)",
    )
    make_overlay(
        frames, str(masks / "%05d.png"), output / "sam2long_wing_labels.mp4",
        "format=gray,lut=y='if(gt(val,0),255,0)',median=radius=1,",
        "r=0:g='val':b=0", sam_count,
    )

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(output / "flow_wing_labels.mp4"),
        "-i", str(output / "sam2long_wing_labels.mp4"),
        "-filter_complex", r"[0:v]select=eq(n\,500)[a];[1:v]select=eq(n\,501)[b];[a][b]hstack=inputs=2",
        "-frames:v", "1", str(output / "frame_500_flow_left_sam2_right.png"),
    ])
    print(f"Saved previews: {output}")


if __name__ == "__main__":
    main()
