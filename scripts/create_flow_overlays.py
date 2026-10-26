#!/usr/bin/env python3
"""Overlay one results_flow motion video on its original extracted frames."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video", nargs="?",
        help="One directory name; omit to process every matching video",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--threshold", type=int, default=6)
    parser.add_argument("--boost", type=int, default=6)
    parser.add_argument("--clean", action="store_true", help="Denoise motion mask before overlay")
    parser.add_argument("--median-radius", type=int, default=2)
    parser.add_argument("--output", help="Output MP4 path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    names = [args.video] if args.video else sorted(
        p.name for p in (root / "output" / "results_flow").iterdir() if p.is_dir()
    )
    if args.output and len(names) != 1:
        raise SystemExit("--output requires one video name")

    for name in names:
        create_overlay(root, name, args)


def create_overlay(root: Path, name: str, args: argparse.Namespace) -> None:
    frames_dir = root / "data" / "frames" / name
    motion = root / "output" / "results_flow" / name / "motion.mp4"
    default_name = "motion_overlay_clean.mp4" if args.clean else "motion_overlay.mp4"
    output = Path(args.output) if args.output else (
        root / "output" / "final_results" / name / default_name
    )

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"SKIP {name}: no JPG frames found")
        return
    if not motion.exists():
        print(f"SKIP {name}: motion.mp4 not found")
        return
    if frames[0].stem != "00000":
        print(f"SKIP {name}: expected first frame 00000.jpg, found {frames[0].name}")
        return

    # Motion frame 0 compares original frames 0 and 1, so it overlays original frame 1.
    frame_count = min(len(frames) - 1, video_frame_count(motion))
    output.parent.mkdir(parents=True, exist_ok=True)
    cleanup = f"median=radius={args.median_radius}," if args.clean else ""
    filter_graph = (
        "[0:v]select='gte(n,1)',setpts=PTS-STARTPTS,format=rgb24[original];"
        f"[1:v]format=gray,{cleanup}lut=y='if(lt(val,{args.threshold}),0,"
        f"min(255,(val-{args.threshold})*{args.boost}))',"
        "split[alpha][color];"
        "[color]format=rgb24,lutrgb=r='val':g=0:b=0[red];"
        "[red][alpha]alphamerge,setpts=PTS-STARTPTS[motion];"
        "[original][motion]overlay=shortest=1,"
        "format=yuv420p[video]"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(args.fps), "-start_number", "0",
        "-i", str(frames_dir / "%05d.jpg"),
        "-i", str(motion),
        "-filter_complex", filter_graph, "-map", "[video]",
        "-frames:v", str(frame_count), "-r", str(args.fps),
        "-c:v", "libx264", "-crf", "20", "-preset", "ultrafast",
        str(output),
    ])
    print(f"DONE {name}: {output} ({frame_count} frames)")


if __name__ == "__main__":
    main()
