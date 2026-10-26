"""Mark approximate outer/far and lower points on propagated wing masks.

The input is expected to look like:
    output/masks/<sequence>/masks/*.png

For each sufficiently large connected component, this script marks:
  - far: the outer contour point farthest from the combined mask center
  - lower: the contour point with the largest image y coordinate

This is intentionally a review/annotation pass, not a final anatomical
landmark detector.  The CSV includes flags that make questionable frames
easy to remove or inspect.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def frame_number(path):
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def choose_point(points, scores, fraction=0.01):
    """Average near-tied contour points to avoid selecting a noisy pixel."""
    best = float(scores.max())
    keep = points[scores >= best * (1.0 - fraction)]
    return np.mean(keep, axis=0), best


def component_points(component_mask, global_center):
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :].astype(np.float64)
    if len(points) == 0:
        return None

    distances = np.linalg.norm(points - global_center[None, :], axis=1)
    far, far_distance = choose_point(points, distances)

    lowest_y = points[:, 1].max()
    lower_candidates = points[points[:, 1] >= lowest_y - 1.0]
    lower = lower_candidates.mean(axis=0)

    return far, lower, far_distance


def draw_marker(image, point, color, label):
    x, y = [int(round(v)) for v in point]
    cv2.circle(image, (x, y), 10, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 7, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)


def source_frame_for(path, frames_root, sequence):
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = frames_root / sequence / (path.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def analysis_roi(sequence, width, height):
    """Return a source-informed ROI for known recordings, or the full frame."""
    if sequence.startswith("200409_"):
        # In these recordings the insect is in the upper-middle of the frame;
        # the lower stand/hand produces frequent bright motion artifacts.
        return (250, 220, min(650, width), min(600, height))
    return (0, 0, width, height)


def analyze_frame(path, annotated_path, csv_writer, min_area, threshold, max_fraction,
                  source_path, previous_fars):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if threshold is None:
        # The stored motion/segmentation masks are not all strictly binary.
        # Otsu separates the bright wing signal from low-level image noise;
        # the relative floor prevents a nearly blank frame from selecting
        # every faint compression artifact as foreground.
        otsu, _ = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(1, int(otsu), int(mask.max() * 0.30))
    binary = (mask >= threshold).astype(np.uint8)
    height, width = binary.shape
    rx0, ry0, rx1, ry1 = analysis_roi(path.parent.parent.name, width, height)
    binary[:ry0, :] = 0
    binary[ry1:, :] = 0
    binary[:, :rx0] = 0
    binary[:, rx1:] = 0
    total_area = int(binary.sum())
    if source_path is not None:
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        annotated = source if source is not None else np.zeros((height, width, 3), dtype=np.uint8)
    else:
        annotated = np.zeros((height, width, 3), dtype=np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_area = int(binary.size * max_fraction)
    component_ids = [
        i for i in range(1, count)
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]
    component_ids.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    component_ids = component_ids[:2]

    if component_ids:
        all_pixels = np.column_stack(np.where(np.isin(labels, component_ids)))
        # np.where gives (y, x); keep points in the drawing convention (x, y).
        global_center = np.array([all_pixels[:, 1].mean(), all_pixels[:, 0].mean()])
    else:
        global_center = np.array([math.nan, math.nan])

    stem = path.stem
    frame = frame_number(path)
    touches_border = bool(
        binary[0, :].any() or binary[-1, :].any() or binary[:, 0].any() or binary[:, -1].any()
    )

    current_fars = {}
    for wing_index, component_id in enumerate(component_ids, start=1):
        component = (labels == component_id).astype(np.uint8) * 255
        result = component_points(component, global_center)
        if result is None:
            continue
        far, lower, far_distance = result
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        x0 = int(stats[component_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[component_id, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[component_id, cv2.CC_STAT_WIDTH]) - 1
        y1 = y0 + int(stats[component_id, cv2.CC_STAT_HEIGHT]) - 1
        component_border = x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1
        current_fars[wing_index] = far
        previous_far = previous_fars.get(wing_index)
        jump = (float(np.linalg.norm(far - previous_far))
                if previous_far is not None else 0.0)
        jump_flag = jump > max(80.0, 0.18 * max(width, height))

        contour, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(annotated, contour, -1, (0, 200, 80), 2, cv2.LINE_AA)
        draw_marker(annotated, far, (0, 0, 255), f"F{wing_index}")
        draw_marker(annotated, lower, (0, 215, 255), f"L{wing_index}")
        csv_writer.writerow({
            "frame": frame,
            "mask_file": path.name,
            "wing": wing_index,
            "area_pixels": component_area,
            "far_x": round(float(far[0]), 2),
            "far_y": round(float(far[1]), 2),
            "far_distance_from_mask_center": round(float(far_distance), 2),
            "lower_x": round(float(lower[0]), 2),
            "lower_y": round(float(lower[1]), 2),
            "mask_center_x": round(float(global_center[0]), 2),
            "mask_center_y": round(float(global_center[1]), 2),
            "touches_image_border": int(touches_border),
            "component_touches_image_border": int(component_border),
            "jump_from_previous_pixels": round(jump, 2),
            "quality": ("border-cropped" if component_border else
                        "temporal-jump" if jump_flag else "ok"),
        })

    cv2.rectangle(annotated, (8, 8), (430, 52), (0, 0, 0), -1)
    cv2.putText(annotated, "RED=far   YELLOW=lower   GREEN=mask contour", (15, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return current_fars


def analyze_sequence(mask_dir, output_dir, min_area, threshold, max_fraction, frames_root):
    masks = sorted(mask_dir.glob("*.png"), key=frame_number)
    if not masks:
        return 0
    frames_dir = output_dir / "frames"
    csv_path = output_dir / "wing_points.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame", "mask_file", "wing", "area_pixels", "far_x", "far_y",
        "far_distance_from_mask_center", "lower_x", "lower_y", "mask_center_x",
        "mask_center_y", "touches_image_border", "component_touches_image_border",
        "jump_from_previous_pixels", "quality",
    ]
    previous_fars = {}
    sequence = mask_dir.parent.name
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in masks:
            source_path = source_frame_for(path, frames_root, sequence)
            previous_fars = analyze_frame(
                path, frames_dir / (path.stem + ".jpg"), writer, min_area, threshold,
                max_fraction, source_path, previous_fars
            )
    return len(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", nargs="?", help="One sequence; omit to analyze all")
    parser.add_argument("--min-area", type=int, default=25,
                        help="Ignore connected components smaller than this many pixels")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Foreground threshold; default is adaptive per frame")
    parser.add_argument("--max-fraction", type=float, default=0.25,
                        help="Reject components larger than this fraction of the image")
    parser.add_argument("--input-root", type=Path, default=ROOT / "output" / "masks")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output" / "wing_point_analysis")
    parser.add_argument("--frames-root", type=Path,
                        default=ROOT / "data" / "frames")
    args = parser.parse_args()

    if args.sequence:
        sequence_dirs = [args.input_root / args.sequence]
    else:
        sequence_dirs = sorted(p for p in args.input_root.iterdir() if (p / "masks").is_dir())

    total = 0
    for sequence_dir in sequence_dirs:
        if not (sequence_dir / "masks").is_dir():
            print(f"SKIP {sequence_dir}: no masks directory")
            continue
        output_dir = args.output_root / sequence_dir.name
        count = analyze_sequence(sequence_dir / "masks", output_dir, args.min_area,
                                 args.threshold, args.max_fraction, args.frames_root)
        total += count
        print(f"{sequence_dir.name}: {count} frames -> {output_dir}")
    print(f"DONE: {total} frames")


if __name__ == "__main__":
    main()
