"""Extract shape-dependent polygon corners from SAM2Long masks.

Input:
    output/results_sam2long/<sequence>/masks/*.png

Output:
    output/wing_corner_analysis/<sequence>/frames/*.jpg
    output/wing_corner_analysis/<sequence>/corners.csv

The corners are vertices of an approximating polygon around the visible mask;
they are not fixed landmarks such as a permanently selected tip or bottom
point.  Contour order is retained in the CSV so the points can be inspected
and matched over time.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def frame_number(path):
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def source_frame_for(path, frames_root, sequence):
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = frames_root / sequence / (path.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def polygon_corners(contour, epsilon_ratio, max_corners):
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, epsilon_ratio * perimeter)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    # Increase simplification only when a noisy contour produces too many
    # vertices. This preserves shape detail without flooding the CSV.
    while len(polygon) > max_corners:
        epsilon *= 1.25
        polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon[:, 0, :].astype(float), perimeter


def analyze_frame(mask_path, output_path, writer, frames_root, min_area,
                  epsilon_ratio, max_corners):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    sequence = mask_path.parent.parent.name
    source_path = source_frame_for(mask_path, frames_root, sequence)
    if source_path is not None:
        annotated = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    else:
        annotated = None
    if annotated is None:
        annotated = np.zeros((height, width, 3), dtype=np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    ids.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    ids = ids[:2]

    frame = frame_number(mask_path)
    colors = [(0, 80, 255), (255, 120, 0)]  # red/orange, blue/orange in BGR
    for component_index, component_id in enumerate(ids, start=1):
        component = (labels == component_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        corners, perimeter = polygon_corners(contour, epsilon_ratio, max_corners)
        color = colors[(component_index - 1) % len(colors)]
        cv2.drawContours(annotated, [contour], -1, (0, 220, 80), 2, cv2.LINE_AA)
        for corner_index, (x, y) in enumerate(corners, start=1):
            point = (int(round(x)), int(round(y)))
            cv2.circle(annotated, point, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, point, 5, color, -1, cv2.LINE_AA)
            cv2.putText(annotated, f"{component_index}.{corner_index}",
                        (point[0] + 9, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)
            writer.writerow({
                "frame": frame,
                "mask_file": mask_path.name,
                "component": component_index,
                "corner_index_in_contour_order": corner_index,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "component_area_pixels": int(stats[component_id, cv2.CC_STAT_AREA]),
                "contour_perimeter_pixels": round(float(perimeter), 2),
                "touches_image_border": int(
                    x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1
                ),
            })

    cv2.rectangle(annotated, (8, 8), (405, 50), (0, 0, 0), -1)
    cv2.putText(annotated, "GREEN=mask contour   COLORED=polygon corners",
                (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255),
                1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])


def analyze_sequence(mask_dir, output_dir, frames_root, min_area,
                     epsilon_ratio, max_corners):
    masks = sorted(mask_dir.glob("*.png"), key=frame_number)
    if not masks:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame", "mask_file", "component", "corner_index_in_contour_order",
        "x", "y", "component_area_pixels", "contour_perimeter_pixels",
        "touches_image_border",
    ]
    with (output_dir / "corners.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mask_path in masks:
            analyze_frame(mask_path, output_dir / "frames" / f"{mask_path.stem}.jpg",
                          writer, frames_root, min_area, epsilon_ratio, max_corners)
    return len(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", nargs="?", help="One sequence; omit for all")
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "output" / "results_sam2long")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output" / "wing_corner_analysis")
    parser.add_argument("--frames-root", type=Path,
                        default=ROOT / "data" / "frames")
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--epsilon-ratio", type=float, default=0.015,
                        help="Polygon simplification relative to contour perimeter")
    parser.add_argument("--max-corners", type=int, default=16)
    args = parser.parse_args()

    if args.sequence:
        sequence_dirs = [args.input_root / args.sequence]
    else:
        sequence_dirs = sorted(p for p in args.input_root.iterdir()
                               if (p / "masks").is_dir())

    total = 0
    for sequence_dir in sequence_dirs:
        mask_dir = sequence_dir / "masks"
        if not mask_dir.is_dir():
            continue
        count = analyze_sequence(mask_dir, args.output_root / sequence_dir.name,
                                 args.frames_root, args.min_area,
                                 args.epsilon_ratio, args.max_corners)
        total += count
        print(f"{sequence_dir.name}: {count} frames")
    print(f"DONE: {total} frames")


if __name__ == "__main__":
    main()
