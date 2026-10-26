"""Extract polygon corners from RAFT optical-flow motion frames."""

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
    while len(polygon) > max_corners:
        epsilon *= 1.25
        polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon[:, 0, :].astype(float), perimeter


def analyze_frame(path, output_path, writer, frames_root, min_area, max_fraction,
                  epsilon_ratio, max_corners):
    motion = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if motion is None:
        raise RuntimeError(f"Could not read motion frame: {path}")
    height, width = motion.shape
    otsu, _ = cv2.threshold(motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(1, int(otsu), int(motion.max() * 0.35))
    binary = (motion >= threshold).astype(np.uint8)

    sequence = path.parent.parent.name
    source_path = source_frame_for(path, frames_root, sequence)
    annotated = cv2.imread(str(source_path), cv2.IMREAD_COLOR) if source_path else None
    if annotated is None:
        annotated = cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_area = int(binary.size * max_fraction)
    ids = [i for i in range(1, count)
           if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    ids.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    ids = ids[:2]

    frame = frame_number(path)
    colors = [(0, 80, 255), (255, 120, 0)]
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
                "motion_file": path.name,
                "component": component_index,
                "corner_index_in_contour_order": corner_index,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "component_area_pixels": int(stats[component_id, cv2.CC_STAT_AREA]),
                "contour_perimeter_pixels": round(float(perimeter), 2),
                "threshold": threshold,
                "touches_image_border": int(
                    x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1
                ),
            })

    cv2.rectangle(annotated, (8, 8), (440, 50), (0, 0, 0), -1)
    cv2.putText(annotated, "GREEN=flow contour   COLORED=flow corners",
                (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255),
                1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])


def analyze_sequence(sequence_dir, output_dir, frames_root, min_area, max_fraction,
                     epsilon_ratio, max_corners):
    motion_dir = sequence_dir / "motion_frames"
    frames = sorted(motion_dir.glob("*.png"), key=frame_number)
    if not frames:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame", "motion_file", "component", "corner_index_in_contour_order",
        "x", "y", "component_area_pixels", "contour_perimeter_pixels",
        "threshold", "touches_image_border",
    ]
    with (output_dir / "corners.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in frames:
            analyze_frame(path, output_dir / "frames" / f"{path.stem}.jpg",
                          writer, frames_root, min_area, max_fraction,
                          epsilon_ratio, max_corners)
    return len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", nargs="?", help="One sequence; omit for all")
    parser.add_argument("--input-root", type=Path, default=ROOT / "results_flow")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output" / "wing_flow_corner_analysis")
    parser.add_argument("--frames-root", type=Path, default=ROOT / "data" / "frames")
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-fraction", type=float, default=0.25)
    parser.add_argument("--epsilon-ratio", type=float, default=0.015)
    parser.add_argument("--max-corners", type=int, default=16)
    parser.add_argument("--exclude", action="append", default=[],
                        help="Sequence name to skip; may be repeated")
    args = parser.parse_args()

    if args.sequence:
        sequence_dirs = [args.input_root / args.sequence]
    else:
        sequence_dirs = sorted(
            p for p in args.input_root.iterdir()
            if (p / "motion_frames").is_dir()
            and not p.name.startswith("Test")
            and p.name not in args.exclude
        )

    total = 0
    for sequence_dir in sequence_dirs:
        count = analyze_sequence(sequence_dir, args.output_root / sequence_dir.name,
                                 args.frames_root, args.min_area, args.max_fraction,
                                 args.epsilon_ratio, args.max_corners)
        total += count
        print(f"{sequence_dir.name}: {count} frames")
    print(f"DONE: {total} frames")


if __name__ == "__main__":
    main()
