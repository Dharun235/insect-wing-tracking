"""Plot polygon-corner x/y trajectories from a corners.csv file."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open()))
    groups = {}
    for row in rows:
        key = f"C{row['component']}.{row['corner_index_in_contour_order']}"
        groups.setdefault(key, []).append((int(row["frame"]),
                                            float(row["x"]), float(row["y"])))
    if not groups:
        raise RuntimeError("CSV contains no corner rows")

    frames = sorted({item[0] for values in groups.values() for item in values})
    max_frame = max(frames)
    width, height = 1800, 1100
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = [tuple(int(v) for v in c) for c in [
        (220, 60, 60), (60, 120, 220), (40, 160, 70), (190, 80, 190),
        (20, 150, 180), (220, 140, 30), (100, 80, 180), (80, 150, 120),
        (180, 100, 50), (120, 120, 120),
    ]]

    left, right = 100, width - 60
    top, bottom = 90, height - 90
    mid = (top + bottom) // 2
    panel_gap = 55
    panel_height = (bottom - top - panel_gap) // 2
    x_panels = [(top, top + panel_height, "x coordinate (pixels)"),
                (top + panel_height + panel_gap, bottom, "y coordinate (pixels)")]

    all_x = [p[1] for values in groups.values() for p in values]
    all_y = [p[2] for values in groups.values() for p in values]
    ranges = [(min(all_x), max(all_x)), (min(all_y), max(all_y))]
    for panel_index, (py0, py1, ylabel) in enumerate(x_panels):
        lo, hi = ranges[panel_index]
        pad = max(10, (hi - lo) * 0.08)
        lo -= pad
        hi += pad
        cv2.line(canvas, (left, py0), (left, py1), (0, 0, 0), 2)
        cv2.line(canvas, (left, py1), (right, py1), (0, 0, 0), 2)
        cv2.putText(canvas, ylabel, (10, py0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2, cv2.LINE_AA)
        for tick in range(5):
            value = lo + (hi - lo) * tick / 4
            y = int(round(py1 - (value - lo) / (hi - lo) * (py1 - py0)))
            cv2.line(canvas, (left - 8, y), (right, y), (225, 225, 225), 1)
            cv2.putText(canvas, f"{value:.0f}", (left - 70, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        for index, (key, values) in enumerate(sorted(groups.items())):
            color = colors[index % len(colors)]
            values = sorted(values)
            for previous, current in zip(values, values[1:]):
                x0 = left + int(round(previous[0] / max_frame * (right - left)))
                x1 = left + int(round(current[0] / max_frame * (right - left)))
                v0, v1 = previous[1 + panel_index], current[1 + panel_index]
                y0 = py1 - int(round((v0 - lo) / (hi - lo) * (py1 - py0)))
                y1 = py1 - int(round((v1 - lo) / (hi - lo) * (py1 - py0)))
                cv2.line(canvas, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"SAM corner trajectories — {args.csv_path.stem}",
                (left, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"time (s), assumed {args.fps:g} fps", (left, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)
    legend_x, legend_y = right - 360, 35
    for index, key in enumerate(sorted(groups)):
        color = colors[index % len(colors)]
        cv2.line(canvas, (legend_x, legend_y + index * 22),
                 (legend_x + 24, legend_y + index * 22), color, 3)
        cv2.putText(canvas, key, (legend_x + 32, legend_y + 5 + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_path), canvas)
    print(args.output_path)


if __name__ == "__main__":
    main()
