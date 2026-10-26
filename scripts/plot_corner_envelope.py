"""Plot per-frame upper/lower corner envelopes from a corner CSV."""

import argparse
import csv
from collections import defaultdict
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
    by_frame_component = defaultdict(list)
    for row in rows:
        by_frame_component[(int(row["frame"]), int(row["component"]))].append(
            (float(row["x"]), float(row["y"]))
        )
    if not by_frame_component:
        raise RuntimeError("CSV contains no corners")

    series = defaultdict(list)
    for (frame, component), points in sorted(by_frame_component.items()):
        ys = np.array([p[1] for p in points])
        series[component].append((frame, float(ys.min()), float(ys.max()), float(ys.mean())))

    width, height = 1800, 900
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 100, width - 50, 80, height - 90
    all_values = [v for values in series.values() for row in values for v in row[1:]]
    lo, hi = min(all_values), max(all_values)
    pad = max(10, (hi - lo) * 0.08)
    lo -= pad
    hi += pad
    max_frame = max(row[0] for values in series.values() for row in values)
    colors = [(30, 80, 220), (40, 150, 60), (180, 70, 180)]
    labels = ["uppermost y", "lowermost y", "mean y"]

    cv2.putText(canvas, "SAM mask shape envelope over time", (left, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "y increases downward; periodic curves indicate repeated wing motion",
                (left, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (0, 0, 0), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (0, 0, 0), 2)
    for tick in range(5):
        value = lo + (hi - lo) * tick / 4
        y = bottom - int(round((value - lo) / (hi - lo) * (bottom - top)))
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{value:.0f}", (left - 70, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    for component, values in sorted(series.items()):
        for metric in range(3):
            color = colors[metric]
            previous = None
            for frame, upper, lower, mean in values:
                current = (frame, (upper, lower, mean)[metric])
                if previous is not None:
                    x0 = left + int(round(previous[0] / max_frame * (right - left)))
                    x1 = left + int(round(frame / max_frame * (right - left)))
                    y0 = bottom - int(round((previous[1] - lo) / (hi - lo) * (bottom - top)))
                    y1 = bottom - int(round((current[1] - lo) / (hi - lo) * (bottom - top)))
                    cv2.line(canvas, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
                previous = current

    legend_x = right - 310
    for index, label in enumerate(labels):
        y = 35 + index * 24
        cv2.line(canvas, (legend_x, y), (legend_x + 24, y), colors[index], 3)
        cv2.putText(canvas, label, (legend_x + 32, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 1, cv2.LINE_AA)
    cv2.putText(canvas, f"time (s), assumed {args.fps:g} fps", (left, height - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_path), canvas)
    print(args.output_path)


if __name__ == "__main__":
    main()
