#!/usr/bin/env python3
"""Native interactive polygon cropper for the gold spider frame sequences.

Usage:
    python3 scripts/spider_crop_segment.py gold-16

Click polygon points at any angle and press ENTER/SPACE to confirm.
The polygon's bounding region is cropped from every frame; pixels outside
the polygon are masked black. Results are saved under
output/spider/<sequence>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SEQUENCES = [f"gold-{number}" for number in range(16, 21)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", choices=SEQUENCES, help="frame sequence to crop")
    parser.add_argument("--input", type=Path, default=Path("data/frames"), dest="input_root")
    parser.add_argument("--output", type=Path, default=Path("output/spider"), dest="output_root")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for the crop window. Install project requirements first: "
            "pip install -r requirements.txt"
        ) from exc

    source_dir = args.input_root / args.sequence
    first_frame = source_dir / "00000.jpg"
    if not first_frame.is_file():
        raise SystemExit(f"Missing frame: {first_frame}")
    image = cv2.imread(str(first_frame))
    if image is None:
        raise SystemExit(f"Could not read frame: {first_frame}")

    points = select_polygon(cv2, image, args.sequence)
    cv2.destroyAllWindows()
    if len(points) < 3:
        print("Crop cancelled.")
        return

    target_dir = args.output_root / args.sequence
    target_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in target_dir.glob("*.jpg"):
        old_frame.unlink()

    count = apply_polygon(cv2, source_dir, target_dir, points)
    print(f"Saved {count} cropped frames to {target_dir}")


def select_polygon(cv2, image, sequence):
    """Collect polygon points in a fit-to-window preview and map to image pixels."""
    max_width, max_height = 1400, 800
    image_height, image_width = image.shape[:2]
    scale = min(1.0, max_width / image_width, max_height / image_height)
    preview_width = max(1, round(image_width * scale))
    preview_height = max(1, round(image_height * scale))
    preview_base = cv2.resize(image, (preview_width, preview_height))
    points = []

    window = f"Polygon crop {sequence}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def redraw():
        canvas = preview_base.copy()
        display_points = [(round(x * scale), round(y * scale)) for x, y in points]
        for point in display_points:
            cv2.circle(canvas, point, 5, (0, 255, 0), -1)
        if len(display_points) > 1:
            cv2.polylines(canvas, [np.array(display_points)], False, (0, 255, 0), 2)
        if len(display_points) > 2:
            cv2.polylines(canvas, [np.array(display_points)], True, (0, 255, 255), 2)
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x / scale, y / scale))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()
    print("Click polygon points. Right-click removes the last point.")
    print("Press ENTER/SPACE to confirm, C to clear, or ESC to cancel.")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and len(points) >= 3:
            return points
        if key in (ord("c"), ord("C")):
            points.clear()
            redraw()
        if key == 27:
            return []


def apply_polygon(cv2, source_dir, target_dir, points):
    """Crop every frame, segment blue pixels, and keep the largest component."""
    first = cv2.imread(str(source_dir / "00000.jpg"))
    height, width = first.shape[:2]
    polygon = np.round(np.asarray(points)).astype(np.int32)
    x, y, crop_width, crop_height = cv2.boundingRect(polygon)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    mask = mask[y:y + crop_height, x:x + crop_width]

    count = 0
    for source in sorted(source_dir.glob("*.jpg")):
        frame = cv2.imread(str(source))
        cropped = frame[y:y + crop_height, x:x + crop_width]
        cropped[mask == 0] = 0
        blue_only = segment_blue_component(cv2, cropped)
        destination = target_dir / source.name
        temporary = destination.with_suffix(".tmp.jpg")
        if not cv2.imwrite(str(temporary), blue_only, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {destination}")
        temporary.replace(destination)
        count += 1
    return count


def segment_blue_component(cv2, frame):
    """Return the largest cleaned blue component from one cropped BGR frame."""
    blue = cv2.inRange(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
        (90, 50, 30),
        (140, 255, 255),
    )
    kernel = np.ones((5, 5), np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Canny edges help bridge small breaks around the blue object.
    blue_gray = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=blue), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(blue_gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    blue = cv2.bitwise_or(blue, cv2.bitwise_and(edges, blue))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    if component_count <= 1:
        return np.zeros_like(frame)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(frame, frame, mask=component_mask)


if __name__ == "__main__":
    main()
