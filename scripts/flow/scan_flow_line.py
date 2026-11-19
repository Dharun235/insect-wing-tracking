"""Select scan lines in a wing-flow video and extract their signals."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_VIDEO = Path("output/wings/flow/mask_videos/blowfly_topview_flow.mp4")
DEFAULT_OUTPUT = Path("output/wings/flow/line_scan")
FRAME_INDEX = 0
THRESHOLD = 180
SAMPLES = 500
COLORS = [(0, 0, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0)]


def crop_for(video: Path) -> tuple[int, int, int, int]:
    name = video.stem
    if "topview_3" in name:
        return 500, 400, 1000, 960
    if name.endswith("topview_flow") or "topview_2wingscut_2" in name:
        return 500, 600, 1000, 960
    return 500, 500, 1000, 960


def read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {index} from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def select_lines(frame: np.ndarray, crop: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    points: list[tuple[int, int]] = []
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = crop
    cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)

    def click(event: int, x: int, y: int, *_: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not (x0 <= x < x1 and y0 <= y < y1):
            return
        points.append((x, y))
        cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
        if len(points) % 2 == 0:
            a, b = points[-2:]
            cv2.line(preview, a, b, COLORS[(len(points) // 2 - 1) % len(COLORS)], 2)

    window = "Click line endpoints; Enter = finish, Esc = cancel"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    while True:
        cv2.imshow(window, preview)
        key = cv2.waitKey(20)
        if key == 13 and len(points) >= 2 and len(points) % 2 == 0:
            break
        if key == 27:
            cv2.destroyWindow(window)
            raise SystemExit("Cancelled")
    cv2.destroyWindow(window)
    return list(zip(points[::2], points[1::2]))


def sample_lines(video: Path, lines: list[tuple[tuple[int, int], tuple[int, int]]]) -> tuple[np.ndarray, list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read {video}")
    coordinates = []
    for start, end in lines:
        x = np.linspace(start[0], end[0], SAMPLES).round().astype(int)
        y = np.linspace(start[1], end[1], SAMPLES).round().astype(int)
        coordinates.append((x, y))
    profiles = [[] for _ in lines]
    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for values, (x, y) in zip(profiles, coordinates):
            values.append(gray[y, x])
        ok, frame = cap.read()
    cap.release()
    return np.arange(len(profiles[0])), [np.asarray(values, dtype=float) for values in profiles]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = read_frame(args.video, FRAME_INDEX)
    lines = select_lines(frame, crop_for(args.video))
    frames, profiles = sample_lines(args.video, lines)
    output = args.output / args.video.stem
    output.mkdir(parents=True, exist_ok=True)

    columns = {"frame": frames}
    stats = []
    for index, values in enumerate(profiles, 1):
        active = values > THRESHOLD
        weights = np.maximum(values - THRESHOLD, 0)
        total = weights.sum(axis=1)
        centroid = np.divide(weights @ np.arange(SAMPLES), total, out=np.full(total.shape, np.nan), where=total > 0)
        has_active = active.any(axis=1)
        top = np.where(has_active, active.argmax(axis=1), np.nan)
        bottom = np.where(has_active, SAMPLES - 1 - active[:, ::-1].argmax(axis=1), np.nan)
        start, end = lines[index - 1]
        x = np.linspace(start[0], end[0], SAMPLES).round().astype(int)
        y = np.linspace(start[1], end[1], SAMPLES).round().astype(int)
        top_i = np.nan_to_num(top, nan=0).astype(int)
        bottom_i = np.nan_to_num(bottom, nan=0).astype(int)
        columns.update({
            f"line_{index}_centroid": centroid,
            f"line_{index}_active_fraction": active.mean(axis=1),
            f"line_{index}_top_x": np.where(has_active, x[top_i], np.nan),
            f"line_{index}_top_y": np.where(has_active, y[top_i], np.nan),
            f"line_{index}_bottom_x": np.where(has_active, x[bottom_i], np.nan),
            f"line_{index}_bottom_y": np.where(has_active, y[bottom_i], np.nan),
        })
        stats.append((values, centroid))

    names = list(columns)
    np.savetxt(output / "multi_line_signal.csv", np.column_stack([columns[name] for name in names]), delimiter=",", header=",".join(names), comments="")

    plt.figure(figsize=(12, 5))
    for index, (_, centroid) in enumerate(stats, 1):
        plt.plot(frames, centroid, linewidth=1, label=f"line_{index}")
    plt.xlabel("Frame")
    plt.ylabel("Brightness centroid")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output / "centroid_signals.png", dpi=200)
    plt.close()

    figure, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 4), squeeze=False)
    for axis, (values, centroid), index in zip(axes[0], stats, range(1, len(stats) + 1)):
        axis.imshow(values.T, aspect="auto", origin="lower", cmap="gray")
        axis.plot(frames, centroid, color="red", linewidth=1)
        axis.set_title(f"line_{index}")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Position along line")
        axis.grid(False)
    figure.tight_layout()
    figure.savefig(output / "line_heatmaps.png", dpi=200)
    plt.close(figure)

    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for index, (start, end) in enumerate(lines):
        cv2.line(preview, start, end, COLORS[index % len(COLORS)], 2)
    cv2.imwrite(str(output / "multi_line_preview.png"), preview)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
