"""Run one pipeline for one sequence.

Usage:
    python scripts/run_pipeline.py flow SEQUENCE
    python scripts/run_pipeline.py sam2long SEQUENCE
    python scripts/run_pipeline.py flow
    python scripts/run_pipeline.py sam2long
"""

import argparse
import gc
import os
import re
import shutil
import tempfile
from pathlib import Path


ROOT = Path(os.environ.get("INSECT_ROOT", "/workspace"))
FPS = 30


def frame_number(path):
    return int(re.findall(r"\d+", path.stem)[-1])


def frames_for(sequence):
    frame_dir = ROOT / "data" / "frames" / sequence
    return sorted(
        [p for p in frame_dir.iterdir()
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=frame_number,
    )


def run_flow(sequence):
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from torchvision.utils import flow_to_image

    size = 512
    max_motion = 20
    input_frames = frames_for(sequence)
    assert len(input_frames) >= 2, "Need at least 2 frames"

    output = ROOT / "output" / "results_flow" / sequence
    (output / "motion_frames").mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(input_frames[0]))
    height, width = first.shape[:2]
    processed_width = max(128, round(width * size / max(width, height) / 8) * 8)
    processed_height = max(128, round(height * size / max(width, height) / 8) * 8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()
    transform = weights.transforms()

    motion_video = cv2.VideoWriter(
        str(output / "motion.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (width, height)
    )
    flow_video = cv2.VideoWriter(
        str(output / "flow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (width, height)
    )
    assert motion_video.isOpened() and flow_video.isOpened()

    def load(path):
        image = cv2.resize(cv2.imread(str(path)),
                           (processed_width, processed_height))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    previous = load(input_frames[0])
    with torch.inference_mode():
        for path in tqdm(input_frames[1:], desc="RAFT optical flow"):
            current = load(path)
            a, b = transform(previous, current)
            flow = model(a.to(device), b.to(device))[-1][0]
            magnitude = torch.linalg.vector_norm(flow, dim=0).cpu().numpy()
            motion = np.clip(magnitude / max_motion * 255, 0, 255).astype(np.uint8)
            motion = cv2.resize(motion, (width, height))
            flow_rgb = flow_to_image(flow.cpu()).permute(1, 2, 0).numpy()
            flow_bgr = cv2.resize(cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR),
                                  (width, height))
            cv2.imwrite(str(output / "motion_frames" /
                            f"{frame_number(path):05d}.png"), motion)
            motion_video.write(cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR))
            flow_video.write(flow_bgr)
            previous = current

    motion_video.release()
    flow_video.release()
    print("DONE")
    print("Motion:", output / "motion.mp4")
    print("Flow:", output / "flow.mp4")
    print("Frames:", output / "motion_frames")


def run_sam2long(sequence):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from sam2.build_sam import build_sam2_video_predictor

    input_frames = frames_for(sequence)
    assert input_frames, "No frames found"
    mask_dir = ROOT / "output" / "masks" / sequence
    output = ROOT / "output" / "results_sam2long" / sequence
    output_masks = output / "masks"
    output_masks.mkdir(parents=True, exist_ok=True)
    width, height = Image.open(input_frames[0]).size
    numbers = [frame_number(p) for p in input_frames]
    lookup = {number: index for index, number in enumerate(numbers)}
    anchors = sorted(
        ((lookup[frame_number(p)], p) for p in mask_dir.glob("*.png")
         if frame_number(p) in lookup),
        key=lambda item: item[0],
    )
    assert anchors, "No matching masks"

    def read_mask(path):
        rgba = np.array(Image.open(path).convert("RGBA"))
        mask = (rgba[:, :, 3] > 127).astype(np.uint8)
        assert mask.shape == (height, width) and mask.any(), f"Invalid mask: {path}"
        return mask

    def save(index, mask):
        cv2.imwrite(str(output_masks / f"{numbers[index]:05d}.png"),
                    mask.astype(np.uint8) * 255)

    print("Anchors:", [numbers[index] for index, _ in anchors], flush=True)
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoints/sam2.1_hiera_base_plus.pt",
    )

    def track(indices, mask_file, label):
        if not indices:
            return
        if len(indices) == 1:
            save(indices[0], read_mask(mask_file))
            return
        temp = Path(tempfile.mkdtemp(prefix="sam2long_"))
        state = logits = None
        try:
            for j, index in enumerate(indices):
                with Image.open(input_frames[index]) as image:
                    image.convert("RGB").save(temp / f"{j:05d}.jpg", quality=95)
            print(f"{label}: {numbers[indices[0]]} -> {numbers[indices[-1]]}", flush=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=str(temp),
                                             offload_video_to_cpu=True,
                                             offload_state_to_cpu=True)
                state["num_pathway"] = 3
                state["iou_thre"] = 0.1
                state["uncertainty"] = 2
                predictor.add_new_mask(inference_state=state, frame_idx=0,
                                       obj_id=1, mask=read_mask(mask_file))
                _, logits = predictor.propagate_in_video(state)
                assert len(logits) == len(indices)
                for j, index in enumerate(tqdm(indices, desc="Saving masks")):
                    save(index, (logits[j].squeeze() > 0).cpu().numpy())
            save(indices[0], read_mask(mask_file))
        finally:
            del state, logits
            shutil.rmtree(temp, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()

    for index, mask_file in anchors:
        save(index, read_mask(mask_file))
    first, first_mask = anchors[0]
    if first > 0:
        track(list(range(first, -1, -1)), first_mask, "Initial backward")
    for k in range(len(anchors) - 1):
        start, start_mask = anchors[k]
        end, end_mask = anchors[k + 1]
        track(list(range(start, end)), start_mask,
              f"Tracking {numbers[start]}-{numbers[end]}")
        save(end, read_mask(end_mask))
    last, last_mask = anchors[-1]
    if last < len(input_frames) - 1:
        track(list(range(last, len(input_frames))), last_mask, "Final forward")

    video_path = output / "tracked.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (width, height))
    assert writer.isOpened(), "Could not create video"
    for path in tqdm(input_frames, desc="Video"):
        frame = cv2.imread(str(path))
        mask = cv2.imread(str(output_masks / f"{frame_number(path):05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        overlay = frame.copy()
        overlay[mask > 0] = (0, 255, 0)
        writer.write(cv2.addWeighted(frame, 0.6, overlay, 0.4, 0))
    writer.release()
    print("DONE")
    print("Video:", video_path)
    print("Masks:", output_masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=("flow", "sam2long"))
    parser.add_argument("sequence", nargs="?")
    args = parser.parse_args()
    source = ROOT / "data" / "frames" if args.pipeline == "flow" else ROOT / "output" / "masks"
    sequences = [args.sequence] if args.sequence else sorted(
        p.name for p in source.iterdir()
        if p.is_dir() and (
            args.pipeline == "flow" or (ROOT / "data" / "frames" / p.name).is_dir()
        )
    )
    for sequence in sequences:
        print(f"\n===== {args.pipeline}: {sequence} =====", flush=True)
        try:
            (run_flow if args.pipeline == "flow" else run_sam2long)(sequence)
        except Exception as error:
            print(f"FAILED: {sequence}: {error}", flush=True)
            if args.sequence:
                raise


if __name__ == "__main__":
    main()
