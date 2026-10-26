# Insect Wing Tracking

Optical-flow and SAM2Long tools for tracking insect wings from extracted video frames.

Repository designed for RunPod with persistent storage mounted at `/workspace`. Large
inputs, checkpoints, and generated outputs stay outside Git.

## Pipelines

| Pipeline | Input | Working output | Final output |
| --- | --- | --- | --- |
| RAFT optical flow | `data/frames/<sequence>/` | `output/results_flow/<sequence>/` | `output/final_results/<sequence>/motion_overlay_clean.mp4` + `masks/` |
| SAM2Long | `data/frames/<sequence>/` + `output/masks/<sequence>/` | `output/results_sam2long/<sequence>/` | `output/final_results/<sequence>/tracked.mp4` + propagated `masks/` |

Both pipelines expect numbered frames such as `00000.jpg`, `00001.jpg`, and so on.

## Repository layout

```text
data/
  frames/<sequence>/*.jpg       # input frames
output/
  masks/<sequence>/*.png        # Labelme/SAM2Long anchor masks
  results_flow/<sequence>/      # RAFT output
  results_sam2long/<sequence>/  # SAM2Long output
  final_results/<sequence>/      # final videos and masks
scripts/
  run_pipeline.py              # flow or sam2long entrypoint
  create_flow_overlays.py
  create_wing_label_preview.py
```

`run_pipeline.py` is the only pipeline runner. It handles one sequence or all sequences.

`data/` and `output/` are Git-ignored. Keep source videos, frames, masks, checkpoints,
and generated MP4 files there.

## RunPod setup

Attach persistent storage as `/workspace`. Network volumes can be attached to a Pod and
reused after Pod replacement.

Copy this repository to the volume:

```bash
cd /workspace
git clone <your-repository-url> insect-wing-tracking
cd insect-wing-tracking
export INSECT_ROOT=/workspace/insect-wing-tracking
```

Install system tools:

```bash
apt-get update && apt-get install -y git ffmpeg
```

Create separate environments. SAM2Long modifies SAM2 and can conflict with another SAM2
install:

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda create -p "$INSECT_ROOT/.conda/raft" python=3.10 -y
conda run -p "$INSECT_ROOT/.conda/raft" pip install \
  torch torchvision numpy opencv-python tqdm pillow

conda create -p "$INSECT_ROOT/.conda/sam2long" python=3.10 -y
conda run -p "$INSECT_ROOT/.conda/sam2long" pip install \
  'torch>=2.5.1' 'torchvision>=0.20.1' numpy opencv-python tqdm pillow
```

If Pod image lacks matching CUDA PyTorch wheels, install matching wheels from
[pytorch.org](https://pytorch.org/get-started/locally/) instead. Check GPU access:

```bash
conda run -p "$INSECT_ROOT/.conda/raft" python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Install SAM2Long and its checkpoint:

```bash
cd "$INSECT_ROOT"
git clone https://github.com/Mark12Ding/SAM2Long.git
cd SAM2Long
conda run -p "$INSECT_ROOT/.conda/sam2long" pip install -e .
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

This uses SAM2Long’s patched `sam2` package, not a second standalone SAM2 install. See
[SAM2Long](https://github.com/Mark12Ding/SAM2Long) and the
[official SAM2 installation notes](https://github.com/facebookresearch/sam2#installation)
for CUDA-specific errors.

## Create masks with Labelme

Install and open Labelme:

```bash
pip install labelme
labelme
```

Open source frame, draw polygon around wing/object, then save annotation JSON. Convert
JSON to a mask:

```bash
labelme_json_to_dataset 00000.json -o 00000_dataset
```

`label.png` from Labelme is usually an opaque label image. Convert it to the transparent
PNG expected by this runner, where label `0` means background:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

src = Path("00000_dataset/label.png")
dst = Path("output/masks/SEQUENCE_NAME/00000.png")
labels = Image.open(src).convert("L")
mask = Image.new("RGBA", labels.size, (255, 255, 255, 0))
mask.putalpha(labels.point(lambda value: 255 if value else 0))
dst.parent.mkdir(parents=True, exist_ok=True)
mask.save(dst)
PY
```

Repeat for anchor frames. Mask filename must match frame number, for example
`00000.png` for `00000.jpg`. Current SAM2Long runner tracks one object per sequence;
separate wing identities come later.

## RAFT optical flow

One sequence:

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "$INSECT_ROOT/.conda/raft"
python -u "$INSECT_ROOT/scripts/run_pipeline.py" flow SEQUENCE_NAME
```

All sequences:

```bash
python -u "$INSECT_ROOT/scripts/run_pipeline.py" flow
```

Settings in `scripts/run_pipeline.py` (`run_flow` function):

```python
START_FRAME = 0
END_FRAME = float("inf")
SIZE = 512
MAX_MOTION = 20
```

Lower `MAX_MOTION` makes small movement brighter. Increase `SIZE` for more detail.

Output:

```text
results_flow/SEQUENCE_NAME/
├── motion.mp4
├── flow.mp4
└── motion_frames/00001.png
```

Motion frame `00001.png` represents movement from source frames `00000` to `00001`.

## SAM2Long

Each sequence needs at least one anchor mask in `output/masks/<sequence>/`. Mask filenames must
contain matching frame numbers, for example `00000.png`.

One sequence:

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "$INSECT_ROOT/.conda/sam2long"
cd "$INSECT_ROOT/SAM2Long"
python -u "$INSECT_ROOT/scripts/run_pipeline.py" sam2long SEQUENCE_NAME
```

All sequences with matching frames and masks:

```bash
python -u "$INSECT_ROOT/scripts/run_pipeline.py" sam2long
```

Output:

```text
results_sam2long/SEQUENCE_NAME/
├── tracked.mp4
└── masks/00000.png
```

## Motion overlays

Raw red overlay for one sequence:

```bash
python scripts/create_flow_overlays.py SEQUENCE_NAME
```

Cleaned overlays for all sequences:

```bash
python scripts/create_flow_overlays.py --clean --threshold 4 --boost 12
```

Raw output is `motion_overlay.mp4`; cleaned output is `motion_overlay_clean.mp4`.
Clean mode applies median denoising, then colors motion pixels red while preserving the
original frame. Outputs go under `output/final_results/<sequence>/`.

## Wing-label preview

Create first-pass previews for one sequence:

```bash
python scripts/create_wing_label_preview.py SEQUENCE_NAME
```

Output:

```text
output/wing_label_previews/SEQUENCE_NAME/
├── flow_wing_labels.mp4              # cleaned flow candidate, red
├── sam2long_wing_labels.mp4           # SAM2Long mask, green
└── frame_100_flow_left_sam2_right.png
```

These are candidate wing masks, not yet left/right wing tracking. Next analysis stage can
calculate connected components, centroids, and temporal identities.

## Final results folder

`output/final_results/` follows the processing list below:

- 33 `optical flow` entries contain `motion_overlay_clean.mp4` and their
  `results_flow/<sequence>/motion_frames/` files under `masks/`.
- 2 `sam` entries contain `tracked.mp4` plus propagated `masks/`.

```text
final_results/<sequence>/
├── masks/*.png                  # motion frames or SAM2Long masks
└── motion_overlay_clean.mp4     # optical-flow entries

final_results/<sam-sequence>/
├── masks/*.png
└── tracked.mp4                  # SAM2Long overlay
```

Files are exact hard links to existing outputs. No masks are converted or regenerated.
Copy this folder for sharing or download; intermediate outputs are elsewhere.

## Video processing list

These are the exact videos included in `output/final_results/`.

| Video group | Processing | Final video | Final `masks/` contents |
| --- | --- | --- | --- |
| 33 names under **Optical flow** below | RAFT motion | `motion_overlay_clean.mp4` | `results_flow/.../motion_frames/*.png` |
| 2 names under **SAM2Long** below | SAM2Long propagation | `tracked.mp4` | `results_sam2long/.../masks/*.png` |

**SAM2Long**

- `bald-faced_hornet_cropped_video_(whole)`
- `black_horsefly_cropped_video_(whole)`

**Optical flow**

- `200409_M02_2_C001H001S0001`
- `200409_M02_3_C001H001S0001`
- `200409_M02_C001H001S0001`
- `blowfly_1`
- `blowfly_2`
- `blowfly_3`
- `blowfly_latview`
- `blowfly_latview_1wingcut`
- `blowfly_latview_2`
- `blowfly_latview_2wingscut`
- `blowfly_latview_2wingscut_2`
- `blowfly_latview_3`
- `blowfly_topview`
- `blowfly_topview_1wingcut`
- `blowfly_topview_2`
- `blowfly_topview_2wingscut`
- `blowfly_topview_2wingscut_2`
- `blowfly_topview_3`
- `bumblebee_1`
- `bumblebee_2`
- `bumblebee_3`
- `bumblebee_4`
- `bumblebee_5`
- `bumblebee_6`
- `bumblebee_latview_1`
- `bumblebee_latview_2`
- `phlebooenes_cropped_video_(whole)`
- `sourcevid`
- `Test4_04.05_flight2`
- `Test10_07.05_t_lateral_intact_flight1`
- `Test12_08.05_t_lateral_intact_flight1 (2)`
- `Test13_11.05_thorax_lateral_intact_flight2`
- `Test19_14.05_t_lateral_intact_flight1`

## Download results from RunPod

Use the Pod's **Connect → SSH** details. For a folder:

```bash
scp -r -P PORT root@HOST:/workspace/insect-wing-tracking/output/final_results ./
```

For large output, archive inside the Pod first:

```bash
cd /workspace/insect-wing-tracking
tar -cf final_results.tar output/final_results
```

Then download:

```bash
scp -P PORT root@HOST:/workspace/insect-wing-tracking/final_results.tar ./
```

## Notes

- Keep anchor masks aligned with frame numbering.
- Default FPS is 30 for both pipelines.
- Check free disk space before full runs; MP4 outputs can be large.
- Do not commit data, checkpoints, API keys, or generated videos.
