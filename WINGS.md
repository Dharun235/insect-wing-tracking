# Wing workflow

## 1. Optical flow

Input frames are numbered JPGs under `<input>/<sequence>/`. The script writes
one motion video per sequence to `<output>/<sequence>_flow.mp4`.

```bash
source .venv/bin/activate
python scripts/flow/wing_optical_flow.py
```

Defaults: input `data/frames/`; output `output/wings/flow/mask_videos/`.
Override them with `--input PATH --output PATH`; use `--overwrite` to rebuild.

Current input → output names:

```text
Test10_07.05_t_lateral_intact_flight1       -> Test10_07_05_t_lateral_intact_flight1_flow.mp4
Test12_08.05_t_lateral_intact_flight1 (2)   -> Test12_08_05_t_lateral_intact_flight1_2_flow.mp4
Test13_11.05_thorax_lateral_intact_flight2  -> Test13_11_05_thorax_lateral_intact_flight2_flow.mp4
Test19_14.05_t_lateral_intact_flight1       -> Test19_14_05_t_lateral_intact_flight1_flow.mp4
Test4_04.05_flight2                          -> Test4_04_05_flight2_flow.mp4
blowfly_1                                    -> blowfly_1_flow.mp4
blowfly_2                                    -> blowfly_2_flow.mp4
blowfly_3                                    -> blowfly_3_flow.mp4
blowfly_latview                              -> blowfly_latview_flow.mp4
blowfly_latview_1wingcut                     -> blowfly_latview_1wingcut_flow.mp4
blowfly_latview_2                           -> blowfly_latview_2_flow.mp4
blowfly_latview_2wingscut                    -> blowfly_latview_2wingscut_flow.mp4
blowfly_latview_2wingscut_2                  -> blowfly_latview_2wingscut_2_flow.mp4
blowfly_latview_3                           -> blowfly_latview_3_flow.mp4
blowfly_topview                              -> blowfly_topview_flow.mp4
blowfly_topview_1wingcut                     -> blowfly_topview_1wingcut_flow.mp4
blowfly_topview_2                           -> blowfly_topview_2_flow.mp4
blowfly_topview_2wingscut                    -> blowfly_topview_2wingscut_flow.mp4
blowfly_topview_2wingscut_2                  -> blowfly_topview_2wingscut_2_flow.mp4
blowfly_topview_3                           -> blowfly_topview_3_flow.mp4
bumblebee_1 ... bumblebee_6                  -> bumblebee_<n>_flow.mp4
bumblebee_latview_1                         -> bumblebee_latview_1_flow.mp4
bumblebee_latview_2                         -> bumblebee_latview_2_flow.mp4
sourcevid                                    -> sourcevid_flow.mp4
```

### Line scan (part of optical flow)

Select scan-line endpoints in any generated flow video and write heatmaps,
centroids, previews, and CSV signals:

```bash
python scripts/flow/scan_flow_line.py \
  --video /path/to/output/<sequence>_flow.mp4 \
  --output output/wings/flow/line_scan
```

It works with any `_flow.mp4` file. Results go to
`<output>/<video-name>/`.

## 2. Easy-Mask full extraction

Easy-Mask is the external tool at
[github.com/Dharun235/easy-mask](https://github.com/Dharun235/easy-mask).
It performs full object extraction and stores results here:

```text
output/wings/tool/<sequence>/
├── cutouts/             # extracted object images
├── masks/               # binary/object masks
├── overlays/            # masks overlaid on the input frames
├── objects.csv          # per-frame measurements
├── objects.json         # object metadata
└── centroid_plot.png    # centroid summary
```

Current result folders:

```text
200409_M02_C001H001S0001
200409_M02_2_C001H001S0001
200409_M02_3_C001H001S0001
bald-faced_hornet_cropped_video_(whole)
black_horsefly_cropped_video_(whole)
blowfly_topview_2
phlebooenes_cropped_video__whole__frames
```
