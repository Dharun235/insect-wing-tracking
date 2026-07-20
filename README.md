## Insect Wing Tracking

Simple notebooks for processing lateral-view insect videos.

### Notebooks

- `scripts/optical-flow.ipynb`  
  Runs RAFT optical flow and saves MP4 flow videos.

- `scripts/dense-tracking.ipynb`  
  Runs CoTracker dense point tracking and saves MP4 tracking videos plus `.pt` track files.

- `scripts/mde-insect.ipynb`  
  Runs Video-Depth-Anything monocular depth estimation and saves MP4 depth videos.

### Input

Kaggle input folder:

```text
/kaggle/input/datasets/dharun235/lateral-insect-view/Lateral view insect video
```

Supported video types:

```text
.mp4 .avi .mov .mkv
```

### Output

Each notebook writes flat output files to `/kaggle/working`.

```text
RAFT_Outputs/
MDE_Outputs/
dense-track/
```

Example output names:

```text
Adrian_bald-faced hornet_c_26832_1064_raft.mp4
Adrian_bald-faced hornet_c_26832_1064_vis.mp4
Adrian_bald-faced hornet_c_26832_1064_tracking.mp4
Adrian_bald-faced hornet_c_26832_1064_tracks.pt
```

### Notes

Use GPU runtime on Kaggle.

For `mde-insect.ipynb`, add a Kaggle secret named:

```text
hf-t1
```

with your Hugging Face token.

## Note
- Store the outputs here - smb://sol.ita.chalmers.se/me-inter-flight and login with CID and password
