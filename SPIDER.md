# Spider workflow

Spider segmentation uses the external
[Easy-Mask](https://github.com/Dharun235/easy-mask) tool. Run Easy-Mask on
each `gold-16` through `gold-20` sequence, then export masks, overlays, and
videos.

Exports live under:

```text
output/spider_blue/gold-16_export/
├── masks/
└── overlays/
```

Mask videos live under:

```text
output/spider_blue/videos/gold-16-mask.mp4
output/spider_blue/videos/gold-17-mask.mp4
output/spider_blue/videos/gold-18-mask.mp4
output/spider_blue/videos/gold-19-mask.mp4
output/spider_blue/videos/gold-20-mask.mp4
```

Overlay videos use the same names with `-overlay.mp4`.
