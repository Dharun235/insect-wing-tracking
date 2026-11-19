# Spider workflow

`spider_crop_segment.py` opens a native polygon selector, then processes every
frame with blue HSV thresholding, Canny cleanup, and largest-component
selection.

```bash
source .venv/bin/activate
python scripts/spider_crop_segment.py gold-16
```

Controls: left-click adds a polygon point, right-click removes one, `C` clears,
and Enter/Space confirms. Use `gold-17` through `gold-20` for the other inputs.

Portable paths:

```bash
python scripts/spider_crop_segment.py gold-16 \
  --input /path/to/data/frames \
  --output /path/to/output/spider
```

Input: `<input>/gold-16/00000.jpg ...`

Output: `<output>/gold-16/00000.jpg ...` (blue largest component on black).
