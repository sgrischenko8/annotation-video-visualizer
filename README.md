# Vehicle Tracking – Video Annotation Dataset (CVAT)

> Bounding-box tracking annotations of vehicles in traffic or street footage,
> made in CVAT with written guidelines and a reproducible quality-check script.

## Overview

| Field | Value |
|---|---|
| Task | Single-object / Multi-object tracking & detection, bounding boxes |
| Classes | `Car`|
| Annotation tool | CVAT (cvat.ai), "CVAT for video 1.1" export |
| Frame size | 1440 × 2560 |
| Clip | 540 frames |
| Annotation mode | Manual keyframes + interpolation |


## Dataset statistics

Generated with `validate_cvat.py` (see below).

- Box dimensions and COCO size buckets (calculated automatically via script).
- Motion metrics: centre shifts and scale changes between frames.

## Annotation guidelines

Full text: [`GUIDELINES.md`](GUIDELINES.md)

Points the guidelines must settle (decide once, apply everywhere):

- Box tightness: box touches the outermost visible parts of the vehicle (bumpers, side mirrors, wheels).
- Keyframe policy: place keyframes during rapid turns, acceleration, or partial occlusions.

## Quality control

- Automated checks (`validate_cvat.py`): frame gaps, duplicate frames, boxes outside the frame,
  abnormal jumps in position/size, tiny boxes, missing attributes.
- Visual review: sample frames with boxes drawn (`--preview`) or full video rendering (`--video`).

## Usage

```bash
# download dependencies
pip install -r requirements.txt

# statistics + checks
python validate_cvat.py --xml ./annotations.xml 

# save a JSON report and an anonymised copy
python validate_cvat.py --xml ./annotations.xml --json report.json --anonymize annotations_public.xml

# visual check: montage of sample frames with boxes
python validate_cvat.py --xml ./annotations.xml --preview video.mp4

# save validated video with drawn bounding boxes
python validate_cvat.py --video ./video.mp4 --xml ./annotations.xml --output output_validated.mp4

## Repository layout

```
annotations.xml                      # CVAT for video 1.1, anonymised
report.json                          # output of validate_cvat.py
GUIDELINES.md                        # annotation guidelines
validate_cvat.py
```