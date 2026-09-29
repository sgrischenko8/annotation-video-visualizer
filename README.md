# Air Target Tracking – Video Annotation Dataset (CVAT)

> Short description: bounding-box tracking annotations of fast-moving jet aircraft in airshow footage,
> made in CVAT with written guidelines and a reproducible quality-check script.

## Overview

| Field | Value |
|---|---|
| Task | Single-object tracking / detection, bounding boxes |
| Classes | `Jet` (1) |
| Attributes | `view`, `motion_blur`, `occlusion` |
| Annotation tool | CVAT (cvat.ai), "CVAT for video 1.1" export |
| Frame size | 854 × 480 |
| Clip | 1560 frames, 52 s (~30 fps) |
| Tracks | 1 |
| Annotation mode | Manual keyframes + interpolation (567 keyframes, 36% of frames) |
| Time spent | 2.5 hours |

### Attributes

| Attribute | Values |
|---|---|
| `view` | `front_side`, `heavily_rotated` |
| `motion_blur` | `none`, `low`, `high` |
| `occlusion` | `none`, `partial`, `full` |

## Dataset statistics

Generated with `validate_cvat.py` (see below).

- Box width: 46–395 px (median 101), height: 26–278 px (median 85)
- Box area: 0.48%–26.6% of the frame (median 2.1%) → the aircraft grows ~8× in width over the clip
- COCO size buckets: 0 small, 862 medium, 698 large
- Motion: median centre shift 0.25 px/frame, max 5.3 px/frame
- No occlusions or exits from the frame in this clip

## Annotation guidelines

Full text: [`GUIDELINES.md`](GUIDELINES.md)

Points the guidelines must settle (decide once, apply everywhere):

- Box tightness: box touches the outermost visible parts (nose, tail, wingtips).
- Vapour cones, contrails and motion blur: included in the box.
- Aircraft partly out of frame: annotate the visible part only.
- Occlusion:
  - `none`: the aircraft is fully visible and is not obscured.
  - `partial`: part of the aircraft is obscured by clouds, glare/overexposure, or vapour emitted by the aircraft itself, while a visible portion remains.
  - `full`: the aircraft is completely obscured and is not visible.
- Keyframe policy: how often to place keyframes; re-check interpolated segments at least every 25 frames.

## Quality control

- Automated checks (`validate_cvat.py`): frame gaps, duplicate frames, boxes outside the frame,
  abnormal jumps in position/size, tiny boxes, missing attributes. Result on this file: no issues.
- Visual review: sample frames with boxes drawn (`--preview`), reviewed by hand.

## Usage

```bash
# download dependecies
pip install -r requirements.txt

# statistics + checks
python validate_cvat.py --xml ./annotations.xml 

# also save a JSON report and a copy without personal data (user name, email, job URLs)
python validate_cvat.py --xml ./annotations.xml --json report.json --anonymize annotations_public.xml

# visual check: montage of sample frames with boxes (needs opencv-python)
python validate_cvat.py --xml ./annotations.xml --preview video.mp4

# save annotated video in output_validated.mp4
python validate_cvat.py --video ./video.mp4 --xml ./annotations.xml --output output_validated.mp4
```

## Repository layout

```
annotations/annotations_public.xml   # CVAT for video 1.1, anonymised
report.json                          # output of validate_cvat.py
GUIDELINES.md                        # annotation guidelines
validate_cvat.py
```

## Data and licensing

- Video source: a clip cut from third-party airshow footage published on the [@ryo_avgeek](https://www.youtube.com/@ryo_avgeek) YouTube channel..
- Only publicly available material is used; no sensitive, geolocated or non-public information.

## Planned next steps

- More clips: different aircraft, aspect angles, backgrounds, distances (especially small targets).
- Attributes: aircraft type, aspect angle, target size class, motion blur, partial occlusion.
- Export to YOLO/COCO and train a small detector (e.g. YOLOv8n) as a sanity check that the data is usable.
