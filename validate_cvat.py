#!/usr/bin/env python3
"""
validate_cvat.py - statistics, sanity checks and anonymisation for
CVAT "CVAT for video 1.1" XML exports (box tracks).

Usage
-----
  python validate_cvat.py annotations.xml
  python validate_cvat.py annotations.xml --json report.json
  python validate_cvat.py annotations.xml --anonymize annotations_public.xml
  python validate_cvat.py annotations.xml --preview video.mp4      # needs opencv-python
  python validate_cvat.py --video ./video.mp4 --xml ./annotations.xml --output result_video.mp4

Exit code: 0 = no errors, 1 = errors found (or warnings with --strict).

The checks are heuristics that catch technical mistakes (gaps, boxes outside
the frame, sudden jumps). They do NOT prove a box fits the object tightly -
that needs a visual review (--preview) or a second annotator.
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import cv2
from pathlib import Path

COCO_SMALL, COCO_MEDIUM = 32 ** 2, 96 ** 2  # COCO area buckets, in pixels^2
PRIVATE_TAGS = {"owner", "assignee", "email", "username", "url", "bugtracker",
                "created", "updated", "dumped"}
ID_PARENTS = {"job", "segment", "task"}  # <id> under these is dropped; <track id=""> is kept


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_cvat_xml(xml_path):
    """Parsing CVAT annotation XML file for extracting boxes by frame numbers."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
    
    # Processing tracks
    for track in root.findall('.//track'):
        label = track.get('label')
        for box in track.findall('box'):
            if box.get('outside') == '1':
                continue
            frame = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append({
                'label': label,
                'box': (int(xtl), int(ytl), int(xbr), int(ybr))
            })
            
    # Processing individual images (image), if annotations are exported per frame
    for image in root.findall('.//image'):
        frame = int(image.get('frame', 0))
        if frame not in annotations:
            annotations[frame] = []
        for box in image.findall('box'):
            label = box.get('label')
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            annotations[frame].append({
                'label': label,
                'box': (int(xtl), int(ytl), int(xbr), int(ybr))
            })
            
    return annotations

def validate_and_save_video(video_path, xml_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Loading annotations from {xml_path}...")
    annotations = parse_cvat_xml(xml_path)
    print(f"Starting video processing ({total_frames} frames)...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in annotations:
            for obj in annotations[frame_idx]:
                x1, y1, x2, y2 = obj['box']
                label = obj['label']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                cv2.putText(frame, label, (x1, max(y1 - 10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed frames: {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"Done! Video successfully saved to: {output_path}")


def parse_meta(root):
    meta = {"width": None, "height": None, "start": 0, "stop": None,
            "labels": {}}  # label -> [declared attribute names]
    size = root.find("meta/original_size")
    if size is not None:
        meta["width"] = int(float(size.findtext("width")))
        meta["height"] = int(float(size.findtext("height")))
    job = root.find("meta/job")
    if job is not None:
        meta["start"] = int(job.findtext("start_frame", "0"))
        if job.findtext("stop_frame"):
            meta["stop"] = int(job.findtext("stop_frame"))
    for lab in root.iter("label"):
        name = child_text(lab, "name", "n")
        if not name:
            continue
        attrs = []
        for a in lab.findall("attributes/attribute"):
            an = child_text(a, "name", "n")
            if an:
                attrs.append(an)
        meta["labels"][name] = attrs
    return meta


def parse_tracks(root):
    tracks = []
    for t in root.findall("track"):
        boxes = []
        for b in t.findall("box"):
            boxes.append({
                "frame": int(b.get("frame")),
                "outside": b.get("outside") == "1",
                "occluded": b.get("occluded") == "1",
                "keyframe": b.get("keyframe") == "1",
                "x1": float(b.get("xtl")), "y1": float(b.get("ytl")),
                "x2": float(b.get("xbr")), "y2": float(b.get("ybr")),
                "attrs": {a.get("name"): (a.text or "") for a in b.findall("attribute")},
            })
        boxes.sort(key=lambda x: x["frame"])
        other = Counter(c.tag for c in t if c.tag not in ("box", "attribute"))
        tracks.append({"id": t.get("id"), "label": t.get("label"),
                       "boxes": boxes, "other_shapes": other})
    return tracks


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summary(vals, nd=1):
    if not vals:
        return None
    return {"min": round(min(vals), nd), "p5": round(pct(vals, 5), nd),
            "median": round(pct(vals, 50), nd), "p95": round(pct(vals, 95), nd),
            "max": round(max(vals), nd)}


def ranges(frames):
    frames = sorted(set(frames))
    out, start, prev = [], None, None
    for f in frames:
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            out.append((start, prev))
            start = prev = f
    if start is not None:
        out.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def centre(b):
    return (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2


def wh(b):
    return b["x2"] - b["x1"], b["y2"] - b["y1"]


# --------------------------------------------------------------------------- #
# Checks + statistics
# --------------------------------------------------------------------------- #
def analyse(root, args):
    meta = parse_meta(root)
    tracks = parse_tracks(root)
    W, H = meta["width"], meta["height"]
    issues = []  # (level, code, track_id, frame, message)

    def add(level, code, tid, frame, msg):
        issues.append((level, code, tid, frame, msg))

    if not tracks:
        add("ERROR", "NO_TRACKS", None, None,
            "No <track> elements found. Export as 'CVAT for video 1.1'.")
    if W is None:
        add("WARN", "NO_SIZE", None, None,
            "meta/original_size missing - bounds checks skipped.")

    seen_ids = Counter(t["id"] for t in tracks)
    for tid, n in seen_ids.items():
        if n > 1:
            add("ERROR", "DUP_TRACK", tid, None, f"track id {tid} appears {n} times")

    widths, heights, areas, shifts = [], [], [], []
    visible_frames = set()
    per_label_tracks, per_label_boxes = Counter(), Counter()
    kf = occ = out = total_boxes = 0
    attr_values = defaultdict(Counter)
    track_lengths = []

    for t in tracks:
        tid, label, boxes = t["id"], t["label"], t["boxes"]
        per_label_tracks[label] += 1
        if meta["labels"] and label not in meta["labels"]:
            add("ERROR", "UNKNOWN_LABEL", tid, None, f"label '{label}' is not declared in meta")
        if t["other_shapes"]:
            add("WARN", "OTHER_SHAPES", tid, None,
                f"non-box shapes ignored: {dict(t['other_shapes'])}")

        frames_seen = Counter(b["frame"] for b in boxes)
        for f, n in frames_seen.items():
            if n > 1:
                add("ERROR", "DUP_FRAME", tid, f, f"{n} boxes on the same frame")

        vis_len = 0
        prev = None
        for b in boxes:
            total_boxes += 1
            kf += b["keyframe"]
            out += b["outside"]
            w, h = wh(b)

            if not b["outside"]:
                vis_len += 1
                per_label_boxes[label] += 1
                occ += b["occluded"]
                visible_frames.add(b["frame"])
                widths.append(w)
                heights.append(h)
                areas.append(w * h)

                if w <= 0 or h <= 0:
                    add("ERROR", "BAD_SIZE", tid, b["frame"], f"non-positive size {w:.1f}x{h:.1f}")
                elif min(w, h) < args.min_side:
                    add("WARN", "TINY", tid, b["frame"], f"very small box {w:.1f}x{h:.1f}px")

                if W is not None:
                    tol = args.bounds_tol
                    if b["x1"] < -tol or b["y1"] < -tol or b["x2"] > W + tol or b["y2"] > H + tol:
                        add("WARN", "OUT_OF_BOUNDS", tid, b["frame"],
                            f"box ({b['x1']:.0f},{b['y1']:.0f})-({b['x2']:.0f},{b['y2']:.0f}) "
                            f"exceeds frame {W}x{H}")

                if b["keyframe"]:
                    for an in meta["labels"].get(label, []):
                        if not b["attrs"].get(an):
                            add("WARN", "MISSING_ATTR", tid, b["frame"],
                                f"attribute '{an}' is empty")
                for an, av in b["attrs"].items():
                    attr_values[f"{label}.{an}"][av] += 1

            if prev is not None:
                if b["frame"] - prev["frame"] != 1 and not prev["outside"] and b["frame"] != prev["frame"]:
                    add("ERROR", "FRAME_GAP", tid, prev["frame"],
                        f"no box between frames {prev['frame']} and {b['frame']}")
                if (b["frame"] - prev["frame"] == 1 and not prev["outside"] and not b["outside"]):
                    (cx0, cy0), (cx1, cy1) = centre(prev), centre(b)
                    shift = math.hypot(cx1 - cx0, cy1 - cy0)
                    shifts.append(shift)
                    pw, ph = wh(prev)
                    diag = math.hypot(pw, ph) or 1.0
                    if shift > args.jump_frac * diag:
                        add("WARN", "JUMP", tid, b["frame"],
                            f"centre moved {shift:.1f}px ({shift / diag:.0%} of box diagonal)")
                    for name, a, c in (("width", pw, w), ("height", ph, h)):
                        if a > 0 and abs(c - a) / a > args.scale_frac:
                            add("WARN", "SCALE_JUMP", tid, b["frame"],
                                f"{name} changed {abs(c - a) / a:.0%} ({a:.0f}->{c:.0f}px)")
            prev = b
        track_lengths.append(vis_len)

    # frames with no visible object
    empty = []
    if meta["stop"] is not None:
        empty = [f for f in range(meta["start"], meta["stop"] + 1) if f not in visible_frames]
        if empty:
            add("INFO", "EMPTY_FRAMES", None, None,
                f"{len(empty)} frame(s) without any visible object: {ranges(empty)}")

    frame_area = (W * H) if W else None
    stats = {
        "frame_size": [W, H],
        "job_frames": (meta["stop"] - meta["start"] + 1) if meta["stop"] is not None else None,
        "tracks": len(tracks),
        "tracks_per_label": dict(per_label_tracks),
        "visible_boxes_per_label": dict(per_label_boxes),
        "keyframes": kf,
        "keyframe_share": round(kf / total_boxes, 3) if total_boxes else None,
        "occluded_boxes": occ,
        "outside_markers": out,
        "track_length_frames": summary(track_lengths, 0),
        "box_width_px": summary(widths),
        "box_height_px": summary(heights),
        "box_area_share_of_frame_pct": summary([a / frame_area * 100 for a in areas], 2) if frame_area else None,
        "coco_size_buckets": {
            "small(<32^2)": sum(a < COCO_SMALL for a in areas),
            "medium(32^2-96^2)": sum(COCO_SMALL <= a < COCO_MEDIUM for a in areas),
            "large(>96^2)": sum(a >= COCO_MEDIUM for a in areas)},
        "centre_shift_px_per_frame": summary(shifts, 2),
        "attribute_values": {k: dict(v) for k, v in attr_values.items()},
        "empty_frames": len(empty),
    }
    return stats, issues


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(stats, issues, max_examples):
    print("=== DATASET STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("\n=== CHECKS ===")
    if not issues:
        print("No issues found.")
        return
    by_code = defaultdict(list)
    for it in issues:
        by_code[(it[0], it[1])].append(it)
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    for (level, code), items in sorted(by_code.items(), key=lambda x: (order[x[0][0]], x[0][1])):
        print(f"\n[{level}] {code}: {len(items)}")
        for _, _, tid, frame, msg in items[:max_examples]:
            where = f"track {tid}" if tid is not None else ""
            where += f" frame {frame}" if frame is not None else ""
            print(f"   {where.strip()}: {msg}" if where else f"   {msg}")
        if len(items) > max_examples:
            print(f"   ... and {len(items) - max_examples} more")


# --------------------------------------------------------------------------- #
# Anonymisation
# --------------------------------------------------------------------------- #
def strip_private(el):
    for child in list(el):
        if child.tag in PRIVATE_TAGS or (child.tag == "id" and el.tag in ID_PARENTS):
            el.remove(child)
        else:
            strip_private(child)


def anonymize(src, dst):
    tree = ET.parse(src)
    strip_private(tree.getroot())
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)


# --------------------------------------------------------------------------- #
# Optional visual preview (needs opencv-python)
# --------------------------------------------------------------------------- #
def preview(xml_path, video_path, out_path, n_frames):
    try:
        import cv2
    except ImportError:
        sys.exit("--preview needs opencv-python: pip install opencv-python")
    root = ET.parse(xml_path).getroot()
    tracks = parse_tracks(root)
    by_frame = defaultdict(list)
    for t in tracks:
        for b in t["boxes"]:
            if not b["outside"]:
                by_frame[b["frame"]].append((t["label"], b))
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = sorted({int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)})
    tiles = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        for label, b in by_frame.get(f, []):
            p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, label, (p1[0], max(p1[1] - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"frame {f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    if not tiles:
        sys.exit("could not read any frames from the video")
    if len(tiles) % 2:
        tiles.append(tiles[-1] * 0)
    rows = [cv2.hconcat(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(out_path, cv2.vconcat(rows))
    print(f"preview saved: {out_path} ({len(picks)} frames)")


def child_text(el, *names):
    for n in names:
        c = el.find(n)
        if c is not None and c.text:
            return c.text.strip()
        
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write stats + issues to this JSON file")
    ap.add_argument("--anonymize", metavar="OUT.xml",
                    help="write a copy without user names, emails, URLs, job ids and timestamps")
    ap.add_argument("--preview", metavar="VIDEO", help="save a montage of sample frames with boxes drawn")
    ap.add_argument("--preview-out", default="preview.jpg")
    ap.add_argument("--preview-frames", type=int, default=8)
    ap.add_argument("--min-side", type=float, default=4.0, help="warn if a box side is below this many px")
    ap.add_argument("--bounds-tol", type=float, default=1.0, help="allowed overshoot outside the frame, px")
    ap.add_argument("--jump-frac", type=float, default=0.15,
                    help="warn if centre moves more than this fraction of the box diagonal per frame")
    ap.add_argument("--scale-frac", type=float, default=0.25,
                    help="warn if width/height changes more than this fraction between adjacent frames")
    ap.add_argument("--max-examples", type=int, default=5)
    ap.add_argument("--strict", action="store_true", help="exit with code 1 on warnings too")
    ap.add_argument('--video', default='', required=False, help='Шлях до вхідного відеофайлу (наприклад, input.mp4)')
    ap.add_argument('--xml', required=True, help='Шлях до XML файлу анотацій з CVAT')
    ap.add_argument('--output', default='output_validated.mp4', help='Шлях для збереження вихідного відео')
    
    args = ap.parse_args()
    video_path = str(Path(args.video))
    xml_path = str(Path(args.xml))
    output_path = str(Path(args.output))
    validate_and_save_video(video_path, xml_path, output_path)

    root = ET.parse(xml_path).getroot()
    stats, issues = analyse(root, args)
    print_report(stats, issues, args.max_examples)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"stats": stats,
                       "issues": [dict(level=l, code=c, track=t, frame=f, message=m)
                                  for l, c, t, f, m in issues]}, fh, indent=2, ensure_ascii=False)
        print(f"\nJSON report: {args.json}")
    if args.anonymize:
        anonymize(args.xml, args.anonymize)
        print(f"anonymised copy: {args.anonymize}")
    if args.preview:
        preview(args.xml, args.preview, args.preview_out, args.preview_frames)

    levels = {i[0] for i in issues}
    sys.exit(1 if "ERROR" in levels or (args.strict and "WARN" in levels) else 0)


if __name__ == "__main__":
    main()