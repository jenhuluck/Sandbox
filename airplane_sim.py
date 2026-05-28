#!/usr/bin/env python3
"""
simulate_airplanes_v3_auto_prototypes.py

Generate synthetic airport video frames by:
1) taking a clean airport background image,
2) taking ONE source RGB image + ONE source mask image that contain MULTIPLE airplane masks,
3) automatically splitting the multi-instance mask into separate airplane prototypes,
4) filtering / randomly selecting valid prototype airplanes,
5) automatically estimating nose/head and tail from each mask,
6) moving them along start->end tracks from tracks.json,
7) exporting frames, mp4, and labels.

No assets.json is needed.

Example:
python simulate_airplanes_v3_auto_prototypes.py \
  --background airport_bg.png \
  --source-rgb prototype_source_rgb.png \
  --source-mask prototype_source_mask.png \
  --tracks-json tracks.json \
  --out-dir sim_out \
  --num-frames 120 \
  --fps 24 \
  --shadow \
  --debug-prototypes \
  --debug-orientation
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_image(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def bbox_visible_fraction(bbox_xywh, image_w, image_h):
    """
    Compute visible fraction of a bbox after clipping to image boundary.
    bbox_xywh: [x, y, w, h]
    """
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return 0.0, None

    x1, y1 = x, y
    x2, y2 = x + w, y + h

    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(image_w, x2)
    cy2 = min(image_h, y2)

    cw = max(0, cx2 - cx1)
    ch = max(0, cy2 - cy1)

    visible_area = cw * ch
    full_area = w * h
    frac = visible_area / max(full_area, 1e-6)

    if visible_area <= 0:
        return 0.0, None

    return float(frac), [int(cx1), int(cy1), int(cw), int(ch)]


def image_angle_deg(p1, p2):
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.degrees(math.atan2(dy, dx))


def make_binary_mask(mask_img, threshold=0):
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    return (mask_img > threshold).astype(np.uint8)


def clean_mask(binary_mask):
    kernel = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def make_alpha(binary_mask):
    mask = (binary_mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    return alpha


def component_shape_features(component_mask):
    mask = (component_mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    area = int(mask.sum())
    if area == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bw = int(x2 - x1 + 1)
    bh = int(y2 - y1 + 1)
    aspect = max(bw, bh) / max(min(bw, bh), 1)

    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        solidity = 0.0
    else:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(cnt_area / hull_area) if hull_area > 0 else 0.0

    return {
        "area": area,
        "bbox_w": bw,
        "bbox_h": bh,
        "aspect": float(aspect),
        "solidity": float(solidity),
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
    }


def crop_to_component(rgb, comp_mask, pad=8):
    ys, xs = np.where(comp_mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty component mask")
    h, w = comp_mask.shape[:2]
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    rgb_crop = rgb[y1:y2, x1:x2].copy()
    mask_crop = comp_mask[y1:y2, x1:x2].copy()
    alpha_crop = make_alpha(mask_crop)
    return rgb_crop, mask_crop, alpha_crop, (x1, y1)


def extract_components(source_rgb, source_mask, min_area=80, max_area=None,
                       min_aspect=1.2, max_aspect=8.0, min_solidity=0.20,
                       scores=None, min_score=0.0):
    binary = make_binary_mask(source_mask)
    binary = clean_mask(binary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []
    for cid in range(1, num_labels):
        comp = (labels == cid).astype(np.uint8)
        feats = component_shape_features(comp)
        if feats is None:
            continue

        area = feats["area"]
        aspect = feats["aspect"]
        solidity = feats["solidity"]

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if solidity < min_solidity:
            continue

        score = None
        if scores is not None:
            score = float(scores.get(str(cid), scores.get(cid, 0.0)))
            if score < min_score:
                continue

        rgb_crop, mask_crop, alpha_crop, offset = crop_to_component(source_rgb, comp, pad=8)

        components.append({
            "component_id": int(cid),
            "score": score,
            "features": feats,
            "rgb_crop": rgb_crop,
            "mask_crop": mask_crop,
            "alpha_crop": alpha_crop,
            "offset_xy": [int(offset[0]), int(offset[1])],
        })

    return components


def infer_head_tail_from_mask(alpha):
    mask = (alpha > 0.2).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        raise ValueError("Too few pixels to infer orientation.")

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)

    proj = centered @ axis
    pmin, pmax = float(proj.min()), float(proj.max())
    length = max(pmax - pmin, 1e-6)

    end_neg = mean + axis * pmin
    end_pos = mean + axis * pmax

    frac = 0.20
    neg_region = pts[proj <= pmin + frac * length]
    pos_region = pts[proj >= pmax - frac * length]

    def local_width(region):
        if len(region) < 5:
            return 1e9
        vals = (region - mean) @ perp
        return float(np.percentile(vals, 95) - np.percentile(vals, 5))

    w_neg = local_width(neg_region)
    w_pos = local_width(pos_region)

    if w_neg <= w_pos:
        head = end_neg
        tail = end_pos
    else:
        head = end_pos
        tail = end_neg

    angle = image_angle_deg(tail, head)
    return head.astype(np.float32), tail.astype(np.float32), angle


def sample_polyline(points, num):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 2:
        raise ValueError("Track needs at least 2 points.")
    segs = pts[1:] - pts[:-1]
    lens = np.sqrt((segs ** 2).sum(axis=1))
    total = lens.sum()
    if total <= 1e-6:
        return np.repeat(pts[:1], num, axis=0)

    dists = np.linspace(0, total, num)
    out = []
    cum = np.concatenate([[0], np.cumsum(lens)])

    for d in dists:
        idx = np.searchsorted(cum, d, side="right") - 1
        idx = min(max(idx, 0), len(lens) - 1)
        t = (d - cum[idx]) / max(lens[idx], 1e-6)
        out.append(pts[idx] * (1 - t) + pts[idx + 1] * t)
    return np.array(out, dtype=np.float32)


def rotate_and_scale_patch(rgb, alpha, cv2_angle_deg, scale=1.0):
    new_w = int(max(2, rgb.shape[1] * scale))
    new_h = int(max(2, rgb.shape[0] * scale))
    rgb_s = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    a_s = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w = rgb_s.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, cv2_angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    bound_w = int(h * sin + w * cos)
    bound_h = int(h * cos + w * sin)

    M[0, 2] += bound_w / 2.0 - center[0]
    M[1, 2] += bound_h / 2.0 - center[1]

    rgb_r = cv2.warpAffine(
        rgb_s, M, (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    a_r = cv2.warpAffine(
        a_s, M, (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    a_r = np.clip(a_r, 0, 1)
    return rgb_r, a_r


def add_simple_shadow(bg, alpha, center_xy, offset=(5, 5), strength=0.22):
    H, W = bg.shape[:2]
    h, w = alpha.shape[:2]
    cx, cy = int(round(center_xy[0] + offset[0])), int(round(center_xy[1] + offset[1]))
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = x1 + w, y1 + h

    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(W, x2), min(H, y2)
    if bx1 >= bx2 or by1 >= by2:
        return bg

    px1, py1 = bx1 - x1, by1 - y1
    px2, py2 = px1 + (bx2 - bx1), py1 + (by2 - by1)

    out = bg.copy().astype(np.float32)
    a = alpha[py1:py2, px1:px2].astype(np.float32)
    a = cv2.GaussianBlur(a, (9, 9), 0)[..., None] * strength
    roi = out[by1:by2, bx1:bx2]
    out[by1:by2, bx1:bx2] = roi * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_patch(bg, patch, alpha, center_xy):
    out = bg.copy()
    H, W = out.shape[:2]
    h, w = patch.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))

    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h

    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(W, x2), min(H, y2)

    if bx1 >= bx2 or by1 >= by2:
        return out, None

    px1, py1 = bx1 - x1, by1 - y1
    px2, py2 = px1 + (bx2 - bx1), py1 + (by2 - by1)

    roi = out[by1:by2, bx1:bx2].astype(np.float32)
    p = patch[py1:py2, px1:px2].astype(np.float32)
    a = alpha[py1:py2, px1:px2].astype(np.float32)[..., None]

    blended = roi * (1 - a) + p * a
    out[by1:by2, bx1:bx2] = blended.astype(np.uint8)

    visible = alpha[py1:py2, px1:px2] > 0.15
    if visible.any():
        ys, xs = np.where(visible)
        bbox = [
            int(bx1 + xs.min()),
            int(by1 + ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
    else:
        bbox = None

    return out, bbox


def draw_prototype_debug(comp, out_path):
    vis = comp["rgb_crop"].copy()
    mask = (comp["alpha_crop"] > 0.15).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    text = f"id={comp['component_id']} area={comp['features']['area']}"
    if comp["score"] is not None:
        text += f" score={comp['score']:.3f}"
    cv2.putText(vis, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(str(out_path), vis)


def draw_orientation_debug(comp, head, tail, out_path):
    vis = comp["rgb_crop"].copy()
    mask = (comp["alpha_crop"] > 0.15).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    h = tuple(np.round(head).astype(int))
    t = tuple(np.round(tail).astype(int))
    cv2.circle(vis, h, 5, (0, 0, 255), -1)
    cv2.circle(vis, t, 5, (255, 0, 0), -1)
    cv2.arrowedLine(vis, t, h, (0, 255, 0), 2, tipLength=0.25)
    cv2.putText(vis, "HEAD", (h[0] + 5, h[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    cv2.putText(vis, "TAIL", (t[0] + 5, t[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1)
    cv2.imwrite(str(out_path), vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True)
    parser.add_argument("--source-rgb", required=True)
    parser.add_argument("--source-mask", required=True)
    parser.add_argument("--tracks-json", required=True)
    parser.add_argument("--out-dir", default="sim_out")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)

    # tracking ground-truth output
    parser.add_argument("--write-mot", action="store_true",
                        help="Write MOTChallenge-style gt.txt and seqinfo.ini.")
    parser.add_argument("--mot-class-id", type=int, default=1,
                        help="Class id used in MOT gt. For single-class airplane tracking, use 1.")
    parser.add_argument("--mot-min-visibility", type=float, default=0.05,
                        help="Drop MOT boxes with visible fraction below this threshold.")

    parser.add_argument("--component-scores-json", default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--max-area", type=int, default=1000000)
    parser.add_argument("--min-aspect", type=float, default=1.2)
    parser.add_argument("--max-aspect", type=float, default=8.0)
    parser.add_argument("--min-solidity", type=float, default=0.20)
    parser.add_argument("--num-prototypes", type=int, default=None)
    parser.add_argument("--sample-with-replacement", action="store_true")

    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--use-random-scale", action="store_true")
    parser.add_argument("--random-scale-min", type=float, default=0.8)
    parser.add_argument("--random-scale-max", type=float, default=1.2)

    parser.add_argument("--start-frame-min", type=int, default=0)
    parser.add_argument("--start-frame-max", type=int, default=0)
    parser.add_argument("--duration-min", type=int, default=None)
    parser.add_argument("--duration-max", type=int, default=None)

    parser.add_argument("--debug-prototypes", action="store_true")
    parser.add_argument("--debug-orientation", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    proto_dir = out_dir / "debug_prototypes"
    orient_dir = out_dir / "debug_orientation"
    ensure_dir(out_dir)
    ensure_dir(frame_dir)
    if args.debug_prototypes:
        ensure_dir(proto_dir)
    if args.debug_orientation:
        ensure_dir(orient_dir)

    bg = read_image(args.background, cv2.IMREAD_COLOR)
    H, W = bg.shape[:2]
    source_rgb = read_image(args.source_rgb, cv2.IMREAD_COLOR)
    source_mask = read_image(args.source_mask, cv2.IMREAD_UNCHANGED)
    tracks_dict = load_json(args.tracks_json)
    track_items = list(tracks_dict.items())

    scores = load_json(args.component_scores_json) if args.component_scores_json else None

    components = extract_components(
        source_rgb=source_rgb,
        source_mask=source_mask,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_solidity=args.min_solidity,
        scores=scores,
        min_score=args.min_score
    )

    if len(components) == 0:
        raise RuntimeError("No valid prototype components found. Relax thresholds.")

    if args.num_prototypes is not None and args.num_prototypes < len(components):
        components = random.sample(components, args.num_prototypes)

    for comp in components:
        head, tail, obj_angle = infer_head_tail_from_mask(comp["alpha_crop"])
        comp["head_xy"] = head.tolist()
        comp["tail_xy"] = tail.tolist()
        comp["object_axis_angle_deg"] = float(obj_angle)

        if args.debug_prototypes:
            draw_prototype_debug(comp, proto_dir / f"component_{comp['component_id']:03d}.png")
        if args.debug_orientation:
            draw_orientation_debug(comp, head, tail, orient_dir / f"component_{comp['component_id']:03d}_axis.png")

    proto_summary = []
    for comp in components:
        proto_summary.append({
            "component_id": comp["component_id"],
            "score": comp["score"],
            "features": comp["features"],
            "head_xy": comp["head_xy"],
            "tail_xy": comp["tail_xy"],
            "object_axis_angle_deg": comp["object_axis_angle_deg"],
        })
    save_json(out_dir / "prototype_summary.json", proto_summary)

    assigned_objects = []
    for idx, (track_name, track_points) in enumerate(track_items):
        if len(track_points) < 2:
            raise ValueError(f"{track_name} needs at least 2 points.")
        if args.sample_with_replacement:
            comp = random.choice(components)
        else:
            comp = components[idx] if idx < len(components) else random.choice(components)

        start_frame = random.randint(args.start_frame_min, args.start_frame_max) if args.start_frame_max >= args.start_frame_min else 0

        duration_min = args.duration_min if args.duration_min is not None else (args.num_frames - start_frame)
        duration_max = args.duration_max if args.duration_max is not None else (args.num_frames - start_frame)
        duration_max = min(duration_max, args.num_frames - start_frame)
        duration_min = min(duration_min, duration_max)
        duration = random.randint(duration_min, duration_max) if duration_max >= duration_min else duration_max
        end_frame = min(args.num_frames - 1, start_frame + duration - 1)

        scale = random.uniform(args.random_scale_min, args.random_scale_max) if args.use_random-scale else 1.0

        n_active = max(1, end_frame - start_frame + 1)
        positions = sample_polyline(track_points, n_active)

        assigned_objects.append({
            "track_name": track_name,
            "track_points": track_points,
            "prototype_component_id": comp["component_id"],
            "rgb": comp["rgb_crop"],
            "alpha": comp["alpha_crop"],
            "obj_angle": comp["object_axis_angle_deg"],
            "scale": scale,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "positions": positions,
        })

    assign_summary = []
    for obj in assigned_objects:
        assign_summary.append({
            "track_name": obj["track_name"],
            "track_points": obj["track_points"],
            "prototype_component_id": obj["prototype_component_id"],
            "object_axis_angle_deg": obj["obj_angle"],
            "scale": obj["scale"],
            "start_frame": obj["start_frame"],
            "end_frame": obj["end_frame"],
        })
    save_json(out_dir / "assigned_objects.json", assign_summary)

    labels = []
    mot_rows = []

    # Map track names to stable integer IDs for tracking GT.
    track_id_map = {
        obj["track_name"]: i + 1
        for i, obj in enumerate(assigned_objects)
    }

    for fidx in range(args.num_frames):
        frame = bg.copy()
        frame_labels = []

        active = [o for o in assigned_objects if o["start_frame"] <= fidx <= o["end_frame"]]
        random.shuffle(active)

        for obj in active:
            local_idx = fidx - obj["start_frame"]
            positions = obj["positions"]
            pos = positions[min(local_idx, len(positions) - 1)]

            if len(positions) == 1:
                traj_angle = obj["obj_angle"]
            else:
                i0 = max(0, min(local_idx, len(positions) - 2))
                p1 = positions[i0]
                p2 = positions[i0 + 1]
                traj_angle = image_angle_deg(p1, p2)

            cv2_rot_angle = obj["obj_angle"] - traj_angle
            patch, alpha = rotate_and_scale_patch(obj["rgb"], obj["alpha"], cv2_rot_angle, obj["scale"])

            if args.shadow:
                frame = add_simple_shadow(frame, alpha, pos)

            frame, bbox = paste_patch(frame, patch, alpha, pos)

            if bbox is not None:
                frame_labels.append({
                    "frame": fidx,
                    "track_name": obj["track_name"],
                    "track_id": track_id_map[obj["track_name"]],
                    "prototype_component_id": obj["prototype_component_id"],
                    "bbox_xywh": bbox,
                    "center_xy": [float(pos[0]), float(pos[1])],
                    "trajectory_angle_deg": float(traj_angle),
                    "object_axis_angle_deg": float(obj["obj_angle"]),
                    "cv2_rotation_angle_deg": float(cv2_rot_angle),
                })

                if args.write_mot:
                    vis_frac, clipped_bbox = bbox_visible_fraction(bbox, W, H)
                    if clipped_bbox is not None and vis_frac >= args.mot_min_visibility:
                        # MOTChallenge gt format:
                        # frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
                        # MOT frames are 1-indexed.
                        x, y, bw, bh = clipped_bbox
                        mot_rows.append([
                            fidx + 1,
                            track_id_map[obj["track_name"]],
                            x,
                            y,
                            bw,
                            bh,
                            1,
                            args.mot_class_id,
                            round(float(vis_frac), 6),
                        ])

        labels.extend(frame_labels)
        cv2.imwrite(str(frame_dir / f"frame_{fidx:04d}.png"), frame)

    save_json(out_dir / "labels.json", labels)

    if args.write_mot:
        mot_gt_dir = out_dir / "gt"
        ensure_dir(mot_gt_dir)
        mot_path = mot_gt_dir / "gt.txt"
        with open(mot_path, "w") as f:
            for row in mot_rows:
                f.write(",".join(map(str, row)) + "\n")

        seqinfo_path = out_dir / "seqinfo.ini"
        seq_name = out_dir.name
        with open(seqinfo_path, "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name={seq_name}\n")
            f.write("imDir=frames\n")
            f.write(f"frameRate={args.fps}\n")
            f.write(f"seqLength={args.num_frames}\n")
            f.write(f"imWidth={W}\n")
            f.write(f"imHeight={H}\n")
            f.write("imExt=.png\n")

    video_path = out_dir / "simulation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (W, H))
    for fidx in range(args.num_frames):
        frame = read_image(frame_dir / f"frame_{fidx:04d}.png")
        writer.write(frame)
    writer.release()

    print(f"Valid prototype components found: {len(components)}")
    print(f"Saved extracted prototype summary to: {out_dir / 'prototype_summary.json'}")
    print(f"Saved assigned object summary to:    {out_dir / 'assigned_objects.json'}")
    print(f"Saved labels to:                    {out_dir / 'labels.json'}")
    if args.write_mot:
        print(f"Saved MOT gt to:                    {out_dir / 'gt' / 'gt.txt'}")
        print(f"Saved MOT seqinfo to:               {out_dir / 'seqinfo.ini'}")
    print(f"Saved frames to:                    {frame_dir}")
    print(f"Saved video to:                     {video_path}")
    if args.debug_prototypes:
        print(f"Saved prototype debug images to:    {proto_dir}")
    if args.debug_orientation:
        print(f"Saved orientation debug images to:  {orient_dir}")


if __name__ == "__main__":
    main()
