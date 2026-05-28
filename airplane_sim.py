#!/usr/bin/env python3
"""
Simulate airplanes moving along manually-defined tracks on an airport background.

This version fixes airplane direction alignment:

1. You may provide manual head/tail points for each airplane asset.
   The canonical airplane direction is tail -> head.

2. If head/tail are not provided, the script estimates the body axis from the
   segmentation mask using PCA and a simple nose/tail heuristic.

3. IMPORTANT OpenCV detail:
   cv2.getRotationMatrix2D uses a rotation sign opposite to image-coordinate
   atan2(y_down, x_right). To align object_angle to trajectory_angle, this script
   uses:
       cv2_rotation_angle = object_angle - trajectory_angle
   not:
       trajectory_angle - object_angle

Inputs:
  --background airport background image
  --assets-json object prototype definitions
  --tracks-json manually picked track start/end or polyline points
  --out-dir output folder

Example:
  python simulate_airplanes_v2.py \
    --background airport_bg.png \
    --assets-json assets.json \
    --tracks-json tracks.json \
    --out-dir sim_out \
    --num-frames 120 \
    --fps 24 \
    --shadow \
    --debug-orientation

tracks.json example:
{
  "track1": [[787,471], [1864,538]],
  "track2": [[1699,560], [853,499]],
  "track3": [[1022,763], [1691,811]],
  "track4": [[1290,1016], [1320,532]],
  "track5": [[983,605], [1868,666]],
  "track6": [[980,766], [1007,438]],
  "track7": [[1332,518], [1872,549]]
}

assets.json example, manual orientation:
{
  "airplanes": [
    {
      "id": "plane_01",
      "class": "airplane",
      "rgb": "plane_01_rgb.png",
      "mask": "plane_01_mask.png",
      "track": "track1",
      "start_frame": 0,
      "end_frame": 119,
      "scale": 1.0,
      "head": [40, 8],
      "tail": [40, 80]
    }
  ]
}

assets.json example, automatic orientation:
{
  "airplanes": [
    {
      "id": "plane_01",
      "class": "airplane",
      "rgb": "plane_01_rgb.png",
      "mask": "plane_01_mask.png",
      "track": "track1",
      "start_frame": 0,
      "end_frame": 119,
      "scale": 1.0
    }
  ]
}

If automatic nose/tail is flipped, add:
  "flip_axis": true
or provide manual "head" and "tail", which is the most reliable.
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_alpha(mask):
    """Convert mask to float alpha [0,1], feathered slightly."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    return alpha


def crop_to_mask(rgb, mask_alpha, pad=6):
    ys, xs = np.where(mask_alpha > 0.05)
    if len(xs) == 0:
        raise ValueError("Empty mask.")
    h, w = mask_alpha.shape[:2]
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    return rgb[y1:y2, x1:x2].copy(), mask_alpha[y1:y2, x1:x2].copy(), (x1, y1)


def image_angle_deg(p1, p2):
    """
    Angle in image coordinates:
      x right is 0 degrees
      y down is +90 degrees
    """
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.degrees(math.atan2(dy, dx))


def sample_polyline(points, num):
    """Sample num points along a polyline by arc length."""
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


def infer_head_tail_from_mask(alpha, flip=False):
    """
    Estimate airplane body axis from the mask.

    Returns:
      head_xy, tail_xy, object_angle_deg

    Method:
      - Use PCA on foreground mask pixels.
      - Longest principal component is body axis.
      - Project contour pixels onto the axis.
      - Two extreme endpoints are possible head/tail.
      - Nose heuristic: the nose side is usually narrower than the tail side.
        We compare perpendicular width near each end.
      - If wrong, set flip_axis=true in assets.json, or provide manual head/tail.
    """
    mask = (alpha > 0.2).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        raise ValueError("Too few mask pixels to infer orientation.")

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

    # Estimate local width near both ends.
    # Use first/last 20% of projected length.
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

    # Smaller local width is treated as the nose/head.
    if w_neg <= w_pos:
        head = end_neg
        tail = end_pos
    else:
        head = end_pos
        tail = end_neg

    if flip:
        head, tail = tail, head

    angle = image_angle_deg(tail, head)
    return head.astype(np.float32), tail.astype(np.float32), angle


def draw_orientation_debug(rgb, alpha, head, tail, out_path):
    vis = rgb.copy()
    mask = (alpha > 0.15).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    h = tuple(np.round(head).astype(int))
    t = tuple(np.round(tail).astype(int))
    cv2.circle(vis, h, 5, (0, 0, 255), -1)      # head = red
    cv2.circle(vis, t, 5, (255, 0, 0), -1)      # tail = blue
    cv2.arrowedLine(vis, t, h, (0, 255, 0), 2, tipLength=0.25)
    cv2.putText(vis, "HEAD", (h[0] + 5, h[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(vis, "TAIL", (t[0] + 5, t[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.imwrite(str(out_path), vis)


def rotate_and_scale_patch(rgb, alpha, cv2_angle_deg, scale=1.0):
    """
    Rotate image patch using OpenCV rotation angle.
    Note: this angle is not the same sign as image-coordinate atan2 angle.
    """
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


def paste_patch(bg, patch, alpha, center_xy):
    """Alpha paste patch at center_xy=(x,y)."""
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

    aa = alpha[py1:py2, px1:px2] > 0.15
    if aa.any():
        ys, xs = np.where(aa)
        bbox = [
            int(bx1 + xs.min()),
            int(by1 + ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
    else:
        bbox = None

    return out, bbox


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True)
    parser.add_argument("--assets-json", required=True)
    parser.add_argument("--tracks-json", required=True)
    parser.add_argument("--out-dir", default="sim_out")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-scale-min", type=float, default=0.8)
    parser.add_argument("--random-scale-max", type=float, default=1.2)
    parser.add_argument("--use-random-scale", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--debug-orientation", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    debug_dir = out_dir / "debug_orientation"
    ensure_dir(frame_dir)
    if args.debug_orientation:
        ensure_dir(debug_dir)

    bg = read_image(args.background)
    H, W = bg.shape[:2]
    tracks = load_json(args.tracks_json)
    assets = load_json(args.assets_json)["airplanes"]

    objects = []
    for idx, obj in enumerate(assets):
        rgb_path = Path(obj["rgb"])
        mask_path = Path(obj["mask"])
        rgb = read_image(rgb_path, cv2.IMREAD_COLOR)
        mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        alpha = make_alpha(mask)

        crop_rgb, crop_alpha, crop_offset = crop_to_mask(rgb, alpha, pad=8)

        if "head" in obj and "tail" in obj:
            # User-provided head/tail are in original asset coordinates.
            head = np.array(obj["head"], dtype=np.float32) - np.array(crop_offset, dtype=np.float32)
            tail = np.array(obj["tail"], dtype=np.float32) - np.array(crop_offset, dtype=np.float32)
            obj_angle = image_angle_deg(tail, head)
            orient_source = "manual"
        else:
            # Auto estimate from cropped mask.
            head, tail, obj_angle = infer_head_tail_from_mask(
                crop_alpha,
                flip=bool(obj.get("flip_axis", False))
            )
            orient_source = "auto_pca"

        if args.debug_orientation:
            name = obj.get("id", f"plane_{idx:02d}")
            draw_orientation_debug(
                crop_rgb, crop_alpha, head, tail,
                debug_dir / f"{name}_{orient_source}_axis.png"
            )

        if args.use_random_scale:
            scale = random.uniform(args.random_scale_min, args.random_scale_max)
        else:
            scale = float(obj.get("scale", 1.0))

        start = int(obj.get("start_frame", 0))
        end = int(obj.get("end_frame", args.num_frames - 1))
        end = min(end, args.num_frames - 1)

        track_name = obj["track"]
        if track_name not in tracks:
            raise KeyError(f"{track_name} not found in tracks-json")

        track_points = tracks[track_name]
        n_active = max(1, end - start + 1)
        positions = sample_polyline(track_points, n_active)

        objects.append({
            "id": obj.get("id", f"obj_{idx}"),
            "class": obj.get("class", "airplane"),
            "rgb": crop_rgb,
            "alpha": crop_alpha,
            "obj_angle": obj_angle,
            "scale": scale,
            "start": start,
            "end": end,
            "positions": positions,
            "orientation_source": orient_source,
        })

    labels = []

    for fidx in range(args.num_frames):
        frame = bg.copy()
        frame_labels = []

        active = [o for o in objects if o["start"] <= fidx <= o["end"]]
        random.shuffle(active)

        for obj in active:
            local_idx = fidx - obj["start"]
            positions = obj["positions"]
            pos = positions[min(local_idx, len(positions) - 1)]

            if len(positions) == 1:
                traj_angle = obj["obj_angle"]
            else:
                i0 = max(0, min(local_idx, len(positions) - 2))
                p1 = positions[i0]
                p2 = positions[i0 + 1]
                traj_angle = image_angle_deg(p1, p2)

            # CRITICAL FIX:
            # image-coordinate angle uses y-down.
            # OpenCV positive rotation has opposite sign.
            # To make object_angle become traj_angle, use object_angle - traj_angle.
            cv2_rot_angle = obj["obj_angle"] - traj_angle

            patch, alpha = rotate_and_scale_patch(
                obj["rgb"], obj["alpha"], cv2_rot_angle, obj["scale"]
            )

            if args.shadow:
                frame = add_simple_shadow(frame, alpha, pos)

            frame, bbox = paste_patch(frame, patch, alpha, pos)

            if bbox is not None:
                frame_labels.append({
                    "frame": fidx,
                    "track_id": obj["id"],
                    "class": obj["class"],
                    "bbox_xywh": bbox,
                    "center_xy": [float(pos[0]), float(pos[1])],
                    "trajectory_angle_deg": float(traj_angle),
                    "object_axis_angle_deg": float(obj["obj_angle"]),
                    "cv2_rotation_angle_deg": float(cv2_rot_angle),
                    "orientation_source": obj["orientation_source"],
                })

        labels.extend(frame_labels)
        cv2.imwrite(str(frame_dir / f"frame_{fidx:04d}.png"), frame)

    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    video_path = out_dir / "simulation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (W, H))
    for fidx in range(args.num_frames):
        frame = read_image(frame_dir / f"frame_{fidx:04d}.png")
        writer.write(frame)
    writer.release()

    print(f"Saved frames to: {frame_dir}")
    print(f"Saved video to:  {video_path}")
    print(f"Saved labels to: {out_dir / 'labels.json'}")
    if args.debug_orientation:
        print(f"Saved orientation debug images to: {debug_dir}")


if __name__ == "__main__":
    main()
