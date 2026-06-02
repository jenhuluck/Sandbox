#!/usr/bin/env python3
"""
Generate frame-level reliability / uncertainty labels for multi-object tracker output
by matching predicted tracks to ground-truth tracks.

Main ideas:
  - Match predictions to GT per frame using Hungarian matching on IoU.
  - Maintain history for each predicted track ID.
  - Label each (frame, predicted_track_id) row as:
        reliable, initialization_matched, initialization_unmatched,
        id_switch, uncertain, false_positive
  - Keep category matching configurable with --match_category.
  - Avoid labeling initialization as false positive by default.
  - Use consecutive same-GT matches for reliability, not global match ratio.

Expected GT COCO fields:
  images[*].id
  images[*].mot_frame_id OR images[*].frame_id OR images[*].file_name fallback
  annotations[*].image_id
  annotations[*].bbox                 # COCO [x, y, w, h]
  annotations[*].category_id
  annotations[*].mot_instance_id OR track_id OR instance_id OR id

Expected prediction JSON formats supported:
  A. Dict keyed by frame id:
       {
         "10": [
           {"track_id": 7, "tlwh": [x,y,w,h], "class_id": 1, "score": 0.91},
           ...
         ]
       }
  B. COCO-like list/dict of detections:
       [
         {"image_id": 10, "track_id": 7, "bbox": [x,y,w,h], "category_id": 1, "score": 0.91},
         ...
       ]

Example:
  python generate_track_labels.py \
      --gt gt.json \
      --pred pred_meta.json \
      --output track_frame_labels.csv \
      --sequence 1-03 \
      --iou_thr 0.5 \
      --n_confirm 5 \
      --init_window 3 \
      --fp_threshold 10

Try category-aware matching:
  python generate_track_labels.py ... --match_category
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

Box = List[float]
TrackId = Union[int, str]
GtId = Union[int, str]


# -----------------------------------------------------------------------------
# Box utilities
# -----------------------------------------------------------------------------

def coco_to_xyxy(box: Sequence[float]) -> Box:
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = map(float, box[:4])
    return [x, y, x + w, y + h]


def xyxy_to_coco(box: Sequence[float]) -> Box:
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h]."""
    x1, y1, x2, y2 = map(float, box[:4])
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def normalize_box(box: Sequence[float], box_format: str = "coco") -> Box:
    """Return box as [x1, y1, x2, y2]."""
    if len(box) < 4:
        raise ValueError(f"Box must contain 4 values, got: {box}")

    if box_format == "coco":
        return coco_to_xyxy(box)
    if box_format == "xyxy":
        x1, y1, x2, y2 = map(float, box[:4])
        return [x1, y1, x2, y2]

    raise ValueError(f"Unsupported box_format={box_format!r}. Use 'coco' or 'xyxy'.")


def compute_iou(
    box1: Sequence[float],
    box2: Sequence[float],
    box1_format: str = "coco",
    box2_format: str = "coco",
) -> float:
    """Compute IoU between two boxes with explicit formats."""
    b1 = normalize_box(box1, box1_format)
    b2 = normalize_box(box2, box2_format)

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def category_matches(pred_class: Any, gt_class: Any, allow_off_by_one: bool = True) -> bool:
    """Return whether prediction and GT classes match.

    allow_off_by_one is useful when one file uses 0-index classes and the other uses
    1-index classes.
    """
    if pred_class == gt_class:
        return True
    if not allow_off_by_one:
        return False

    try:
        p = int(pred_class)
        g = int(gt_class)
        return p == g - 1 or p + 1 == g
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class GtObject:
    gt_id: GtId
    bbox: Box
    category_id: Any
    image_id: Any = None


@dataclass
class PredObject:
    track_id: TrackId
    bbox: Box
    category_id: Any
    score: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackState:
    current_gt_id: Optional[GtId] = None
    previous_gt_id: Optional[GtId] = None
    matched_same_gt_count: int = 0          # consecutive same-GT match count
    unmatched_count: int = 0                # consecutive unmatched count
    track_age: int = 0                      # number of predicted frames seen
    id_switch_frame: Optional[int] = None
    switched_from_gt_id: Optional[GtId] = None
    switched_to_gt_id: Optional[GtId] = None
    history: List[Tuple[int, Optional[GtId], float]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def _safe_frame_from_file_name(file_name: str) -> Optional[int]:
    """Try to extract a frame number from a file name."""
    stem = Path(file_name).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if not digits:
        return None
    return int(digits[-1])


def build_image_id_to_frame(images: List[Dict[str, Any]]) -> Dict[Any, int]:
    image_id_to_frame: Dict[Any, int] = {}
    for img in images:
        image_id = img.get("id")
        if image_id is None:
            continue

        if "mot_frame_id" in img:
            frame_id = int(img["mot_frame_id"])
        elif "frame_id" in img:
            frame_id = int(img["frame_id"])
        elif "frame" in img:
            frame_id = int(img["frame"])
        elif "file_name" in img:
            parsed = _safe_frame_from_file_name(str(img["file_name"]))
            frame_id = int(image_id) if parsed is None else parsed
        else:
            frame_id = int(image_id)

        image_id_to_frame[image_id] = frame_id
    return image_id_to_frame


def load_gt_annotations(
    gt_path: Union[str, Path],
    gt_box_format: str = "coco",
    crop_region: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[int, List[GtObject]]:
    """Load GT annotations from COCO-style JSON."""
    gt_path = Path(gt_path)
    with gt_path.open("r") as f:
        gt_data = json.load(f)

    if "annotations" not in gt_data:
        raise ValueError("GT JSON must contain an 'annotations' field.")

    image_id_to_frame = build_image_id_to_frame(gt_data.get("images", []))
    gt_by_frame: Dict[int, List[GtObject]] = defaultdict(list)

    total = 0
    filtered = 0
    missing_track_id = 0

    for ann in gt_data["annotations"]:
        total += 1
        image_id = ann.get("image_id")
        if image_id not in image_id_to_frame:
            # Fallback: image_id itself as frame id.
            frame_id = int(image_id)
        else:
            frame_id = image_id_to_frame[image_id]

        gt_id = (
            ann.get("mot_instance_id")
            if ann.get("mot_instance_id") is not None
            else ann.get("track_id", ann.get("instance_id", ann.get("id")))
        )
        if gt_id is None:
            missing_track_id += 1
            gt_id = ann.get("id", f"gt_missing_{missing_track_id}")

        bbox = list(map(float, ann["bbox"][:4]))
        class_id = ann.get("category_id", ann.get("class_id", -1))

        if crop_region is not None:
            x_min, y_min, x_max, y_max = crop_region
            x1, y1, x2, y2 = normalize_box(bbox, gt_box_format)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                filtered += 1
                continue

        gt_by_frame[frame_id].append(GtObject(gt_id=gt_id, bbox=bbox, category_id=class_id, image_id=image_id))

    print(f"Loaded GT: {total} annotations across {len(gt_by_frame)} frames from {gt_path}")
    if crop_region is not None:
        print(f"Filtered GT outside crop region: {filtered}/{total}")
    if missing_track_id:
        print(f"Warning: {missing_track_id} GT annotations had no explicit track ID; annotation id was used.")

    return gt_by_frame


def _frame_from_prediction_item(item: Dict[str, Any]) -> int:
    for key in ("frame_id", "frame", "mot_frame_id", "image_id"):
        if key in item and item[key] is not None:
            return int(item[key])
    raise ValueError(f"Prediction item has no frame/image id: {item}")


def _parse_prediction_track(item: Dict[str, Any], temp_id: int) -> Tuple[PredObject, int]:
    track_id = item.get("track_id", item.get("id", item.get("pred_track_id")))
    if track_id is None:
        track_id = f"temp_{temp_id}"
        temp_id += 1

    if "tlwh" in item:
        bbox = list(map(float, item["tlwh"][:4]))
    elif "bbox" in item:
        bbox = list(map(float, item["bbox"][:4]))
    else:
        raise ValueError(f"Prediction item has no 'tlwh' or 'bbox': {item}")

    category_id = item.get("category_id", item.get("class_id", item.get("label", -1)))
    score = item.get("score", item.get("confidence", item.get("conf")))
    if score is not None:
        score = float(score)

    pred = PredObject(track_id=track_id, bbox=bbox, category_id=category_id, score=score, raw=item)
    return pred, temp_id


def load_predictions(
    pred_path: Union[str, Path],
    coord_offset_x: float = 0.0,
    coord_offset_y: float = 0.0,
) -> Dict[int, List[PredObject]]:
    """Load prediction file.

    Supports a dict keyed by frame id or a COCO-like list/dict of prediction items.
    Prediction boxes are expected in COCO/TLWH format by default.
    """
    pred_path = Path(pred_path)
    with pred_path.open("r") as f:
        pred_data = json.load(f)

    pred_by_frame: Dict[int, List[PredObject]] = defaultdict(list)
    next_temp_id = 1_000_000

    if isinstance(pred_data, dict) and all(isinstance(v, list) for v in pred_data.values()):
        # Format: {"frame": [track, track, ...]}
        for frame_str, tracks in pred_data.items():
            frame_id = int(frame_str)
            for item in tracks:
                pred, next_temp_id = _parse_prediction_track(item, next_temp_id)
                pred.bbox[0] += coord_offset_x
                pred.bbox[1] += coord_offset_y
                pred_by_frame[frame_id].append(pred)
    else:
        # COCO-like: either list or dict with annotations/predictions/results.
        if isinstance(pred_data, list):
            items = pred_data
        elif isinstance(pred_data, dict):
            items = pred_data.get("annotations") or pred_data.get("predictions") or pred_data.get("results")
            if items is None:
                raise ValueError(
                    "Unsupported prediction dict format. Expected frame-keyed dict or an "
                    "'annotations'/'predictions'/'results' list."
                )
        else:
            raise ValueError("Unsupported prediction JSON type.")

        for item in items:
            frame_id = _frame_from_prediction_item(item)
            pred, next_temp_id = _parse_prediction_track(item, next_temp_id)
            pred.bbox[0] += coord_offset_x
            pred.bbox[1] += coord_offset_y
            pred_by_frame[frame_id].append(pred)

    print(f"Loaded predictions: {sum(len(v) for v in pred_by_frame.values())} boxes across {len(pred_by_frame)} frames from {pred_path}")
    return pred_by_frame


# -----------------------------------------------------------------------------
# Matching and labeling
# -----------------------------------------------------------------------------

def hungarian_matching(
    preds: List[PredObject],
    gts: List[GtObject],
    iou_threshold: float = 0.5,
    match_category: bool = False,
    pred_box_format: str = "coco",
    gt_box_format: str = "coco",
    allow_off_by_one_category: bool = True,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Match predictions to GT with Hungarian assignment.

    Returns:
      matches: list of (pred_index, gt_index, iou)
      unmatched_pred_indices
      unmatched_gt_indices
    """
    n_pred = len(preds)
    n_gt = len(gts)

    if n_pred == 0 or n_gt == 0:
        return [], list(range(n_pred)), list(range(n_gt))

    # Large cost means impossible category match.
    cost_matrix = np.full((n_pred, n_gt), fill_value=1e6, dtype=float)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=float)

    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            if match_category and not category_matches(pred.category_id, gt.category_id, allow_off_by_one_category):
                continue
            iou = compute_iou(pred.bbox, gt.bbox, pred_box_format, gt_box_format)
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int, float]] = []
    matched_preds = set()
    matched_gts = set()

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= 1e6:
            continue
        iou = float(iou_matrix[i, j])
        if iou >= iou_threshold:
            matches.append((i, j, iou))
            matched_preds.add(i)
            matched_gts.add(j)

    unmatched_preds = [i for i in range(n_pred) if i not in matched_preds]
    unmatched_gts = [j for j in range(n_gt) if j not in matched_gts]
    return matches, unmatched_preds, unmatched_gts


class TrackHistory:
    def __init__(self, id_switch_recent_window: int = 5):
        self.tracks: Dict[TrackId, TrackState] = defaultdict(TrackState)
        self.id_switch_recent_window = id_switch_recent_window

    def update(self, frame: int, track_id: TrackId, matched_gt_id: Optional[GtId], iou: float) -> TrackState:
        state = self.tracks[track_id]
        state.track_age += 1
        state.history.append((frame, matched_gt_id, iou))

        if matched_gt_id is None:
            state.unmatched_count += 1
            return state

        # Currently matched.
        state.unmatched_count = 0

        if state.current_gt_id is None:
            # First valid GT association.
            state.current_gt_id = matched_gt_id
            state.previous_gt_id = None
            state.matched_same_gt_count = 1
            return state

        if state.current_gt_id == matched_gt_id:
            # Stable consecutive same-GT association.
            state.matched_same_gt_count += 1
            return state

        # ID switch: same predicted track now maps to a different GT.
        state.previous_gt_id = state.current_gt_id
        state.switched_from_gt_id = state.current_gt_id
        state.switched_to_gt_id = matched_gt_id
        state.current_gt_id = matched_gt_id
        state.id_switch_frame = frame
        state.matched_same_gt_count = 1
        return state

    def get_label(
        self,
        frame: int,
        track_id: TrackId,
        matched_gt_id: Optional[GtId],
        n_confirm: int,
        init_window: int,
        fp_threshold: int,
    ) -> Tuple[str, int]:
        """Assign label after history update.

        binary_label convention:
            1  reliable
            0  unreliable / negative
           -1  ignore by default, useful for initialization rows
        """
        state = self.tracks[track_id]

        # 1. ID switch has highest priority, even during early frames.
        if state.id_switch_frame is not None and frame == state.id_switch_frame:
            return "id_switch", 0

        if state.id_switch_frame is not None:
            frames_since_switch = frame - state.id_switch_frame
            if 0 <= frames_since_switch <= self.id_switch_recent_window:
                return "id_switch", 0

        # 2. Initialization should not automatically be negative.
        # Split matched/unmatched initialization for cleaner analysis.
        if state.track_age <= init_window:
            if matched_gt_id is not None:
                return "initialization_matched", -1
            return "initialization_unmatched", -1

        # 3. Currently matched.
        if matched_gt_id is not None:
            if state.matched_same_gt_count >= n_confirm:
                return "reliable", 1
            return "uncertain", 0

        # 4. Currently unmatched.
        if state.unmatched_count >= fp_threshold and state.track_age > init_window:
            return "false_positive", 0

        return "uncertain", 0


# -----------------------------------------------------------------------------
# Future stability
# -----------------------------------------------------------------------------

def compute_future_stability(
    df: pd.DataFrame,
    future_k: int = 5,
    future_ratio: float = 0.8,
) -> pd.DataFrame:
    """Add future_stable target.

    For each row matched to GT g, look ahead future_k appearances of the same predicted
    track. future_stable=1 if >= future_ratio of those rows are matched to the same GT g.
    If no future rows exist, future_stable=-1.
    """
    if df.empty:
        df["future_stable"] = []
        return df

    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    future_stable = np.full(len(df), -1, dtype=int)

    for _, idxs in df.groupby("track_id", sort=False).groups.items():
        idx_list = list(idxs)
        for pos, idx in enumerate(idx_list):
            gt_id = df.at[idx, "matched_gt_id"]
            if gt_id == -1 or pd.isna(gt_id):
                continue

            future_idxs = idx_list[pos + 1 : pos + 1 + future_k]
            if not future_idxs:
                future_stable[idx] = -1
                continue

            same = 0
            valid = 0
            for fidx in future_idxs:
                valid += 1
                if df.at[fidx, "matched_gt_id"] == gt_id:
                    same += 1

            ratio = same / max(1, valid)
            future_stable[idx] = 1 if ratio >= future_ratio else 0

    df["future_stable"] = future_stable
    return df


# -----------------------------------------------------------------------------
# Optional attribute merge
# -----------------------------------------------------------------------------

def merge_attributes(labels_df: pd.DataFrame, attr_file: Optional[Union[str, Path]]) -> pd.DataFrame:
    """Merge optional tracker attributes by frame and track_id.

    Attribute file may be CSV or JSON. It should contain frame and track_id columns/keys.
    """
    if attr_file is None:
        return labels_df

    attr_path = Path(attr_file)
    if not attr_path.exists():
        raise FileNotFoundError(attr_path)

    if attr_path.suffix.lower() == ".csv":
        attr_df = pd.read_csv(attr_path)
    elif attr_path.suffix.lower() in {".json", ".jsonl"}:
        if attr_path.suffix.lower() == ".jsonl":
            attr_df = pd.read_json(attr_path, lines=True)
        else:
            attr_df = pd.read_json(attr_path)
    else:
        raise ValueError("attr_file must be .csv, .json, or .jsonl")

    # Normalize common frame column names.
    rename = {}
    if "frame_id" in attr_df.columns and "frame" not in attr_df.columns:
        rename["frame_id"] = "frame"
    if "predicted_track_id" in attr_df.columns and "track_id" not in attr_df.columns:
        rename["predicted_track_id"] = "track_id"
    attr_df = attr_df.rename(columns=rename)

    if "frame" not in attr_df.columns or "track_id" not in attr_df.columns:
        raise ValueError("Attribute file must contain frame/frame_id and track_id/predicted_track_id columns.")

    # Cast to string for robust merging when track_id is mixed int/string.
    left = labels_df.copy()
    right = attr_df.copy()
    left["_merge_track_id"] = left["track_id"].astype(str)
    right["_merge_track_id"] = right["track_id"].astype(str)

    merged = left.merge(
        right.drop(columns=["track_id"], errors="ignore"),
        how="left",
        left_on=["frame", "_merge_track_id"],
        right_on=["frame", "_merge_track_id"],
        suffixes=("", "_attr"),
    ).drop(columns=["_merge_track_id"], errors="ignore")

    return merged


# -----------------------------------------------------------------------------
# Main generation
# -----------------------------------------------------------------------------

def format_box(box: Optional[Sequence[float]]) -> str:
    if box is None:
        return ""
    return "[" + ",".join(f"{float(v):.2f}" for v in box[:4]) + "]"


def generate_labels(
    gt_path: Union[str, Path],
    pred_path: Union[str, Path],
    output: Union[str, Path],
    sequence_name: str,
    attr_file: Optional[Union[str, Path]] = None,
    iou_thr: float = 0.5,
    n_confirm: int = 5,
    init_window: int = 3,
    fp_threshold: int = 10,
    id_switch_recent_window: int = 5,
    future_k: int = 5,
    future_ratio: float = 0.8,
    match_category: bool = False,
    allow_off_by_one_category: bool = True,
    pred_box_format: str = "coco",
    gt_box_format: str = "coco",
    coord_offset_x: float = 0.0,
    coord_offset_y: float = 0.0,
    crop_region: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    gt_by_frame = load_gt_annotations(gt_path, gt_box_format=gt_box_format, crop_region=crop_region)
    pred_by_frame = load_predictions(pred_path, coord_offset_x=coord_offset_x, coord_offset_y=coord_offset_y)

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    if not all_frames:
        raise ValueError("No frames found in GT or prediction files.")

    print(f"Processing {len(all_frames)} frames: {all_frames[0]} to {all_frames[-1]}")
    print(f"match_category={match_category}, iou_thr={iou_thr}")

    history = TrackHistory(id_switch_recent_window=id_switch_recent_window)
    rows: List[Dict[str, Any]] = []

    for frame_id in all_frames:
        gts = gt_by_frame.get(frame_id, [])
        preds = pred_by_frame.get(frame_id, [])
        if not preds:
            continue

        matches, unmatched_pred_idxs, unmatched_gt_idxs = hungarian_matching(
            preds=preds,
            gts=gts,
            iou_threshold=iou_thr,
            match_category=match_category,
            pred_box_format=pred_box_format,
            gt_box_format=gt_box_format,
            allow_off_by_one_category=allow_off_by_one_category,
        )

        match_dict: Dict[int, Tuple[int, float]] = {pred_i: (gt_j, iou) for pred_i, gt_j, iou in matches}

        for pred_idx, pred in enumerate(preds):
            matched_gt: Optional[GtObject] = None
            iou = 0.0
            if pred_idx in match_dict:
                gt_idx, iou = match_dict[pred_idx]
                matched_gt = gts[gt_idx]

            matched_gt_id = matched_gt.gt_id if matched_gt is not None else None
            state = history.update(frame_id, pred.track_id, matched_gt_id, iou)
            label, binary_label = history.get_label(
                frame=frame_id,
                track_id=pred.track_id,
                matched_gt_id=matched_gt_id,
                n_confirm=n_confirm,
                init_window=init_window,
                fp_threshold=fp_threshold,
            )

            rows.append(
                {
                    "sequence": sequence_name,
                    "frame": frame_id,
                    "track_id": pred.track_id,
                    "matched_gt_id": matched_gt_id if matched_gt_id is not None else -1,
                    "iou": round(float(iou), 6),
                    "pred_bbox": format_box(pred.bbox),
                    "gt_bbox": format_box(matched_gt.bbox if matched_gt is not None else None),
                    "pred_category_id": pred.category_id,
                    "gt_category_id": matched_gt.category_id if matched_gt is not None else -1,
                    "score": pred.score if pred.score is not None else np.nan,
                    "track_age": state.track_age,
                    "matched_same_gt_count": state.matched_same_gt_count,
                    "unmatched_count": state.unmatched_count,
                    "switched_from_gt_id": state.switched_from_gt_id if state.switched_from_gt_id is not None else -1,
                    "switched_to_gt_id": state.switched_to_gt_id if state.switched_to_gt_id is not None else -1,
                    "id_switch_frame": state.id_switch_frame if state.id_switch_frame is not None else -1,
                    "label": label,
                    "binary_label": binary_label,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print("Warning: no prediction rows produced.")
    else:
        df = compute_future_stability(df, future_k=future_k, future_ratio=future_ratio)
        df = merge_attributes(df, attr_file)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved frame labels to: {output}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print("\nLabel distribution:")
        print(df["label"].value_counts(dropna=False))
        print("\nBinary label distribution:")
        print(df["binary_label"].value_counts(dropna=False))
        print("\nFuture stability distribution:")
        print(df["future_stable"].value_counts(dropna=False))

    return df


def parse_crop_region(args: argparse.Namespace) -> Optional[Tuple[float, float, float, float]]:
    crop_values = [args.crop_x_min, args.crop_y_min, args.crop_x_max, args.crop_y_max]
    if all(v is None for v in crop_values):
        return None
    if args.crop_x_min is None or args.crop_x_max is None:
        raise ValueError("If using crop, provide at least --crop_x_min and --crop_x_max.")
    y_min = 0.0 if args.crop_y_min is None else args.crop_y_min
    y_max = 1e9 if args.crop_y_max is None else args.crop_y_max
    return (args.crop_x_min, y_min, args.crop_x_max, y_max)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frame-level GT labels for tracker outputs.")
    parser.add_argument("--gt", required=True, help="Path to GT COCO JSON.")
    parser.add_argument("--pred", required=True, help="Path to tracker prediction JSON.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--sequence", default="sequence", help="Sequence name saved in output CSV.")
    parser.add_argument("--attr_file", default=None, help="Optional tracker attribute CSV/JSON/JSONL to merge by frame and track_id.")

    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for valid pred-GT match.")
    parser.add_argument("--n_confirm", type=int, default=5, help="Consecutive same-GT matches required for reliable label.")
    parser.add_argument("--init_window", type=int, default=3, help="Track age window labeled as initialization.")
    parser.add_argument("--fp_threshold", type=int, default=10, help="Consecutive unmatched frames before false_positive.")
    parser.add_argument("--id_switch_recent_window", type=int, default=5, help="Frames after ID switch still labeled id_switch.")
    parser.add_argument("--future_k", type=int, default=5, help="Lookahead rows per track for future_stable target.")
    parser.add_argument("--future_ratio", type=float, default=0.8, help="Required ratio of future rows preserving same GT.")

    parser.add_argument("--match_category", action="store_true", help="Require category-compatible pred-GT matching.")
    parser.add_argument("--no_off_by_one_category", action="store_true", help="Disable 0-index/1-index category tolerance.")
    parser.add_argument("--pred_box_format", choices=["coco", "xyxy"], default="coco", help="Prediction bbox format.")
    parser.add_argument("--gt_box_format", choices=["coco", "xyxy"], default="coco", help="GT bbox format.")

    parser.add_argument("--offset_x", type=float, default=0.0, help="Add X offset to prediction boxes.")
    parser.add_argument("--offset_y", type=float, default=0.0, help="Add Y offset to prediction boxes.")
    parser.add_argument("--crop_x_min", type=float, default=None)
    parser.add_argument("--crop_y_min", type=float, default=None)
    parser.add_argument("--crop_x_max", type=float, default=None)
    parser.add_argument("--crop_y_max", type=float, default=None)

    args = parser.parse_args()
    crop_region = parse_crop_region(args)

    print("=" * 80)
    print("TRACK LABEL GENERATION")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    if crop_region is not None:
        print(f"crop_region: {crop_region}")
    print("=" * 80)

    generate_labels(
        gt_path=args.gt,
        pred_path=args.pred,
        output=args.output,
        sequence_name=args.sequence,
        attr_file=args.attr_file,
        iou_thr=args.iou_thr,
        n_confirm=args.n_confirm,
        init_window=args.init_window,
        fp_threshold=args.fp_threshold,
        id_switch_recent_window=args.id_switch_recent_window,
        future_k=args.future_k,
        future_ratio=args.future_ratio,
        match_category=args.match_category,
        allow_off_by_one_category=not args.no_off_by_one_category,
        pred_box_format=args.pred_box_format,
        gt_box_format=args.gt_box_format,
        coord_offset_x=args.offset_x,
        coord_offset_y=args.offset_y,
        crop_region=crop_region,
    )


if __name__ == "__main__":
    main()
