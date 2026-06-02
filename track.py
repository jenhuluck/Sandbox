#!/usr/bin/env python3
"""
Generate frame-level uncertainty labels for tracking by comparing predictions to GT.

Labels:
- reliable: Track matched to same GT ID for N_confirm frames, no recent ID switch
- initialization: Track age <= init_window
- id_switch: Track switched from one GT ID to another
- uncertain: Unmatched but short history or recent match
- false_positive: Unmatched for FP_threshold consecutive frames

Binary labels:
- 1: reliable
- 2: initialization
- 0: id_switch, uncertain, false_positive
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import argparse
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================
N_CONFIRM = 5                    # Frames to confirm as reliable
INIT_WINDOW = 3                  # Initialization window (frames)
FP_UNMATCHED_THRESHOLD = 10      # Consecutive unmatched frames → FP
ID_SWITCH_RECENT_WINDOW = 5      # Look back for recent ID switches
IOU_THRESHOLD = 0.5              # Minimum IoU for valid match


# ============================================================================
# Utility Functions
# ============================================================================
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    Boxes in format: [x, y, w, h] (COCO format) or [x1, y1, x2, y2]
    """
    # Convert to [x1, y1, x2, y2]
    if len(box1) == 4 and len(box2) == 4:
        # Assume COCO format [x, y, w, h]
        box1_xyxy = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2_xyxy = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    else:
        box1_xyxy = box1
        box2_xyxy = box2

    # Intersection
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union
    area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def hungarian_matching(pred_boxes, gt_boxes, iou_threshold=0.5, match_category=True):
    """
    Perform Hungarian matching between predictions and GT based on IoU.

    Args:
        pred_boxes: List of (pred_idx, bbox, class_id)
        gt_boxes: List of (gt_id, bbox, class_id)
        iou_threshold: Minimum IoU for valid match
        match_category: If False, ignore category mismatch (match by IoU only)

    Returns:
        matches: List of (pred_idx, gt_id, iou)
        unmatched_preds: List of pred_idx
        unmatched_gts: List of gt_id
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), [g[0] for g in gt_boxes]

    # Build cost matrix (negative IoU, since we minimize cost)
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    cost_matrix = np.zeros((n_pred, n_gt))

    for i, (pred_idx, pred_bbox, pred_class) in enumerate(pred_boxes):
        for j, (gt_id, gt_bbox, gt_class) in enumerate(gt_boxes):
            # Check category match (with tolerance for 0-indexed vs 1-indexed)
            if match_category:
                # Allow match if classes are equal OR off-by-one (0 vs 1 indexing)
                if pred_class != gt_class and pred_class != gt_class - 1 and pred_class + 1 != gt_class:
                    cost_matrix[i, j] = 1e6  # Large cost (no match)
                    continue

            iou = compute_iou(pred_bbox, gt_bbox)
            cost_matrix[i, j] = 1.0 - iou  # Minimize cost = maximize IoU

    # Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by IoU threshold
    matches = []
    matched_pred = set()
    matched_gt = set()

    for pred_i, gt_j in zip(pred_indices, gt_indices):
        iou = 1.0 - cost_matrix[pred_i, gt_j]
        if iou >= iou_threshold:
            pred_idx = pred_boxes[pred_i][0]
            gt_id = gt_boxes[gt_j][0]
            matches.append((pred_idx, gt_id, iou))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_id)

    # Unmatched predictions and GTs
    unmatched_preds = [pred_boxes[i][0] for i in range(n_pred) if pred_boxes[i][0] not in matched_pred]
    unmatched_gts = [gt_boxes[j][0] for j in range(n_gt) if gt_boxes[j][0] not in matched_gt]

    return matches, unmatched_preds, unmatched_gts


# ============================================================================
# Track History Manager
# ============================================================================
class TrackHistory:
    def __init__(self):
        self.tracks = defaultdict(lambda: {
            'matched_gt_id': None,
            'previous_gt_id': None,
            'matched_same_gt_count': 0,
            'unmatched_count': 0,
            'track_age': 0,
            'id_switch_frame': None,  # Frame where last ID switch occurred
            'history': []  # List of (frame, matched_gt_id, iou)
        })

    def update(self, frame, track_id, matched_gt_id, iou):
        """Update track history with new matching result."""
        track = self.tracks[track_id]
        track['track_age'] += 1
        track['history'].append((frame, matched_gt_id, iou))

        # Update matching status
        if matched_gt_id is not None:
            # Matched
            track['unmatched_count'] = 0

            # Check for ID switch
            if track['matched_gt_id'] is not None and track['matched_gt_id'] != matched_gt_id:
                # ID switch detected!
                track['previous_gt_id'] = track['matched_gt_id']
                track['id_switch_frame'] = frame
                track['matched_same_gt_count'] = 1  # Reset count
            elif track['matched_gt_id'] == matched_gt_id:
                # Continue matching to same GT
                track['matched_same_gt_count'] += 1
            else:
                # First match
                track['matched_same_gt_count'] = 1

            track['matched_gt_id'] = matched_gt_id
        else:
            # Unmatched
            track['unmatched_count'] += 1
            # Don't reset matched_gt_id yet (might recover)

    def get_label(self, frame, track_id, n_confirm=N_CONFIRM, init_window=INIT_WINDOW, fp_threshold=FP_UNMATCHED_THRESHOLD):
        """
        Assign label based on track history.

        Returns:
            label: One of ['reliable', 'initialization', 'id_switch', 'uncertain', 'false_positive']
            binary_label: 1 (reliable), 2 (initialization), 0 (others)
        """
        track = self.tracks[track_id]

        # 1. Initialization
        if track['track_age'] <= init_window:
            return 'initialization', 2

        # 2. ID Switch (check if recent)
        if track['id_switch_frame'] is not None:
            frames_since_switch = frame - track['id_switch_frame']
            if frames_since_switch <= ID_SWITCH_RECENT_WINDOW:
                return 'id_switch', 0

        # 3. Currently matched
        if track['unmatched_count'] == 0:
            # NEW: Use match ratio for more flexible reliability assessment
            # If track_age > 3 and has good match ratio, consider reliable
            match_ratio = track['matched_same_gt_count'] / track['track_age']

            # Reliable if:
            # - Option A: Has enough consecutive matches (original logic)
            # - Option B: Track age > 3 AND match ratio > 70%
            if (track['matched_same_gt_count'] >= n_confirm or
                (track['track_age'] > 3 and match_ratio >= 0.7)):
                # Also check no recent ID switch
                if (track['id_switch_frame'] is None or
                    frame - track['id_switch_frame'] > ID_SWITCH_RECENT_WINDOW):
                    return 'reliable', 1

            # Matched but not reliable yet
            return 'uncertain', 0

        # 4. Currently unmatched
        else:
            # Calculate how long it was matched before losing match
            # If it was previously reliable and just lost match temporarily, still uncertain
            # If it's been unmatched for a long time, it's a false positive
            if (track['unmatched_count'] >= fp_threshold and
                track['track_age'] > init_window):
                return 'false_positive', 0
            else:
                # Short-term unmatched (temporary occlusion)
                return 'uncertain', 0


# ============================================================================
# Main Processing
# ============================================================================
def load_gt_annotations(gt_path, crop_region=None):
    """
    Load GT annotations from COCO format.

    Args:
        gt_path: Path to GT JSON file
        crop_region: Optional (x_min, y_min, x_max, y_max) to filter GT to crop region

    Returns:
        gt_by_frame: Dict[frame_id -> List[(gt_track_id, bbox, class_id)]]
        image_id_to_frame: Dict[image_id -> frame_id]
    """
    print(f"Loading GT annotations from: {gt_path}")
    if crop_region is not None:
        print(f"  Filtering GT to crop region: x=[{crop_region[0]}, {crop_region[2]}], y=[{crop_region[1]}, {crop_region[3]}]")

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    # Build image_id to frame mapping
    image_id_to_frame = {}
    for img in gt_data['images']:
        # Use mot_frame_id as frame number
        image_id_to_frame[img['id']] = img['mot_frame_id']

    # Group annotations by frame
    gt_by_frame = defaultdict(list)
    filtered_count = 0
    total_count = 0

    for ann in gt_data['annotations']:
        total_count += 1
        frame_id = image_id_to_frame[ann['image_id']]
        gt_track_id = ann['mot_instance_id']
        bbox = ann['bbox']  # [x, y, w, h]
        class_id = ann['category_id']

        # Filter by crop region if specified
        if crop_region is not None:
            x_min, y_min, x_max, y_max = crop_region
            bbox_x_center = bbox[0] + bbox[2] / 2
            bbox_y_center = bbox[1] + bbox[3] / 2

            # Check if bbox center is within crop region
            if not (x_min <= bbox_x_center <= x_max and y_min <= bbox_y_center <= y_max):
                filtered_count += 1
                continue

        gt_by_frame[frame_id].append((gt_track_id, bbox, class_id))

    print(f"  Loaded {total_count} annotations across {len(gt_by_frame)} frames")
    if crop_region is not None:
        print(f"  Filtered out {filtered_count} annotations outside crop region ({filtered_count/total_count*100:.1f}%)")

    return gt_by_frame, image_id_to_frame


def load_predictions(pred_path, coord_offset_x=0, coord_offset_y=0):
    """
    Load predictions from meta.json.

    Args:
        pred_path: Path to meta.json
        coord_offset_x: X-coordinate offset to add (for resolution mismatch)
        coord_offset_y: Y-coordinate offset to add

    Returns:
        pred_by_frame: Dict[frame_id -> List[(track_id, bbox, class_id)]]
    """
    print(f"Loading predictions from: {pred_path}")
    if coord_offset_x != 0 or coord_offset_y != 0:
        print(f"  Applying coordinate offset: x+={coord_offset_x}, y+={coord_offset_y}")

    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    pred_by_frame = {}
    next_temp_id = 1000000  # Temporary IDs for tracks without track_id

    for frame_str, tracks in pred_data.items():
        frame_id = int(frame_str)
        pred_by_frame[frame_id] = []

        for track in tracks:
            track_id = track.get('track_id')

            # Assign temporary ID for tracks without track_id (initialization phase)
            if track_id is None:
                track_id = f"temp_{next_temp_id}"
                next_temp_id += 1

            tlwh = track['tlwh']  # [x, y, w, h]

            # Apply coordinate offset if predictions are in different resolution
            if coord_offset_x != 0 or coord_offset_y != 0:
                tlwh = [
                    tlwh[0] + coord_offset_x,
                    tlwh[1] + coord_offset_y,
                    tlwh[2],
                    tlwh[3]
                ]

            class_id = track['class_id']

            pred_by_frame[frame_id].append((track_id, tlwh, class_id))

    print(f"  Loaded predictions for {len(pred_by_frame)} frames")
    return pred_by_frame


def generate_labels(gt_path, pred_path, output_path, sequence_name, n_confirm=None, init_window=None, fp_threshold=None, coord_offset_x=0, coord_offset_y=0, crop_region=None, match_category=False):
    """Generate frame-level labels by comparing predictions to GT."""

    # Use provided parameters or defaults
    _n_confirm = n_confirm if n_confirm is not None else N_CONFIRM
    _init_window = init_window if init_window is not None else INIT_WINDOW
    _fp_threshold = fp_threshold if fp_threshold is not None else FP_UNMATCHED_THRESHOLD

    # Load data
    gt_by_frame, _ = load_gt_annotations(gt_path, crop_region)
    pred_by_frame = load_predictions(pred_path, coord_offset_x, coord_offset_y)

    # Get all frames (union of GT and predictions)
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    print(f"\nProcessing {len(all_frames)} frames ({min(all_frames)} to {max(all_frames)})")

    # Initialize track history
    history = TrackHistory()

    # Results
    results = []

    # Process each frame
    for frame_id in all_frames:
        gt_boxes = gt_by_frame.get(frame_id, [])
        pred_boxes = pred_by_frame.get(frame_id, [])

        if len(pred_boxes) == 0:
            continue  # No predictions this frame

        # Prepare for Hungarian matching
        # pred_boxes format: (track_id, bbox, class_id)
        # We need (index, bbox, class_id) for matching
        pred_for_matching = [(i, bbox, class_id) for i, (track_id, bbox, class_id) in enumerate(pred_boxes)]
        gt_for_matching = [(gt_id, bbox, class_id) for gt_id, bbox, class_id in gt_boxes]

        # Perform matching
        matches, unmatched_preds, unmatched_gts = hungarian_matching(
            pred_for_matching, gt_for_matching, iou_threshold=IOU_THRESHOLD, match_category=match_category
        )

        # Build pred_idx -> (matched_gt_id, iou) mapping
        match_dict = {pred_idx: (gt_id, iou) for pred_idx, gt_id, iou in matches}

        # Update history and assign labels
        for pred_idx, (track_id, bbox, class_id) in enumerate(pred_boxes):
            if pred_idx in match_dict:
                matched_gt_id, iou = match_dict[pred_idx]
            else:
                matched_gt_id, iou = None, 0.0

            # Update track history
            history.update(frame_id, track_id, matched_gt_id, iou)

            # Get label
            label, binary_label = history.get_label(frame_id, track_id, _n_confirm, _init_window, _fp_threshold)

            # Get track info
            track_info = history.tracks[track_id]

            # Get GT bbox if matched
            gt_bbox_str = ''
            if matched_gt_id is not None:
                # Find the GT bbox
                for gt_id, gt_bbox, gt_class in gt_boxes:
                    if gt_id == matched_gt_id:
                        gt_bbox_str = f"[{gt_bbox[0]:.1f},{gt_bbox[1]:.1f},{gt_bbox[2]:.1f},{gt_bbox[3]:.1f}]"
                        break

            # Get prediction bbox
            pred_bbox_str = f"[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]"

            # Save result
            results.append({
                'sequence': sequence_name,
                'frame': frame_id,
                'track_id': track_id,
                'matched_gt_id': matched_gt_id if matched_gt_id is not None else -1,
                'iou': round(iou, 4),
                'pred_bbox': pred_bbox_str,
                'gt_bbox': gt_bbox_str,
                'track_age': track_info['track_age'],
                'matched_same_gt_count': track_info['matched_same_gt_count'],
                'unmatched_count': track_info['unmatched_count'],
                'label': label,
                'binary_label': binary_label,
                'detail': ''
            })

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved labels to: {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nBinary label distribution:")
    print(df['binary_label'].value_counts())

    return df


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate track uncertainty labels')
    parser.add_argument('--gt', type=str, required=True, help='Path to GT annotations (COCO format JSON)')
    parser.add_argument('--pred', type=str, required=True, help='Path to predictions (meta.json)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence name (e.g., 1-03)')
    parser.add_argument('--n_confirm', type=int, default=N_CONFIRM, help=f'Frames to confirm (default: {N_CONFIRM})')
    parser.add_argument('--init_window', type=int, default=INIT_WINDOW, help=f'Init window (default: {INIT_WINDOW})')
    parser.add_argument('--fp_threshold', type=int, default=FP_UNMATCHED_THRESHOLD,
                       help=f'FP unmatched threshold (default: {FP_UNMATCHED_THRESHOLD})')
    parser.add_argument('--offset_x', type=float, default=0,
                       help='X-coordinate offset for resolution mismatch (e.g., 64 for 512x512 -> 640x512)')
    parser.add_argument('--offset_y', type=float, default=0,
                       help='Y-coordinate offset for resolution mismatch')
    parser.add_argument('--crop_x_min', type=float, default=None,
                       help='Crop region x_min (filter GT outside this region)')
    parser.add_argument('--crop_x_max', type=float, default=None,
                       help='Crop region x_max')
    parser.add_argument('--crop_y_min', type=float, default=None,
                       help='Crop region y_min')
    parser.add_argument('--crop_y_max', type=float, default=None,
                       help='Crop region y_max')
    parser.add_argument('--match_category', action='store_true',
                       help='Enable category matching (default: False, match by IoU only)')

    args = parser.parse_args()

    # Build crop region if specified
    crop_region = None
    if args.crop_x_min is not None and args.crop_x_max is not None:
        y_min = args.crop_y_min if args.crop_y_min is not None else 0
        y_max = args.crop_y_max if args.crop_y_max is not None else 9999
        crop_region = (args.crop_x_min, y_min, args.crop_x_max, y_max)

    print("=" * 80)
    print("TRACK LABEL GENERATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  N_CONFIRM (reliable threshold): {args.n_confirm}")
    print(f"  INIT_WINDOW: {args.init_window}")
    print(f"  FP_UNMATCHED_THRESHOLD: {args.fp_threshold}")
    print(f"  ID_SWITCH_RECENT_WINDOW: {ID_SWITCH_RECENT_WINDOW}")
    print(f"  IOU_THRESHOLD: {IOU_THRESHOLD}")
    print(f"  Match by category: {args.match_category}")
    print(f"  Coordinate offset: x+={args.offset_x}, y+={args.offset_y}")
    if crop_region is not None:
        print(f"  Crop region: x=[{crop_region[0]}, {crop_region[2]}], y=[{crop_region[1]}, {crop_region[3]}]")
    print("=" * 80)

    # Generate labels
    df = generate_labels(args.gt, args.pred, args.output, args.sequence,
                        args.n_confirm, args.init_window, args.fp_threshold,
                        args.offset_x, args.offset_y, crop_region, args.match_category)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
