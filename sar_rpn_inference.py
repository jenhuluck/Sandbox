#!/usr/bin/env python3
"""
SAR RPN Inference Script
Generate and cache RPN proposals from SAR images using MMDetection model.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import cv2

# Add MSFA to path to register custom modules
MSFA_PATH = "/home/jhu/SATLOCK/MSFA/MSFA"
if MSFA_PATH not in sys.path:
    sys.path.insert(0, MSFA_PATH)

# Import MSFA modules to register custom backbones and datasets
try:
    import msfa
    from msfa.models.backbones.MSFA import MSFA
    from mmdet.registry import MODELS

    # Register MSFA as Self_features_model (alias for backwards compatibility)
    if not MODELS.get('Self_features_model'):
        MODELS.register_module(name='Self_features_model', module=MSFA)

    print(f"Successfully imported MSFA modules from {MSFA_PATH}")
except ImportError as e:
    print(f"Warning: Could not import MSFA modules from {MSFA_PATH}: {e}")
    print("Custom models like MSFA backbone may not be available.")

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector


def load_sar_model(config_file, checkpoint_file, device='cuda'):
    """
    Load SAR detection model from MMDetection.

    Args:
        config_file: Path to MMDetection config file
        checkpoint_file: Path to model checkpoint
        device: Device to run inference on

    Returns:
        model: Loaded MMDetection model
    """
    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()
    return model


def extract_rpn_proposals(model, image_path, device='cuda', target_size=(800, 800)):
    """
    Extract RPN proposals from a single image.

    Args:
        model: MMDetection model
        image_path: Path to input image
        device: Device for inference
        target_size: Target size (H, W) to resize image to

    Returns:
        boxes: numpy array of shape (N, 4) with boxes in xyxy format (in original image coordinates)
        scores: numpy array of shape (N,) with objectness scores
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ori_h, ori_w = img.shape[:2]

    # Resize image to target size
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))

    # Calculate scale factor
    scale_h = target_size[0] / ori_h
    scale_w = target_size[1] / ori_w

    # Run inference to get RPN proposals
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    # Create data sample in the format MMDet expects
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_id': 0,
        'img_path': str(image_path),
        'ori_shape': (ori_h, ori_w),
        'img_shape': target_size,
        'scale_factor': (scale_w, scale_h)
    })

    # Preprocess image - convert to tensor and normalize
    # The model expects a batch, so we add batch dimension
    data = {
        'inputs': [torch.from_numpy(img_resized).permute(2, 0, 1).float().to(device)],
        'data_samples': [data_sample]
    }

    # Use data preprocessor
    data = model.data_preprocessor(data, False)

    # Extract features
    with torch.no_grad():
        img_tensor = data['inputs']
        batch_data_samples = data['data_samples']

        # Get features from backbone and neck
        x = model.extract_feat(img_tensor)

        # Get RPN proposals
        rpn_results_list = model.rpn_head.predict(
            x, batch_data_samples, rescale=True  # Rescale to original image size
        )

    # Extract boxes and scores from the first (and only) image
    rpn_results = rpn_results_list[0]

    boxes = rpn_results.bboxes.cpu().numpy()  # (N, 4) in xyxy format (original size)
    scores = rpn_results.scores.cpu().numpy()  # (N,)

    return boxes, scores


def run_sar_rpn_inference(
    image_dir,
    output_cache_dir,
    config_file,
    checkpoint_file,
    device='cuda',
    topk=1000,
    score_thresh=0.0,
    save_format='pickle'
):
    """
    Run RPN inference on all images in a directory and cache results.

    Args:
        image_dir: Directory containing input images
        output_cache_dir: Directory to save cached proposals
        config_file: Path to MMDetection config
        checkpoint_file: Path to model checkpoint
        device: Device for inference
        topk: Keep only top-K proposals per image
        score_thresh: Minimum objectness score threshold
        save_format: Format to save cache ('pickle', 'torch', or 'npz')
    """
    os.makedirs(output_cache_dir, exist_ok=True)

    # Load model
    print(f"Loading SAR model from {checkpoint_file}...")
    model = load_sar_model(config_file, checkpoint_file, device)

    # Get all image files
    image_dir = Path(image_dir)
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    ])

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Extract RPN proposals
            boxes, scores = extract_rpn_proposals(model, img_path, device)

            # Sort by objectness score
            idx = np.argsort(scores)[::-1]
            boxes = boxes[idx]
            scores = scores[idx]

            # Apply score threshold
            if score_thresh > 0:
                keep = scores >= score_thresh
                boxes = boxes[keep]
                scores = scores[keep]

            # Keep top-K
            if topk is not None and len(boxes) > topk:
                boxes = boxes[:topk]
                scores = scores[:topk]

            # Prepare data to cache
            cache_data = {
                'boxes': boxes,
                'scores': scores,
                'image_name': img_path.stem,
                'image_shape': cv2.imread(str(img_path)).shape[:2]
            }

            # Save to cache
            cache_name = img_path.stem
            if save_format == 'pickle':
                cache_path = Path(output_cache_dir) / f"{cache_name}.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            elif save_format == 'torch':
                cache_path = Path(output_cache_dir) / f"{cache_name}.pt"
                torch.save(cache_data, cache_path)
            elif save_format == 'npz':
                cache_path = Path(output_cache_dir) / f"{cache_name}.npz"
                np.savez(cache_path, **cache_data)
            else:
                raise ValueError(f"Unknown save format: {save_format}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f"Done! Cached proposals saved to: {output_cache_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate and cache RPN proposals from SAR images'
    )
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing input SAR images')
    parser.add_argument('--output_cache_dir', type=str, required=True,
                       help='Directory to save cached RPN proposals')
    parser.add_argument('--config_file', type=str, required=True,
                       help='Path to MMDetection config file')
    parser.add_argument('--checkpoint_file', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (default: cuda)')
    parser.add_argument('--topk', type=int, default=1000,
                       help='Keep top-K proposals per image (default: 1000)')
    parser.add_argument('--score_thresh', type=float, default=0.0,
                       help='Minimum objectness score threshold (default: 0.0)')
    parser.add_argument('--save_format', type=str, default='pickle',
                       choices=['pickle', 'torch', 'npz'],
                       help='Format to save cached proposals (default: pickle)')

    args = parser.parse_args()

    run_sar_rpn_inference(
        image_dir=args.image_dir,
        output_cache_dir=args.output_cache_dir,
        config_file=args.config_file,
        checkpoint_file=args.checkpoint_file,
        device=args.device,
        topk=args.topk,
        score_thresh=args.score_thresh,
        save_format=args.save_format
    )


if __name__ == '__main__':
    main()
