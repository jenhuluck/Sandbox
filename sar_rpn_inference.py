#!/usr/bin/env python3
import os
import sys
import cv2
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------
# Optional: add MSFA repo to path so custom modules can register
# ---------------------------------------------------------------------
MSFA_PATH = "/home/jhu/SATLOCK/MSFA/MSFA"
if MSFA_PATH not in sys.path:
    sys.path.insert(0, MSFA_PATH)

try:
    import msfa  # noqa: F401
    from msfa.models.backbones.MSFA import MSFA
    from mmdet.registry import MODELS

    # Alias used in some custom configs
    if MODELS.get("Self_features_model") is None:
        MODELS.register_module(name="Self_features_model", module=MSFA)
except Exception as e:
    print(f"Warning: MSFA registration issue: {e}")

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.dataset import Compose
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def build_model(cfg_path: str, ckpt_path: str, device: str = "cuda:0"):
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(cfg_path)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model, cfg


def build_test_pipeline(cfg):
    """
    Use cfg.test_pipeline if present; otherwise fall back to
    cfg.test_dataloader.dataset.pipeline.
    """
    if hasattr(cfg, "test_pipeline") and cfg.test_pipeline is not None:
        pipeline_cfg = cfg.test_pipeline
    elif (
        hasattr(cfg, "test_dataloader")
        and "dataset" in cfg.test_dataloader
        and "pipeline" in cfg.test_dataloader.dataset
    ):
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
    else:
        raise ValueError("No test pipeline found in config.")

    # Make sure we do not require GT annotations for plain inference.
    cleaned = [t for t in pipeline_cfg if t.get("type") != "LoadAnnotations"]
    return Compose(cleaned)


@torch.no_grad()
def extract_rpn_proposals(model, cfg, image_path: str, device: str = "cuda:0"):
    pipeline = build_test_pipeline(cfg)

    data = dict(
        img_path=image_path,
        img_id=0,
    )
    data = pipeline(data)

    # Pipeline output format expected by the data_preprocessor
    batch = dict(
        inputs=[data["inputs"].to(device)],
        data_samples=[data["data_samples"]],
    )
    batch = model.data_preprocessor(batch, training=False)

    feats = model.extract_feat(batch["inputs"])
    rpn_results_list = model.rpn_head.predict(
        feats,
        batch["data_samples"],
        rescale=True,
    )

    result = rpn_results_list[0]
    boxes = result.bboxes.detach().cpu().numpy()
    scores = result.scores.detach().cpu().numpy()

    return boxes, scores


def filter_and_truncate(boxes, scores, score_thresh=0.0, topk=4000):
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    order = np.argsort(-scores)
    if topk is not None:
        order = order[:topk]

    return boxes[order], scores[order]


def save_result(save_path: Path, boxes, scores, save_format="pickle"):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "pickle":
        with open(save_path, "wb") as f:
            pickle.dump(
                {"boxes": boxes.astype(np.float32), "scores": scores.astype(np.float32)},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    elif save_format == "npz":
        np.savez_compressed(
            save_path,
            boxes=boxes.astype(np.float32),
            scores=scores.astype(np.float32),
        )
    elif save_format == "json":
        payload = {
            "boxes": boxes.astype(float).tolist(),
            "scores": scores.astype(float).tolist(),
        }
        with open(save_path, "w") as f:
            json.dump(payload, f)
    else:
        raise ValueError(f"Unsupported save_format: {save_format}")


def iter_images(image_dir):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in sorted(Path(image_dir).rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--image_dir", required=True, help="Directory of input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save proposals")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--save-format", default=None, choices=["pickle", "npz", "json"])
    parser.add_argument("--debug-vis", action="store_true")
    parser.add_argument("--debug-vis-topk", type=int, default=200)
    args = parser.parse_args()

    model, cfg = build_model(args.config, args.checkpoint, args.device)

    # Read defaults from cfg.rpn_cache_cfg if present
    rpn_cache_cfg = getattr(cfg, "rpn_cache_cfg", {})
    topk = args.topk if args.topk is not None else rpn_cache_cfg.get("max_proposals_per_image", 4000)
    score_thresh = args.score_thresh if args.score_thresh is not None else rpn_cache_cfg.get("score_threshold", 0.0)
    save_format = args.save_format if args.save_format is not None else rpn_cache_cfg.get("save_format", "pickle")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_images(args.image_dir))
    print(f"Found {len(image_paths)} images")

    for idx, image_path in enumerate(image_paths, 1):
        boxes, scores = extract_rpn_proposals(model, cfg, str(image_path), args.device)
        raw_n = len(boxes)

        boxes, scores = filter_and_truncate(
            boxes, scores, score_thresh=score_thresh, topk=topk
        )
        kept_n = len(boxes)

        stem = image_path.stem
        suffix = {"pickle": ".pkl", "npz": ".npz", "json": ".json"}[save_format]
        save_path = output_dir / f"{stem}{suffix}"
        save_result(save_path, boxes, scores, save_format=save_format)

        print(f"[{idx}/{len(image_paths)}] {image_path.name}: raw={raw_n}, kept={kept_n}")

        if args.debug_vis:
            img = cv2.imread(str(image_path))
            if img is not None:
                for box in boxes[:args.debug_vis_topk]:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                vis_path = output_dir / f"{stem}_debug.jpg"
                cv2.imwrite(str(vis_path), img)


if __name__ == "__main__":
    main()