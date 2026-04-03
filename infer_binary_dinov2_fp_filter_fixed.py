#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Dinov2Model
from tqdm import tqdm


def expand_bbox_xywh(bbox: Sequence[float], img_w: int, img_h: int, expand_ratio: float) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * (1.0 + expand_ratio * 2.0)
    new_h = h * (1.0 + expand_ratio * 2.0)
    x1 = max(0, int(round(cx - new_w / 2.0)))
    y1 = max(0, int(round(cy - new_h / 2.0)))
    x2 = min(img_w, int(round(cx + new_w / 2.0)))
    y2 = min(img_h, int(round(cy + new_h / 2.0)))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def load_coco_records(coco_json: str, image_roots: List[str], label: Optional[int] = None) -> List[Dict]:
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    image_id_to_info = {img["id"]: img for img in coco["images"]}
    roots = [Path(p) for p in image_roots]
    records: List[Dict] = []
    for ann in coco["annotations"]:
        img_info = image_id_to_info.get(ann["image_id"])
        if img_info is None:
            continue
        file_name = img_info["file_name"]
        img_path = None
        for root in roots:
            cand = root / file_name
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        records.append({
            "image_id": ann["image_id"],
            "annotation_id": ann.get("id", -1),
            "file_name": file_name,
            "image_path": str(img_path),
            "bbox": ann["bbox"],
            "label": label,
        })
    return records


class InferenceCocoCropDataset(Dataset):
    def __init__(self, records: List[Dict], processor: AutoImageProcessor, bbox_expand_ratio: float = 0.15):
        self.records = records
        self.processor = processor
        self.bbox_expand_ratio = bbox_expand_ratio

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = expand_bbox_xywh(rec["bbox"], w, h, self.bbox_expand_ratio)
        crop = img.crop((x1, y1, x2, y2)).convert("RGB")
        pixel_values = self.preprocess(crop)
        meta = {
            "image_id": rec["image_id"],
            "annotation_id": rec["annotation_id"],
            "file_name": rec["file_name"],
            "bbox": rec["bbox"],
            "gt_label": rec["label"],
        }
        return pixel_values, meta


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return pixel_values, metas


class BinaryDinov2Classifier(nn.Module):
    def __init__(self, backbone_name: str = "facebook/dinov2-base", hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        feat_dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.head(cls_token)


@torch.no_grad()
def run_inference(model, loader, device, positive_threshold: float):
    model.eval()
    rows = []
    for pixel_values, metas in tqdm(loader, desc="Inference"):
        pixel_values = pixel_values.to(device, non_blocking=True)
        logits = model(pixel_values)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        for i, meta in enumerate(metas):
            neg_score = float(probs[i, 0].item())
            pos_score = float(probs[i, 1].item())
            pred_label = int(pred[i].item())
            rows.append({
                "image_id": meta["image_id"],
                "annotation_id": meta["annotation_id"],
                "file_name": meta["file_name"],
                "bbox_x": meta["bbox"][0],
                "bbox_y": meta["bbox"][1],
                "bbox_w": meta["bbox"][2],
                "bbox_h": meta["bbox"][3],
                "negative_score": neg_score,
                "positive_score": pos_score,
                "pred_label": pred_label,
                "pred_name": "positive" if pred_label == 1 else "negative",
                "keep_as_candidate": bool(pos_score >= positive_threshold),
                **({"gt_label": int(meta["gt_label"])} if meta["gt_label"] is not None else {}),
            })
    return rows


def summarize(rows: List[Dict]):
    labeled = [r for r in rows if "gt_label" in r]
    if not labeled:
        return None
    tp = sum((r["pred_label"] == 1 and r["gt_label"] == 1) for r in labeled)
    tn = sum((r["pred_label"] == 0 and r["gt_label"] == 0) for r in labeled)
    fp = sum((r["pred_label"] == 1 and r["gt_label"] == 0) for r in labeled)
    fn = sum((r["pred_label"] == 0 and r["gt_label"] == 1) for r in labeled)
    total = len(labeled)
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_roots", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--positive_json", type=str, default=None)
    parser.add_argument("--negative_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--positive_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.model_path, map_location="cpu")
    model = BinaryDinov2Classifier(ckpt["backbone_name"], ckpt["hidden_dim"], ckpt["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    processor = AutoImageProcessor.from_pretrained(ckpt["backbone_name"])
    bbox_expand_ratio = ckpt.get("bbox_expand_ratio", 0.15)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    image_roots = [s.strip() for s in args.image_roots.split(",") if s.strip()]

    records: List[Dict] = []
    if args.input_json:
        records.extend(load_coco_records(args.input_json, image_roots, label=None))
    else:
        if args.positive_json:
            records.extend(load_coco_records(args.positive_json, image_roots, label=1))
        if args.negative_json:
            records.extend(load_coco_records(args.negative_json, image_roots, label=0))
    if not records:
        raise RuntimeError("No records loaded. Provide --input_json or labeled --positive_json/--negative_json.")

    ds = InferenceCocoCropDataset(records, processor, bbox_expand_ratio=bbox_expand_ratio)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True, collate_fn=collate_fn)
    rows = run_inference(model, dl, device, args.positive_threshold)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to: {out_csv}")

    stats = summarize(rows)
    if stats:
        print(f"acc={stats['acc']:.4f} precision={stats['precision']:.4f} recall={stats['recall']:.4f} "
              f"f1={stats['f1']:.4f} tp={stats['tp']} tn={stats['tn']} fp={stats['fp']} fn={stats['fn']}")


if __name__ == "__main__":
    main()
