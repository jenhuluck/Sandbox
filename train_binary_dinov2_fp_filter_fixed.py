#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from transformers import AutoImageProcessor, Dinov2Model
from tqdm import tqdm

from typing import List, Dict
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms as T
from transformers import AutoImageProcessor


def safe_expand_bbox_xywh(
    bbox,
    img_w: int,
    img_h: int,
    expand_ratio: float = 0.15,
    min_crop_size: int = 8,
):
    """
    bbox: [x, y, w, h]
    returns integer xyxy box, clamped to image bounds, with minimum size enforced
    """
    x, y, w, h = bbox
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox with non-positive size: {bbox}")

    cx = x + w / 2.0
    cy = y + h / 2.0

    new_w = max(w * (1.0 + 2.0 * expand_ratio), float(min_crop_size))
    new_h = max(h * (1.0 + 2.0 * expand_ratio), float(min_crop_size))

    x1 = int(np.floor(cx - new_w / 2.0))
    y1 = int(np.floor(cy - new_h / 2.0))
    x2 = int(np.ceil(cx + new_w / 2.0))
    y2 = int(np.ceil(cy + new_h / 2.0))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Enforce minimum crop size again after clamping
    cur_w = x2 - x1
    cur_h = y2 - y1

    if cur_w < min_crop_size:
        deficit = min_crop_size - cur_w
        left_add = deficit // 2
        right_add = deficit - left_add
        x1 = max(0, x1 - left_add)
        x2 = min(img_w, x2 + right_add)

    if cur_h < min_crop_size:
        deficit = min_crop_size - cur_h
        top_add = deficit // 2
        bot_add = deficit - top_add
        y1 = max(0, y1 - top_add)
        y2 = min(img_h, y2 + bot_add)

    # Final fallback in case image itself is tiny / edge case
    if x2 <= x1:
        x1 = max(0, min(x1, img_w - 1))
        x2 = min(img_w, x1 + max(1, min_crop_size))
    if y2 <= y1:
        y1 = max(0, min(y1, img_h - 1))
        y2 = min(img_h, y1 + max(1, min_crop_size))

    return int(x1), int(y1), int(x2), int(y2)


class ResizeKeepAspectPad:
    def __init__(self, size=224, fill=(0, 0, 0), interpolation=Image.BICUBIC):
        self.size = int(size)
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid crop size: {img.size}")

        scale = min(self.size / w, self.size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img = img.resize((new_w, new_h), resample=self.interpolation)

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)


class SafeGaussianBlur:
    def __init__(self, kernel_size=3, sigma=(0.1, 1.5), p=0.2, min_side=8):
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p
        self.min_side = min_side

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        if min(w, h) < self.min_side:
            return img
        return self.blur(img)


class BinaryCocoCropDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        processor: AutoImageProcessor,
        train: bool = True,
        bbox_expand_ratio: float = 0.15,
        min_crop_size: int = 8,
        use_vertical_flip: bool = False,
        use_affine: bool = False,
    ):
        self.records = records
        self.processor = processor
        self.train = train
        self.bbox_expand_ratio = bbox_expand_ratio
        self.min_crop_size = min_crop_size

        # Robustly parse processor size
        size = getattr(processor, "size", {"height": 224, "width": 224})
        if isinstance(size, dict):
            if "shortest_edge" in size:
                resize_size = int(size["shortest_edge"])
            elif "height" in size and "width" in size:
                # Use square target for pad-to-size pipeline
                resize_size = int(max(size["height"], size["width"]))
            else:
                resize_size = 224
        else:
            resize_size = 224

        image_mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

        if len(image_mean) != 3:
            image_mean = [0.485, 0.456, 0.406]
        if len(image_std) != 3:
            image_std = [0.229, 0.224, 0.225]

        aug_list = []

        if train:
            if use_affine:
                # Mild affine only; keep optional because it can hurt bbox-crop semantics
                aug_list.append(
                    T.RandomAffine(
                        degrees=8,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        fill=0,
                    )
                )

            aug_list.extend([
                T.RandomHorizontalFlip(p=0.5),
            ])

            if use_vertical_flip:
                aug_list.append(T.RandomVerticalFlip(p=0.5))

            aug_list.extend([
                T.RandomApply([
                    T.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.20,
                        hue=0.05,
                    )
                ], p=0.8),
                SafeGaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.5),
                    p=0.2,
                    min_side=max(8, min_crop_size),
                ),
            ])

        aug_list.extend([
            ResizeKeepAspectPad(size=resize_size, fill=(0, 0, 0)),
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std),
        ])

        self.aug = T.Compose(aug_list)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        img = Image.open(rec["image_path"]).convert("RGB")
        img_w, img_h = img.size

        x1, y1, x2, y2 = safe_expand_bbox_xywh(
            rec["bbox"],
            img_w=img_w,
            img_h=img_h,
            expand_ratio=self.bbox_expand_ratio,
            min_crop_size=self.min_crop_size,
        )

        crop = img.crop((x1, y1, x2, y2)).convert("RGB")
        pixel_values = self.aug(crop)

        meta = {
            "image_id": rec["image_id"],
            "annotation_id": rec["annotation_id"],
            "file_name": rec["file_name"],
            "bbox": rec["bbox"],
            "crop_xyxy": [x1, y1, x2, y2],
        }

        return pixel_values, torch.tensor(rec["label"], dtype=torch.long), meta
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expand_bbox_xywh(
    bbox: Sequence[float],
    img_w: int,
    img_h: int,
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
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


def load_coco_records(coco_json: str, image_roots: List[str], label: int) -> List[Dict]:
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

        records.append(
            {
                "image_id": ann["image_id"],
                "annotation_id": ann.get("id", -1),
                "file_name": file_name,
                "image_path": str(img_path),
                "bbox": ann["bbox"],
                "label": int(label),
            }
        )
    return records


class BinaryCocoCropDataset(Dataset):
    def __init__(self, records: List[Dict], processor: AutoImageProcessor, train: bool = True, bbox_expand_ratio: float = 0.15):
        self.records = records
        self.processor = processor
        self.train = train
        self.bbox_expand_ratio = bbox_expand_ratio

        size = getattr(processor, "size", {"height": 224, "width": 224})
        if isinstance(size, dict):
            if "shortest_edge" in size:
                resize_hw = (int(size["shortest_edge"]), int(size["shortest_edge"]))
            else:
                resize_hw = (int(size.get("height", 224)), int(size.get("width", 224)))
        else:
            resize_hw = (224, 224)

        image_mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

        aug_list = []
        if train:
            aug_list.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05)], p=0.8),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
                T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            ])

        aug_list.extend([
            T.Resize(resize_hw, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std),
        ])
        self.aug = T.Compose(aug_list)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = expand_bbox_xywh(rec["bbox"], w, h, self.bbox_expand_ratio)
        crop = img.crop((x1, y1, x2, y2)).convert("RGB")
        pixel_values = self.aug(crop)
        meta = {
            "image_id": rec["image_id"],
            "annotation_id": rec["annotation_id"],
            "file_name": rec["file_name"],
            "bbox": rec["bbox"],
        }
        return pixel_values, torch.tensor(rec["label"], dtype=torch.long), meta


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return pixel_values, labels, metas


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

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.head(cls_token)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = tn = fp = fn = 0
    criterion = nn.CrossEntropyLoss()

    for pixel_values, labels, _ in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(pixel_values)
        loss = criterion(logits, labels)

        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

        total_loss += float(loss.item()) * labels.shape[0]
        total += labels.shape[0]
        correct += int((preds == labels).sum().item())

        tp += int(((preds == 1) & (labels == 1)).sum().item())
        tn += int(((preds == 0) & (labels == 0)).sum().item())
        fp += int(((preds == 1) & (labels == 0)).sum().item())
        fn += int(((preds == 0) & (labels == 1)).sum().item())

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": total_loss / max(total, 1),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def build_sampler(records: List[Dict]) -> WeightedRandomSampler:
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)


def split_records(records: List[Dict], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(records))
    rng.shuffle(idxs)
    n_val = int(round(len(records) * val_ratio))
    val_idxs = set(idxs[:n_val].tolist())
    train_records, val_records = [], []
    for i, rec in enumerate(records):
        (val_records if i in val_idxs else train_records).append(rec)
    return train_records, val_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_json", type=str, required=True)
    parser.add_argument("--negative_json", type=str, required=True)
    parser.add_argument("--image_roots", type=str, required=True, help="Comma-separated image root directories")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--backbone_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--unfreeze_backbone", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    image_roots = [s.strip() for s in args.image_roots.split(",") if s.strip()]

    pos_records = load_coco_records(args.positive_json, image_roots, label=1)
    neg_records = load_coco_records(args.negative_json, image_roots, label=0)
    all_records = pos_records + neg_records
    if not all_records:
        raise RuntimeError("No valid records found. Check COCO JSONs and image_roots.")

    train_records, val_records = split_records(all_records, args.val_ratio, args.seed)

    processor = AutoImageProcessor.from_pretrained(args.backbone_name)
    train_ds = BinaryCocoCropDataset(train_records, processor, train=True, bbox_expand_ratio=args.bbox_expand_ratio)
    val_ds = BinaryCocoCropDataset(val_records, processor, train=False, bbox_expand_ratio=args.bbox_expand_ratio)

    sampler = build_sampler(train_records) if args.use_weighted_sampler else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model = BinaryDinov2Classifier(args.backbone_name, args.hidden_dim, args.dropout).to(device)
    if args.unfreeze_backbone:
        model.unfreeze_backbone()
    else:
        model.freeze_backbone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_path = out_dir / "best_model.pt"
    log_path = out_dir / "train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for pixel_values, labels, _ in pbar:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.shape[0]
            total += labels.shape[0]
            pbar.set_postfix(train_loss=running_loss / max(total, 1))

        train_loss = running_loss / max(total, 1)
        val_stats = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_stats.items()}}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[{epoch:03d}/{args.epochs:03d}] train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
              f"val_acc={val_stats['acc']:.4f} val_f1={val_stats['f1']:.4f} "
              f"val_precision={val_stats['precision']:.4f} val_recall={val_stats['recall']:.4f}")

        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone_name": args.backbone_name,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "bbox_expand_ratio": args.bbox_expand_ratio,
                "best_f1": best_f1,
                "label_map": {0: "negative", 1: "positive"},
            }, best_path)

    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
