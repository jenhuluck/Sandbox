#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#a simple mlp network to map the features from source domain to target domain

CLASS_NAMES = [
    "airplane",
    "helicopter",
    "ship",
    "wind mill",
    "storage tank",
    "small vehicle",
    "large vehicle",
    "swimming pool",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}


# -------------------------
# Utils
# -------------------------
def save_projection_artifacts(
    out_dir: Path,
    model: "ProtoMapperModel",
    proto_a: torch.Tensor,
    proto_b: torch.Tensor,
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    class_names: list[str],
    prefix: str = "viz",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        mapped_proto_a = model.map_prototypes(proto_a)

    np.save(out_dir / f"{prefix}_proto_a_before.npy", proto_a.detach().cpu().numpy())
    np.save(out_dir / f"{prefix}_proto_a_after.npy", mapped_proto_a.detach().cpu().numpy())
    np.save(out_dir / f"{prefix}_proto_b.npy", proto_b.detach().cpu().numpy())   # optional
    np.save(out_dir / f"{prefix}_b_instances.npy", b_feats.detach().cpu().numpy())
    np.save(out_dir / f"{prefix}_b_labels.npy", b_labels.detach().cpu().numpy())
    np.save(out_dir / f"{prefix}_class_names.npy", np.asarray(class_names, dtype=object))

    print(f"Saved projection artifacts to: {out_dir}")
    
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_logits(x: torch.Tensor, proto: torch.Tensor, temperature: float) -> torch.Tensor:
    x = l2_normalize(x)
    proto = l2_normalize(proto)
    return (x @ proto.T) / temperature


def parse_class_from_filename(filename: str, class_names: List[str]) -> Optional[str]:
    for class_name in class_names:
        if class_name in filename:
            return class_name
    print(f"Can't find class name in {filename}")
    return None

    # lower_name = filename.lower()
    # matches = [c for c in class_names if c in lower_name]
    # if not matches:
    #     return None
    # matches = sorted(matches, key=len, reverse=True)
    # return matches[0]


def load_feature_folder(
    folder: Path,
    class_names: List[str],
    max_features: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feats = []
    labels = []
    skipped = []

    files = list(folder.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")

    # Group files by class
    if max_features is not None and max_features > 0:
        class_files = defaultdict(list)
        for f in files:
            cls_name = parse_class_from_filename(f.name, class_names)
            if cls_name is not None:
                class_files[cls_name].append(f)
            else:
                skipped.append(f.name)

        # Shuffle and limit per class
        files = []
        
        for cls_name, cls_file_list in class_files.items():
            print(folder, cls_name,len(cls_file_list))
            np.random.shuffle(cls_file_list)
            files.extend(cls_file_list[:max_features])
    else:
        files = sorted(files)

    for f in files:
        cls_name = parse_class_from_filename(f.name, class_names)
        if cls_name is None:
            skipped.append(f.name)
            continue

        arr = np.load(f)
        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 1:
            feat = arr
        elif arr.ndim == 2 and arr.shape[0] == 1:
            feat = arr[0]
        else:
            feat = arr.reshape(-1)

        feats.append(feat)
        labels.append(CLASS_TO_ID[cls_name])

    if not feats:
        raise RuntimeError(f"No valid features parsed from {folder}")

    feats = np.stack(feats, axis=0).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return feats, labels, skipped


def compute_class_prototypes(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    One prototype per class: mean feature.
    Returns [C, D]
    """
    device = feats.device
    dim = feats.shape[1]
    protos = torch.zeros((num_classes, dim), dtype=feats.dtype, device=device)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            raise ValueError(f"No samples found for class {ID_TO_CLASS[c]}")
        protos[c] = feats[mask].mean(dim=0)

    protos = l2_normalize(protos)
    return protos


# -------------------------
# Dataset
# -------------------------

class InstanceDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        self.feats = feats.detach().cpu()
        self.labels = labels.detach().cpu()

    def __len__(self) -> int:
        return self.feats.shape[0]

    def __getitem__(self, idx: int):
        return self.feats[idx], self.labels[idx]


# -------------------------
# Model
# -------------------------

class ResidualMapper(nn.Module):
    """
    A-space prototype -> B-space prototype
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.net(x)
        if self.normalize_output:
            out = l2_normalize(out)
        return out


class ClassHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProtoMapperModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        mapper_hidden_dim: int = 512,
        cls_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mapper = ResidualMapper(
            dim=feat_dim,
            hidden_dim=mapper_hidden_dim,
            dropout=dropout,
            normalize_output=True,
        )
        self.class_head = ClassHead(
            dim=feat_dim,
            num_classes=num_classes,
            hidden_dim=cls_hidden_dim,
            dropout=dropout,
        )

    def map_prototypes(self, proto_a: torch.Tensor) -> torch.Tensor:
        return self.mapper(proto_a)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.class_head(z)


# -------------------------
# Losses
# -------------------------

def prototype_alignment_loss(
    mapped_proto_a: torch.Tensor,
    proto_b: torch.Tensor,
) -> torch.Tensor:
    """
    Direct alignment between converted A prototypes and B prototypes
    """
    return (1.0 - F.cosine_similarity(mapped_proto_a, proto_b, dim=-1)).mean()


def class_head_loss(
    class_head: nn.Module,
    mapped_proto_a: torch.Tensor,
    proto_b: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Train class head on:
    - converted A prototypes
    - B prototypes
    """
    device = mapped_proto_a.device
    num_classes = mapped_proto_a.shape[0]
    labels = torch.arange(num_classes, device=device)

    logits_a = class_head(mapped_proto_a)
    logits_b = class_head(proto_b)

    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss = 0.5 * (loss_a + loss_b)

    with torch.no_grad():
        acc_a = (logits_a.argmax(dim=1) == labels).float().mean().item()
        acc_b = (logits_b.argmax(dim=1) == labels).float().mean().item()

    return loss, {"proto_acc_A2B": acc_a, "proto_acc_B": acc_b}


def instance_to_converted_proto_ce_loss(
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    mapped_proto_a: torch.Tensor,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Main fine-tuning loss:
    B instances vs converted A prototypes

    Same class should be close.
    Different classes should be far.
    """
    logits = cosine_logits(b_feats, mapped_proto_a, temperature=temperature)
    loss = F.cross_entropy(logits, b_labels)

    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == b_labels).float().mean().item()

    return loss, {"b_instance_proto_acc": acc}

@torch.no_grad()
def compute_confusion_matrix(
    model: ProtoMapperModel,
    proto_a: torch.Tensor,
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Returns confusion matrix with:
      rows = true class
      cols = predicted class
    """
    model.eval()

    mapped_proto_a = model.map_prototypes(proto_a)   # [C, D]
    logits = cosine_logits(b_feats, mapped_proto_a, temperature=temperature)  # [N, C]
    preds = logits.argmax(dim=1)

    num_classes = proto_a.shape[0]
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=b_feats.device)

    for t, p in zip(b_labels, preds):
        cm[t.long(), p.long()] += 1

    return cm

@torch.no_grad()
def print_fp_sources_from_confusion_matrix(cm: torch.Tensor, class_names: list[str]) -> None:
    """
    cm: rows=true, cols=pred
    For each predicted class j:
      false positives are rows i!=j in column j
    """
    cm_np = cm.cpu().numpy()
    num_classes = len(class_names)

    print("\nFalse-positive source breakdown by predicted category:")
    for pred_j in range(num_classes):
        pred_name = class_names[pred_j]

        col = cm_np[:, pred_j].copy()
        tp = col[pred_j]
        col[pred_j] = 0   # remove TP, keep only FP sources
        total_fp = col.sum()

        print(f"\nPredicted as '{pred_name}':")
        print(f"  TP = {tp}")
        print(f"  Total FP = {total_fp}")

        if total_fp == 0:
            print("  No false positives.")
            continue

        # sort source classes by contribution
        order = np.argsort(-col)
        for true_i in order:
            if col[true_i] == 0:
                continue
            true_name = class_names[true_i]
            frac = col[true_i] / total_fp
            print(f"  from true '{true_name}': {col[true_i]} ({frac:.2%})")

@torch.no_grad()
def compute_confusion_matrix_from_prototypes(
    proto_a: torch.Tensor,
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    temperature: float = 0.07,
    model=None,
) -> torch.Tensor:
    """
    rows = true class
    cols = predicted class

    If model is None:
        use raw A prototypes (before training / no conversion)

    If model is not None:
        use converted A prototypes = model.map_prototypes(proto_a)
    """
    if model is not None:
        model.eval()
        proto_used = model.map_prototypes(proto_a)
    else:
        proto_used = proto_a

    proto_used = l2_normalize(proto_used)
    b_feats = l2_normalize(b_feats)

    logits = cosine_logits(b_feats, proto_used, temperature=temperature)  # [N, C]
    preds = logits.argmax(dim=1)

    num_classes = proto_a.shape[0]
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=b_feats.device)

    for t, p in zip(b_labels, preds):
        cm[t.long(), p.long()] += 1

    return cm

def plot_confusion_matrix(
    cm,
    class_names,
    save_path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize=(10, 8),
    cmap="Blues",
):
    """
    cm: torch.Tensor or np.ndarray, shape [C, C]
    rows = true class
    cols = predicted class
    """
    if isinstance(cm, torch.Tensor):
        cm = cm.detach().cpu().numpy()
    else:
        cm = np.asarray(cm)

    cm_plot = cm.astype(np.float64).copy()

    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # annotate cells
    if normalize:
        fmt = ".2f"
        thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.5
    else:
        fmt = "d"
        thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 1

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_display = cm_plot[i, j] if normalize else int(cm[i, j])
            ax.text(
                j, i,
                format(val_display, fmt),
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def print_fp_sources_from_confusion_matrix(cm: torch.Tensor, class_names: list[str]) -> None:
    """
    cm: rows=true, cols=pred
    For each predicted class j:
      false positives are rows i!=j in column j
    """
    cm_np = cm.detach().cpu().numpy()
    num_classes = len(class_names)

    print("\nFalse-positive source breakdown by predicted category:")
    for pred_j in range(num_classes):
        pred_name = class_names[pred_j]

        col = cm_np[:, pred_j].copy()
        tp = int(col[pred_j])
        col[pred_j] = 0
        total_fp = int(col.sum())

        print(f"\nPredicted as '{pred_name}':")
        print(f"  TP = {tp}")
        print(f"  Total FP = {total_fp}")

        if total_fp == 0:
            print("  No false positives.")
            continue

        order = np.argsort(-col)
        for true_i in order:
            if col[true_i] == 0:
                continue
            true_name = class_names[true_i]
            frac = col[true_i] / total_fp
            print(f"  from true '{true_name}': {int(col[true_i])} ({frac:.2%})")

def instance_to_converted_proto_margin_loss(
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    mapped_proto_a: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Optional margin separation loss:
    make same-class prototype similarity larger than hardest wrong-class similarity
    """
    sims = l2_normalize(b_feats) @ l2_normalize(mapped_proto_a).T  # [N, C]

    pos = sims.gather(1, b_labels[:, None]).squeeze(1)

    neg_mask = torch.ones_like(sims, dtype=torch.bool)
    neg_mask.scatter_(1, b_labels[:, None], False)
    neg = sims.masked_fill(~neg_mask, -1e9).max(dim=1).values

    loss = F.relu(margin - pos + neg).mean()
    return loss


def mapped_proto_distribution_loss(
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    mapped_proto_a: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Optional: each B instance should induce a distribution that peaks at its class prototype.
    This is similar to CE, but kept separate if you want weighting control later.
    """
    logits = cosine_logits(b_feats, mapped_proto_a, temperature=temperature)
    return F.cross_entropy(logits, b_labels)


# -------------------------
# Eval
# -------------------------

@torch.no_grad()
def evaluate_per_class(
    model: ProtoMapperModel,
    proto_a: torch.Tensor,
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    temperature: float = 0.07,
) -> Dict[str, Dict[str, float]]:
    model.eval()

    mapped_proto_a = model.map_prototypes(proto_a)   # [C, D]
    logits = cosine_logits(b_feats, mapped_proto_a, temperature=temperature)  # [N, C]
    preds = logits.argmax(dim=1)

    results = {}
    num_classes = proto_a.shape[0]

    for c in range(num_classes):
        mask = (b_labels == c)
        total = int(mask.sum().item())

        if total == 0:
            results[ID_TO_CLASS[c]] = {
                "total": 0,
                "correct": 0,
                "acc": float("nan"),
            }
            continue

        correct = int((preds[mask] == b_labels[mask]).sum().item())
        acc = correct / total

        results[ID_TO_CLASS[c]] = {
            "total": total,
            "correct": correct,
            "acc": acc,
        }

    return results

@torch.no_grad()
def evaluate(
    model: ProtoMapperModel,
    proto_a: torch.Tensor,
    proto_b: torch.Tensor,
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    temperature: float,
    margin: float,
) -> Dict[str, float]:
    model.eval()
    mapped_proto_a = model.map_prototypes(proto_a)

    loss_align = prototype_alignment_loss(mapped_proto_a, proto_b)
    loss_cls, cls_stats = class_head_loss(model.class_head, mapped_proto_a, proto_b)
    loss_inst_ce, inst_stats = instance_to_converted_proto_ce_loss(
        b_feats=b_feats,
        b_labels=b_labels,
        mapped_proto_a=mapped_proto_a,
        temperature=temperature,
    )
    loss_margin = instance_to_converted_proto_margin_loss(
        b_feats=b_feats,
        b_labels=b_labels,
        mapped_proto_a=mapped_proto_a,
        margin=margin,
    )

    per_class = evaluate_per_class(
        model=model,
        proto_a=proto_a,
        b_feats=b_feats,
        b_labels=b_labels,
        temperature=temperature,
    )

    out = {
        "eval_align_loss": float(loss_align.item()),
        "eval_cls_loss": float(loss_cls.item()),
        "eval_inst_ce_loss": float(loss_inst_ce.item()),
        "eval_margin_loss": float(loss_margin.item()),
        "eval_proto_acc_A2B": cls_stats["proto_acc_A2B"],
        "eval_proto_acc_B": cls_stats["proto_acc_B"],
        "eval_b_instance_proto_acc": inst_stats["b_instance_proto_acc"],
    }

    # flatten per-class results into loggable keys
    for cls_name, stats in per_class.items():
        safe_name = cls_name.replace(" ", "_")
        out[f"{safe_name}_acc"] = stats["acc"]
        out[f"{safe_name}_total"] = stats["total"]
        out[f"{safe_name}_correct"] = stats["correct"]

    return out



# -------------------------
# Training
# -------------------------

def train(
    model: ProtoMapperModel,
    proto_a: torch.Tensor,
    proto_b: torch.Tensor,
    b_feats: torch.Tensor,
    b_labels: torch.Tensor,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    lambda_align: float,
    lambda_cls: float,
    lambda_inst_ce: float,
    lambda_margin: float,
    margin: float,
):
    
    device = b_feats.device

    ds_b = InstanceDataset(b_feats, b_labels)
    dl_b = DataLoader(ds_b, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -math.inf
    best_path = out_dir / "best_model.pt"
    log_path = out_dir / "train_log.jsonl"

    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            "total": 0.0,
            "align": 0.0,
            "cls": 0.0,
            "inst_ce": 0.0,
            "margin": 0.0,
        }
        steps = 0

        for batch_feats_cpu, batch_labels_cpu in dl_b:
            batch_feats = batch_feats_cpu.to(device, non_blocking=True)
            batch_labels = batch_labels_cpu.to(device, non_blocking=True)

            mapped_proto_a = model.map_prototypes(proto_a)

            loss_align = prototype_alignment_loss(mapped_proto_a, proto_b)

            loss_cls, _ = class_head_loss(
                class_head=model.class_head,
                mapped_proto_a=mapped_proto_a,
                proto_b=proto_b,
            )

            loss_inst_ce, _ = instance_to_converted_proto_ce_loss(
                b_feats=batch_feats,
                b_labels=batch_labels,
                mapped_proto_a=mapped_proto_a,
                temperature=temperature,
            )

            loss_margin = instance_to_converted_proto_margin_loss(
                b_feats=batch_feats,
                b_labels=batch_labels,
                mapped_proto_a=mapped_proto_a,
                margin=margin,
            )

            loss = (
                lambda_align * loss_align
                + lambda_cls * loss_cls
                + lambda_inst_ce * loss_inst_ce
                + lambda_margin * loss_margin
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running["total"] += float(loss.item())
            running["align"] += float(loss_align.item())
            running["cls"] += float(loss_cls.item())
            running["inst_ce"] += float(loss_inst_ce.item())
            running["margin"] += float(loss_margin.item())
            steps += 1

        avg_train = {k: v / max(steps, 1) for k, v in running.items()}

        eval_stats = evaluate(
            model=model,
            proto_a=proto_a,
            proto_b=proto_b,
            b_feats=b_feats,
            b_labels=b_labels,
            temperature=temperature,
            margin=margin,
        )
        print("Per-class eval acc:")

        score = (
            2.0 * eval_stats["eval_b_instance_proto_acc"]
            + 0.5 * eval_stats["eval_proto_acc_A2B"]
            + 0.5 * eval_stats["eval_proto_acc_B"]
            - 0.2 * eval_stats["eval_align_loss"]
            - 0.2 * eval_stats["eval_inst_ce_loss"]
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in avg_train.items()},
            **eval_stats,
            "model_score": score,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[{epoch:03d}/{epochs:03d}] "
            f"train_total={avg_train['total']:.4f} "
            f"align={avg_train['align']:.4f} "
            f"cls={avg_train['cls']:.4f} "
            f"inst_ce={avg_train['inst_ce']:.4f} "
            f"margin={avg_train['margin']:.4f} "
            f"| eval_inst_acc={eval_stats['eval_b_instance_proto_acc']:.4f} "
            f"eval_protoA2B={eval_stats['eval_proto_acc_A2B']:.4f} "
            f"eval_protoB={eval_stats['eval_proto_acc_B']:.4f}"
        )
        for cls_name in CLASS_NAMES:
            safe_name = cls_name.replace(" ", "_")
            acc = eval_stats.get(f"{safe_name}_acc", float("nan"))
            total = eval_stats.get(f"{safe_name}_total", 0)
            correct = eval_stats.get(f"{safe_name}_correct", 0)
            print(f"  {cls_name:15s} acc={acc:.4f} ({correct}/{total})")
            
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_score": best_score,
                    "class_names": CLASS_NAMES,
                    "feat_dim": proto_a.shape[1],
                    "num_classes": len(CLASS_NAMES),
                    "proto_a": proto_a.detach().cpu(),
                    "proto_b": proto_b.detach().cpu(),
                },
                best_path,
            )

    print(f"\nBest model saved to: {best_path}")


# -------------------------
# Main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_dir", type=str, required=True,
                        help="Folder with domain A instance features (.npy), one file per instance")
    parser.add_argument("--b_dir", type=str, required=True,
                        help="Folder with domain B instance features (.npy), one file per instance")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--mapper_hidden_dim", type=int, default=512)
    parser.add_argument("--cls_hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.2)

    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lambda_inst_ce", type=float, default=1.0)
    parser.add_argument("--lambda_margin", type=float, default=0.2)

    parser.add_argument("--max_features", type=int, default=None,
                        help="Maximum number of feature files per class to process (randomly shuffled). If None, use all files.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_dir = Path(args.a_dir)
    b_dir = Path(args.b_dir)

    a_feats_np, a_labels_np, a_skipped = load_feature_folder(a_dir, CLASS_NAMES, max_features=args.max_features)
    b_feats_np, b_labels_np, b_skipped = load_feature_folder(b_dir, CLASS_NAMES, max_features=args.max_features)

    if a_skipped:
        print(f"Skipped {len(a_skipped)} files in A with no matched class.")
    if b_skipped:
        print(f"Skipped {len(b_skipped)} files in B with no matched class.")

    a_feats = torch.tensor(a_feats_np, dtype=torch.float32, device=device)
    a_labels = torch.tensor(a_labels_np, dtype=torch.long, device=device)
    b_feats = torch.tensor(b_feats_np, dtype=torch.float32, device=device)
    b_labels = torch.tensor(b_labels_np, dtype=torch.long, device=device)

    a_feats = l2_normalize(a_feats)
    b_feats = l2_normalize(b_feats)

    num_classes = len(CLASS_NAMES)
    feat_dim = a_feats.shape[1]

    print(f"\nA dataset: {len(a_labels)} samples")
    print("Class distribution in A:")
    for c in range(num_classes):
        count = (a_labels == c).sum().item()
        print(f"  {ID_TO_CLASS[c]}: {count} samples")

    print(f"\nB dataset: {len(b_labels)} samples")
    print("Class distribution in B:")
    for c in range(num_classes):
        count = (b_labels == c).sum().item()
        print(f"  {ID_TO_CLASS[c]}: {count} samples")

    proto_a = compute_class_prototypes(a_feats, a_labels, num_classes=num_classes)
    proto_b = compute_class_prototypes(b_feats, b_labels, num_classes=num_classes)

    print("\nInstance counts:")
    for c in range(num_classes):
        na = int((a_labels == c).sum().item())
        nb = int((b_labels == c).sum().item())
        print(f"  {ID_TO_CLASS[c]:15s}  A={na:5d}  B={nb:5d}")

    model = ProtoMapperModel(
        feat_dim=feat_dim,
        num_classes=num_classes,
        mapper_hidden_dim=args.mapper_hidden_dim,
        cls_hidden_dim=args.cls_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    
    cm_before = compute_confusion_matrix_from_prototypes(
        proto_a=proto_a,
        b_feats=b_feats,
        b_labels=b_labels,
        temperature=args.temperature,
        model=None,   # no conversion
    )

    plot_confusion_matrix(
        cm_before,
        class_names=CLASS_NAMES,
        save_path=out_dir / "cm_before_counts.png",
        normalize=True,
        title="Confusion Matrix Before Training (Counts)",
        )
    
    train(
        model=model,
        proto_a=proto_a,
        proto_b=proto_b,
        b_feats=b_feats,
        b_labels=b_labels,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        lambda_align=args.lambda_align,
        lambda_cls=args.lambda_cls,
        lambda_inst_ce=args.lambda_inst_ce,
        lambda_margin=args.lambda_margin,
        margin=args.margin,
    )

    cm = compute_confusion_matrix(
    model=model,
    proto_a=proto_a,
    b_feats=b_feats,
    b_labels=b_labels,
    temperature=args.temperature,
    )

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm.cpu().numpy())

    print_fp_sources_from_confusion_matrix(cm, CLASS_NAMES)
    plot_confusion_matrix(
        cm,
        class_names=CLASS_NAMES,
        save_path=out_dir / "cm_after_counts.png",
        normalize=True,
        title="Confusion Matrix After Training (Counts)",
    )

if __name__ == "__main__":
    main()


# from the output, importance inst_ce > align > margin > cls loss
