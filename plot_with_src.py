import os
import csv
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# -----------------------------
# I/O helpers
# -----------------------------
def load_labels(path):
    if path is None:
        return None
    if path.endswith(".npy"):
        labels = np.load(path, allow_pickle=True)
        return labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def read_txt(path):
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return None
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def save_coords_csv(path, proto_labels, proto_before_xy, proto_after_xy, inst_labels, inst_xy):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "class_name", "x", "y"])

        for i, cls in enumerate(proto_labels):
            writer.writerow(["prototype_before", f"{cls}_before", cls,
                             proto_before_xy[i, 0], proto_before_xy[i, 1]])
        for i, cls in enumerate(proto_labels):
            writer.writerow(["prototype_after", f"{cls}_after", cls,
                             proto_after_xy[i, 0], proto_after_xy[i, 1]])
        for i, cls in enumerate(inst_labels):
            writer.writerow(["instance", f"inst_{i}", cls,
                             inst_xy[i, 0], inst_xy[i, 1]])


def save_class_avg_distances_csv(path, class_avg_dists):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "avg_dist_before", "avg_dist_after", "n_instances"])
        for cls, vals in class_avg_dists.items():
            writer.writerow([cls, vals["before"], vals["after"], vals["n_instances"]])


# -----------------------------
# Feature / projection helpers
# -----------------------------
def l2_normalize_np(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def ensure_same_dim(*arrays):
    dims = [arr.shape[1] for arr in arrays]
    if len(set(dims)) != 1:
        raise ValueError(f"Feature dims do not match: {dims}")


def project_features(all_feats, method="pca", random_state=0):
    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(all_feats)
    elif method == "mds":
        reducer = MDS(
            n_components=2,
            dissimilarity="euclidean",
            random_state=random_state,
            normalized_stress="auto",
        )
        coords = reducer.fit_transform(all_feats)
    else:
        raise ValueError(f"Unknown method: {method}")
    return coords


def parse_keep_classes(keep_classes_arg):
    if keep_classes_arg is None:
        return None
    classes = [x.strip() for x in keep_classes_arg.split(",") if x.strip()]
    return set(classes) if classes else None


def filter_by_classes(
    proto_before,
    proto_after,
    proto_labels,
    instances,
    inst_labels,
    instance_image_paths=None,
    keep_classes=None,
    drop_proto_without_instances=True,
):
    if keep_classes is not None:
        proto_idx = [i for i, c in enumerate(proto_labels) if c in keep_classes]
        inst_idx = [i for i, c in enumerate(inst_labels) if c in keep_classes]
    else:
        proto_idx = list(range(len(proto_labels)))
        inst_idx = list(range(len(inst_labels)))

    proto_before_f = proto_before[proto_idx]
    proto_after_f = proto_after[proto_idx]
    proto_labels_f = [proto_labels[i] for i in proto_idx]

    instances_f = instances[inst_idx]
    inst_labels_f = [inst_labels[i] for i in inst_idx]

    if instance_image_paths is not None:
        instance_image_paths_f = [instance_image_paths[i] for i in inst_idx]
    else:
        instance_image_paths_f = None

    if drop_proto_without_instances:
        inst_class_set = set(inst_labels_f)
        proto_idx2 = [i for i, c in enumerate(proto_labels_f) if c in inst_class_set]
        proto_before_f = proto_before_f[proto_idx2]
        proto_after_f = proto_after_f[proto_idx2]
        proto_labels_f = [proto_labels_f[i] for i in proto_idx2]

    return proto_before_f, proto_after_f, proto_labels_f, instances_f, inst_labels_f, instance_image_paths_f


def split_projected_coords(proto_before, proto_after, instances, method="pca"):
    n_proto = proto_before.shape[0]
    n_inst = instances.shape[0]

    all_feats = np.concatenate([proto_before, proto_after, instances], axis=0)
    coords = project_features(all_feats, method=method)

    proto_before_xy = coords[:n_proto]
    proto_after_xy = coords[n_proto:2 * n_proto]
    inst_xy = coords[2 * n_proto:2 * n_proto + n_inst]
    return proto_before_xy, proto_after_xy, inst_xy


def plot_projection(
    proto_labels,
    proto_before_xy,
    proto_after_xy,
    inst_labels,
    inst_xy,
    out_path,
    title="Prototype Projection Before and After Mapping",
    show_text=True,
    alpha_instances=0.75,
):
    colors = class_color_map(proto_labels)
    fig, ax = plt.subplots(figsize=(12, 10))

    setup_axes(ax, proto_before_xy, proto_after_xy, inst_xy, title)
    add_instance_scatter(ax, inst_xy, inst_labels, colors, alpha=alpha_instances, size=55)

    for i, cls in enumerate(proto_labels):
        c = colors[cls]
        xb, yb = proto_before_xy[i]
        xa, ya = proto_after_xy[i]

        ax.scatter(xb, yb, s=220, c=[c], marker="X", edgecolors="black", linewidths=1.0, zorder=4)
        ax.scatter(xa, ya, s=260, c=[c], marker="*", edgecolors="black", linewidths=1.0, zorder=5)

        ax.annotate(
            "",
            xy=(xa, ya),
            xytext=(xb, yb),
            arrowprops=dict(arrowstyle="->", linestyle="--", lw=1.8, color=c),
        )

        if show_text:
            ax.text(xb, yb, f"{cls}\n(before)", fontsize=14, ha="right", va="bottom")
            ax.text(xa, ya, f"{cls}\n(after)", fontsize=14, ha="left", va="top")

    add_marker_legend(ax)
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Distance helpers
# -----------------------------
def compute_class_avg_distances(
    proto_before_xy,
    proto_after_xy,
    inst_xy,
    proto_labels,
    inst_labels,
):
    results = {}

    unique_proto_classes = list(dict.fromkeys(proto_labels))
    for cls in unique_proto_classes:
        pidx = [i for i, c in enumerate(proto_labels) if c == cls]
        iidx = [i for i, c in enumerate(inst_labels) if c == cls]

        if len(pidx) == 0 or len(iidx) == 0:
            continue

        pb = proto_before_xy[pidx].mean(axis=0)
        pa = proto_after_xy[pidx].mean(axis=0)
        inst_pts = inst_xy[iidx]

        d_before = np.linalg.norm(inst_pts - pb[None, :], axis=1)
        d_after = np.linalg.norm(inst_pts - pa[None, :], axis=1)

        results[cls] = {
            "before": float(d_before.mean()),
            "after": float(d_after.mean()),
            "n_instances": int(len(iidx)),
        }

    return results


# -----------------------------
# Plot helpers
# -----------------------------
def class_color_map(class_names):
    unique_classes = list(dict.fromkeys(class_names))
    return {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}


def compute_axis_limits(proto_before_xy, proto_after_xy, inst_xy, pad_ratio=0.15):
    all_xy = np.concatenate([proto_before_xy, proto_after_xy, inst_xy], axis=0)
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    padx = pad_ratio * max(1e-6, xmax - xmin)
    pady = pad_ratio * max(1e-6, ymax - ymin)
    return (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)


def setup_axes(ax, proto_before_xy, proto_after_xy, inst_xy, title):
    (xmin, xmax), (ymin, ymax) = compute_axis_limits(proto_before_xy, proto_after_xy, inst_xy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Space Feature 1", fontsize=16)
    ax.set_ylabel("Space Feature 2", fontsize=16)
    ax.set_title(title, fontsize=18)
    # Add white background to title so it covers thumbnails
    ax.title.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.95))
    ax.title.set_zorder(1000)
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    ax.set_yticks([])


def add_thumbnail(
    ax,
    xy,
    img_path,
    zoom=0.30,
    border_color=None,
    border_width=3,
    min_zoom_ratio=0.28,
    max_fraction_of_axis=0.10,
    edge_margin_ratio=0.01,
):
    """
    Add a thumbnail at data position xy.

    It keeps the requested zoom when reasonable, but shrinks the image
    when it would be too large for the current plot area or too close to
    the boundary.
    """
    if img_path is None or not os.path.exists(img_path):
        return None

    try:
        img = plt.imread(img_path)
        h, w = img.shape[:2]

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax_bbox = ax.get_window_extent()
        ax_w_px = max(ax_bbox.width, 1.0)
        ax_h_px = max(ax_bbox.height, 1.0)

        x_per_px = (xmax - xmin) / ax_w_px
        y_per_px = (ymax - ymin) / ax_h_px

        x, y = xy

        req_half_w = 0.5 * w * zoom * x_per_px
        req_half_h = 0.5 * h * zoom * y_per_px

        max_half_w = 0.5 * max_fraction_of_axis * (xmax - xmin)
        max_half_h = 0.5 * max_fraction_of_axis * (ymax - ymin)
        scale_size = min(
            1.0,
            max_half_w / max(req_half_w, 1e-12),
            max_half_h / max(req_half_h, 1e-12),
        )

        margin_x = edge_margin_ratio * (xmax - xmin)
        margin_y = edge_margin_ratio * (ymax - ymin)

        avail_left = max(x - xmin - margin_x, 1e-12)
        avail_right = max(xmax - x - margin_x, 1e-12)
        avail_down = max(y - ymin - margin_y, 1e-12)
        avail_up = max(ymax - y - margin_y, 1e-12)

        avail_half_w = min(avail_left, avail_right)
        avail_half_h = min(avail_down, avail_up)

        scale_edge = min(
            1.0,
            avail_half_w / max(req_half_w, 1e-12),
            avail_half_h / max(req_half_h, 1e-12),
        )

        fit_scale = min(scale_size, scale_edge)
        if fit_scale < 1.0:
            fit_scale = max(fit_scale, min_zoom_ratio)

        final_zoom = zoom * fit_scale
        imagebox = OffsetImage(img, zoom=final_zoom)

        if border_color is not None:
            ab = AnnotationBbox(
                imagebox,
                (x, y),
                frameon=True,
                pad=0.1,
                bboxprops=dict(
                    edgecolor=border_color,
                    linewidth=border_width,
                    facecolor="none",
                    boxstyle="square,pad=0.0",
                ),
                box_alignment=(0.5, 0.5),
                annotation_clip=True,
            )
        else:
            ab = AnnotationBbox(
                imagebox,
                (x, y),
                frameon=False,
                pad=0.0,
                box_alignment=(0.5, 0.5),
                annotation_clip=True,
            )

        ax.add_artist(ab)
        return ab

    except Exception as e:
        print(f"Failed to add thumbnail {img_path}: {e}")
        return None


def select_thumbnail_indices(inst_xy, inst_labels, max_per_class=2):
    selected = []
    classes = list(dict.fromkeys(inst_labels))
    for cls in classes:
        idx = [i for i, x in enumerate(inst_labels) if x == cls]
        if not idx:
            continue
        pts = inst_xy[idx]
        center = pts.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(pts - center, axis=1)
        order = np.argsort(dist)
        chosen = [idx[j] for j in order[:max_per_class]]
        selected.extend(chosen)
    return sorted(selected)


def add_source_background_thumbnails(
    ax,
    source_crop_xy,
    source_crop_paths,
    source_crop_labels=None,
    keep_classes=None,
    thumbnail_zoom=0.10,
    alpha=0.16,
    zorder=1,
    border_color="blue",
    border_width=2,
):
    artists = []
    for i, (xy, img_path) in enumerate(zip(source_crop_xy, source_crop_paths)):
        if keep_classes is not None and source_crop_labels is not None:
            if source_crop_labels[i] not in keep_classes:
                continue

        artist = add_thumbnail(
            ax,
            xy,
            img_path,
            zoom=thumbnail_zoom,
            border_color=border_color,
            border_width=border_width,
        )
        if artist is not None:
            artist.set_alpha(alpha)
            try:
                artist.zorder = zorder
            except Exception:
                pass
            artists.append(artist)
    return artists


def add_instance_scatter(ax, inst_xy, inst_labels, color_map, alpha=0.65, size=50):
    for cls in list(dict.fromkeys(inst_labels)):
        idx = [i for i, x in enumerate(inst_labels) if x == cls]
        if not idx:
            continue
        pts = inst_xy[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=size,
            c=[color_map[cls]],
            alpha=alpha,
            marker="o",
            edgecolors="none",
            zorder=2,
        )


def add_thumbnails(ax, inst_xy, inst_labels, instance_image_paths, thumbs_per_class=1, thumbnail_zoom=0.30, border_color='red', border_width=3):
    artists = []
    if instance_image_paths is None:
        return artists
    selected_idx = select_thumbnail_indices(inst_xy, inst_labels, max_per_class=thumbs_per_class)
    for i in selected_idx:
        artist = add_thumbnail(ax, inst_xy[i], instance_image_paths[i], zoom=thumbnail_zoom, border_color=border_color, border_width=border_width)
        if artist is not None:
            artists.append(artist)
    return artists


def add_marker_legend(ax):
    marker_guide = [
        plt.Line2D([0], [0], marker='X', linestyle='', label='Prototype before',
                   markersize=10, markerfacecolor='gray', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='*', linestyle='', label='Prototype after / moving',
                   markersize=12, markerfacecolor='gray', markeredgecolor='black'),
    ]
    ax.legend(
        handles=marker_guide,
        loc='upper left',
        bbox_to_anchor=(0, -0.05),
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10
    )




# -----------------------------
# Animation save helper
# -----------------------------
def save_animation(anim, out_path, fps=20, dpi_mp4=200, dpi_gif=140):
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    if out_path.lower().endswith(".mp4"):
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer, dpi=dpi_mp4)
    elif out_path.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=dpi_gif)
    else:
        raise ValueError("Output must end with .mp4 or .gif")

def load_source_crops_and_features(
    source_crop_dir,
    source_feature_dir,
    allowed_classes=None,
    image_exts=(".png", ".jpg", ".jpeg", ".webp"),
):
    """
    Match source crop images with source feature .npy files by filename stem.

    Example:
        crop dir:     airplane_001.png
        feature dir:  airplane_001.npy

    Returns:
        source_feats: [Ns, D]
        source_paths: list[str]
        source_labels: list[str]
    """
    source_crop_dir = Path(source_crop_dir)
    source_feature_dir = Path(source_feature_dir)

    feat_files = sorted(source_feature_dir.glob("*.npy"))
    if len(feat_files) == 0:
        raise ValueError(f"No .npy files found in {source_feature_dir}")

    # build image lookup by stem
    img_map = {}
    for ext in image_exts:
        for p in source_crop_dir.glob(f"*{ext}"):
            img_map[p.stem] = str(p)

    feats = []
    paths = []
    labels = []

    for feat_path in feat_files:
        stem = feat_path.stem
        if stem not in img_map:
            continue

        feat = np.load(feat_path, allow_pickle=True)
        if feat.ndim == 1:
            feat = feat[None, :]
        elif feat.ndim != 2 or feat.shape[0] != 1:
            raise ValueError(f"Expected one feature vector per file, got shape {feat.shape} in {feat_path}")

        label = infer_class_from_name(stem, allowed_classes=allowed_classes)
        if allowed_classes is not None and label not in allowed_classes:
            continue

        feats.append(feat[0])
        paths.append(img_map[stem])
        labels.append(label)

    if len(feats) == 0:
        raise ValueError("No matched source crop/feature pairs found.")

    return np.stack(feats, axis=0), paths, labels


def infer_class_from_name(name, allowed_classes=None):
    """
    Infer class from filename stem.
    If allowed_classes is given, match the longest class string found in the stem.
    """
    stem = str(name).lower()

    if allowed_classes is not None:
        allowed = sorted([c.lower() for c in allowed_classes], key=len, reverse=True)
        for cls in allowed:
            if cls in stem:
                return cls
        raise ValueError(f"Could not infer class from: {name}")

    # fallback: take the first token
    parts = re.split(r"[_\-\s]+", stem)
    return parts[0]


def split_projected_coords_with_source(
    proto_before,
    proto_after,
    instances,
    source_feats,
    method="pca",
):
    """
    Project prototypes-before, prototypes-after, target instances,
    and source crop features into one shared 2D space.
    """
    n_proto = proto_before.shape[0]
    n_inst = instances.shape[0]
    n_src = source_feats.shape[0]

    all_feats = np.concatenate([proto_before, proto_after, instances, source_feats], axis=0)
    coords = project_features(all_feats, method=method)

    proto_before_xy = coords[:n_proto]
    proto_after_xy = coords[n_proto:2 * n_proto]
    inst_xy = coords[2 * n_proto:2 * n_proto + n_inst]
    source_xy = coords[2 * n_proto + n_inst: 2 * n_proto + n_inst + n_src]

    return proto_before_xy, proto_after_xy, inst_xy, source_xy
    

    
    
def make_focus_zoom_animation(
    proto_before_xy,
    proto_after_xy,
    inst_xy,
    proto_labels,
    inst_labels,
    focus_class,
    out_path,
    instance_image_paths=None,
    show_thumbnails=False,
    thumbs_per_class=1,
    thumbnail_zoom=0.24,
    fps=20,
    whole_move_seconds=3,
    zoom_seconds=2,
    focus_move_seconds=3,
    final_hold_seconds=2,
    title=None,
    inactive_instance_alpha=0.9,
    inactive_proto_alpha=0.9,
    active_instance_alpha=0.90,
    active_proto_alpha=1.00,
    label_dx_ratio=0.020,
    label_dy_ratio=0.040,
    show_distance_bars=True,
    distance_bar_alpha=0.80,
    source_crop_xy=None,
    source_crop_paths=None,
    source_crop_labels=None,
    source_crop_zoom=0.10,
    source_crop_alpha=0.16,
):
    def fmt_label(s):
        s = str(s).replace("_", " ").strip()
        return s[:1].upper() + s[1:] if s else s

    focus_class_disp = fmt_label(focus_class)

    unique_classes = [c for c in dict.fromkeys(proto_labels) if c in set(inst_labels)]
    if focus_class not in unique_classes:
        raise ValueError(f"focus_class '{focus_class}' not found in filtered classes: {unique_classes}")

    color_map = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}

    if show_distance_bars:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.4], wspace=0.18, top=0.93, bottom=0.10)
        ax = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.subplots_adjust(top=0.93, bottom=0.10)
        ax_bar = None

    # ----- global limits
    xy_list = [proto_before_xy, proto_after_xy, inst_xy]
    if source_crop_xy is not None and len(source_crop_xy) > 0:
        xy_list.append(source_crop_xy)
    all_xy = np.concatenate(xy_list, axis=0)

    gxmin, gymin = all_xy.min(axis=0)
    gxmax, gymax = all_xy.max(axis=0)
    gpadx = 0.08 * max(1e-6, gxmax - gxmin)
    gpady = 0.08 * max(1e-6, gymax - gymin)
    gxmin, gxmax = gxmin - gpadx, gxmax + gpadx
    gymin, gymax = gymin - gpady, gymax + gpady

    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymin, gymax)
    ax.set_xlabel("Feature space 1", fontsize=16)
    ax.set_ylabel("Feature space 2", fontsize=16)
    ax.set_title("" if title is None else fmt_label(title), fontsize=18)
    # Add white background to title so it covers thumbnails
    title_obj = ax.title
    title_obj.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.95))
    title_obj.set_zorder(1000)  # High zorder to appear on top
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    ax.set_yticks([])

    # ----- focus local limits
    pidx_focus = [i for i, c in enumerate(proto_labels) if c == focus_class]
    iidx_focus = [i for i, c in enumerate(inst_labels) if c == focus_class]

    if len(pidx_focus) == 0 or len(iidx_focus) == 0:
        raise ValueError(f"focus_class '{focus_class}' must have both prototype(s) and instance(s)")

    focus_points = np.concatenate([
        inst_xy[iidx_focus],
        proto_before_xy[pidx_focus],
        proto_after_xy[pidx_focus],
    ], axis=0)

    fxmin, fymin = focus_points.min(axis=0)
    fxmax, fymax = focus_points.max(axis=0)
    fpadx = 0.22 * max(1e-6, fxmax - fxmin)
    fpady = 0.22 * max(1e-6, fymax - fymin)
    fxmin, fxmax = fxmin - fpadx, fxmax + fpadx
    fymin, fymax = fymin - fpady, fymax + fpady

    xspan_global = gxmax - gxmin
    yspan_global = gymax - gymin

    def label_offset(x, y, xspan, yspan):
        return x + label_dx_ratio * xspan, y + label_dy_ratio * yspan

    # ----- source crop background
    source_bg_artists = []
    if source_crop_xy is not None and source_crop_paths is not None:
        source_bg_artists = add_source_background_thumbnails(
            ax=ax,
            source_crop_xy=source_crop_xy,
            source_crop_paths=source_crop_paths,
            source_crop_labels=source_crop_labels,
            keep_classes=set(unique_classes),
            thumbnail_zoom=source_crop_zoom,
            alpha=source_crop_alpha,
            zorder=1,
        )

    # ----- focused-class-only distances for side bars
    focus_proto_before = proto_before_xy[pidx_focus].mean(axis=0)
    focus_proto_after = proto_after_xy[pidx_focus].mean(axis=0)
    focus_inst_xy = inst_xy[iidx_focus]

    d_before = np.linalg.norm(focus_inst_xy - focus_proto_before[None, :], axis=1)
    d_after = np.linalg.norm(focus_inst_xy - focus_proto_after[None, :], axis=1)

    order = np.argsort(d_before)[::-1]
    d_before = d_before[order]
    d_after = d_after[order]
    bar_labels = [str(i + 1) for i in range(len(order))]

    max_bar_val = float(max(d_before.max(), d_after.max())) if len(d_before) > 0 else 1.0
    max_bar_val *= 1.15

    inst_scatters = {}
    proto_before_sc = {}
    proto_move_sc = {}
    proto_after_sc = {}
    texts = {}
    traj_lines = {}
    thumb_by_class = {}

    for cls in unique_classes:
        cls_disp = fmt_label(cls)
        pidx = [i for i, c in enumerate(proto_labels) if c == cls]
        iidx = [i for i, c in enumerate(inst_labels) if c == cls]
        is_focus = (cls == focus_class)

        inst_pts = inst_xy[iidx]
        inst_scatters[cls] = ax.scatter(
            inst_pts[:, 0], inst_pts[:, 1],
            s=55,
            c=[color_map[cls]],
            alpha=active_instance_alpha if is_focus else 0.55,
            marker="o",
            edgecolors="none",
            zorder=2,
        )

        pb = proto_before_xy[pidx]
        pa = proto_after_xy[pidx]

        proto_before_sc[cls] = ax.scatter(
            pb[:, 0], pb[:, 1],
            s=220,
            c=[color_map[cls]],
            marker="X",
            alpha=0.20 if is_focus else 0.16,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

        proto_move_sc[cls] = ax.scatter(
            pb[:, 0], pb[:, 1],
            s=260,
            c=[color_map[cls]],
            marker="*",
            alpha=active_proto_alpha if is_focus else 0.40,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

        proto_after_sc[cls] = ax.scatter(
            pa[:, 0], pa[:, 1],
            s=240,
            c=[color_map[cls]],
            marker="*",
            alpha=0.0,
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

        cls_texts = []
        cls_lines = []
        for j, i_proto in enumerate(pidx):
            tx, ty = label_offset(pb[j, 0], pb[j, 1], xspan_global, yspan_global)
            t = ax.text(
                tx, ty,
                cls_disp,
                fontsize=14 if is_focus else 12,
                ha="left",
                va="bottom",
                alpha=0.95 if is_focus else 0.40,
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65, edgecolor="none"),
            )
            cls_texts.append(t)

            (line,) = ax.plot(
                [pb[j, 0], pb[j, 0]],
                [pb[j, 1], pb[j, 1]],
                linestyle="--",
                linewidth=1.8,
                color=color_map[cls],
                alpha=0.95 if is_focus else 0.25,
                zorder=4,
            )
            cls_lines.append(line)

        texts[cls] = cls_texts
        traj_lines[cls] = cls_lines

        if show_thumbnails and instance_image_paths is not None and len(iidx) > 0:
            sub_xy = inst_xy[iidx]
            sub_labels = [inst_labels[i] for i in iidx]
            sub_paths = [instance_image_paths[i] for i in iidx]
            selected_local = select_thumbnail_indices(
                sub_xy, sub_labels, max_per_class=min(thumbs_per_class, len(iidx))
            )
            cls_thumb_artists = []
            for local_i in selected_local:
                artist = add_thumbnail(
                    ax,
                    sub_xy[local_i],
                    sub_paths[local_i],
                    zoom=thumbnail_zoom,
                    border_color="red",
                    border_width=3,
                )
                if artist is not None:
                    artist.set_alpha(0.0)  # start invisible
                    cls_thumb_artists.append(artist)
            thumb_by_class[cls] = cls_thumb_artists
        else:
            thumb_by_class[cls] = []

    subtitle = ax.text(
        0.02, 0.98, "All categories",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=16,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        zorder=10,
    )

    if show_thumbnails:
        marker_guide = [
            plt.Line2D([0], [0], marker='X', linestyle='', label='Prototype before',
                       markersize=10, markerfacecolor='gray', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='*', linestyle='', label='Moving / after prototype',
                       markersize=12, markerfacecolor='gray', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='s', linestyle='', label='Target images (red border)',
                       markersize=8, markerfacecolor='none', markeredgecolor='red', markeredgewidth=2),
            plt.Line2D([0], [0], marker='s', linestyle='', label='Source images (blue border)',
                       markersize=8, markerfacecolor='none', markeredgecolor='blue', markeredgewidth=2),
        ]
    else:
        marker_guide = [
            plt.Line2D([0], [0], marker='o', linestyle='', label='Target instances',
                       markersize=8, markerfacecolor='gray', markeredgecolor='none'),
            plt.Line2D([0], [0], marker='X', linestyle='', label='Prototype before',
                       markersize=10, markerfacecolor='gray', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='*', linestyle='', label='Moving / after prototype',
                       markersize=12, markerfacecolor='gray', markeredgecolor='black'),
        ]
    legend = ax.legend(
        handles=marker_guide,
        loc='upper left',
        bbox_to_anchor=(0, -0.05),
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10
    )

    # ----- side bar chart
    if show_distance_bars:
        y_pos = np.arange(len(bar_labels))
        bars = ax_bar.barh(
            y_pos,
            d_before,
            color=color_map[focus_class],
            alpha=distance_bar_alpha,
            edgecolor="black",
            linewidth=0.6,
        )
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(bar_labels, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max_bar_val)
        ax_bar.set_xlabel("Distance")
        ax_bar.set_title(f"Distances to {focus_class_disp} prototype", fontsize=15)
        ax_bar.grid(True, axis="x", alpha=0.25)
        ax_bar.set_visible(False)

        avg_before = float(d_before.mean()) if len(d_before) > 0 else 0.0
        avg_after = float(d_after.mean()) if len(d_after) > 0 else 0.0

        vline_before = ax_bar.axvline(avg_before, linestyle="--", linewidth=1.4, alpha=0.85, color="gray")
        vline_after = ax_bar.axvline(avg_before, linestyle=":", linewidth=1.8, alpha=0.0, color="black")

        avg_text_before = ax_bar.text(
            avg_before - 0.04 * max_bar_val,
            -0.45,
            f"Avg before = {avg_before:.3f}",
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        avg_text_after = ax_bar.text(
            avg_before - 0.04 * max_bar_val,
            -0.9,
            f"Avg after = {avg_after:.3f}",
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
    else:
        bars = None
        vline_before = None
        vline_after = None
        avg_text_before = None
        avg_text_after = None

    whole_move_frames = fps * whole_move_seconds
    zoom_frames = fps * zoom_seconds
    focus_move_frames = fps * focus_move_seconds
    hold_frames = fps * final_hold_seconds
    total_frames = whole_move_frames + zoom_frames + focus_move_frames + hold_frames

    def lerp(a, b, t):
        return (1 - t) * a + t * b

    def set_visibility(mode, show_source_thumbs=True, show_target_thumbs=False):
        for cls in unique_classes:
            is_focus = (cls == focus_class)

            if mode == "whole_move":
                inst_a = 0.60 if not is_focus else active_instance_alpha
                proto_a = 0.45 if not is_focus else active_proto_alpha
                text_a = 0.45 if not is_focus else 0.95
                line_a = 0.25 if not is_focus else 0.95
            elif mode == "zoom":
                inst_a = inactive_instance_alpha if not is_focus else active_instance_alpha
                proto_a = inactive_proto_alpha if not is_focus else active_proto_alpha
                text_a = inactive_proto_alpha if not is_focus else 0.95
                line_a = inactive_proto_alpha if not is_focus else 0.95
            elif mode == "focus_move":
                # Make unfocused categories equally visible during focus move
                inst_a = active_instance_alpha
                proto_a = active_proto_alpha
                text_a = 0.95
                line_a = 0.95
            else:
                inst_a = inactive_instance_alpha if not is_focus else active_instance_alpha
                proto_a = inactive_proto_alpha if not is_focus else active_proto_alpha
                text_a = inactive_proto_alpha if not is_focus else 0.95
                line_a = inactive_proto_alpha if not is_focus else 0.95

            inst_scatters[cls].set_alpha(inst_a)
            proto_before_sc[cls].set_alpha(0.18 if is_focus else proto_a * 0.5)
            proto_move_sc[cls].set_alpha(proto_a)
            proto_after_sc[cls].set_alpha(0.16 if (mode == "final" and is_focus) else 0.0)

            for t in texts[cls]:
                t.set_alpha(text_a)
            for line in traj_lines[cls]:
                line.set_alpha(line_a)

        # Control thumbnail visibility (skip during whole_move - handled manually for fading)
        if mode != "whole_move":
            # Control source thumbnail visibility
            for artist in source_bg_artists:
                artist.set_alpha(source_crop_alpha if show_source_thumbs else 0.0)

            # Control target thumbnail visibility
            if not show_target_thumbs:
                for cls in unique_classes:
                    for thumb in thumb_by_class[cls]:
                        thumb.set_alpha(0.0)

        if show_distance_bars:
            if mode in ["whole_move", "zoom"]:
                ax_bar.set_visible(False)
            else:
                ax_bar.set_visible(True)
                for bar in bars:
                    bar.set_alpha(distance_bar_alpha)

    def update_positions(current_proto_xy, local_xspan, local_yspan):
        for cls in unique_classes:
            pidx = [i for i, c in enumerate(proto_labels) if c == cls]
            current_xy = current_proto_xy[cls]
            proto_move_sc[cls].set_offsets(current_xy)

            for j, i_proto in enumerate(pidx):
                tx, ty = label_offset(current_xy[j, 0], current_xy[j, 1], local_xspan, local_yspan)
                texts[cls][j].set_position((tx, ty))
                traj_lines[cls][j].set_data(
                    [proto_before_xy[i_proto, 0], current_xy[j, 0]],
                    [proto_before_xy[i_proto, 1], current_xy[j, 1]],
                )

    def update_bars(alpha, visible):
        if not show_distance_bars:
            return

        ax_bar.set_visible(visible)
        if not visible:
            return

        current_d = (1 - alpha) * d_before + alpha * d_after

        for rect, val in zip(bars, current_d):
            rect.set_width(float(val))

        current_avg = float(np.mean(current_d)) if len(current_d) > 0 else 0.0
        vline_before.set_xdata([avg_before, avg_before])
        vline_after.set_xdata([current_avg, current_avg])

        avg_text_before.set_position((avg_before - 0.04 * max_bar_val, -0.45))
        avg_text_after.set_position((current_avg - 0.04 * max_bar_val, -0.9))
        avg_text_after.set_text(f"Avg after = {current_avg:.3f}")

        if alpha <= 0.0:
            avg_text_after.set_alpha(0.0)
            vline_after.set_alpha(0.0)
        else:
            avg_text_after.set_alpha(0.9)
            vline_after.set_alpha(0.85)

    def update(frame):
        if frame < whole_move_frames:
            t = frame / max(1, whole_move_frames - 1)
            subtitle.set_visible(True)
            subtitle.set_text("All categories")
            legend.set_visible(True)
            ax.set_title("" if title is None else fmt_label(title), fontsize=18)
            ax.title.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.95))
            ax.title.set_zorder(1000)

            # Don't use show_source_thumbs/show_target_thumbs here - we handle it manually
            set_visibility("whole_move")
            ax.set_xlim(gxmin, gxmax)
            ax.set_ylim(gymin, gymax)

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                current_proto_xy[cls] = (1 - t) * proto_before_xy[pidx] + t * proto_after_xy[pidx]
            update_positions(current_proto_xy, gxmax - gxmin, gymax - gymin)

            # Fade out source thumbnails and fade in target thumbnails during stage 1
            fade_transition = min(1.0, t * 1.5)  # Faster fade at beginning

            # Fade out source thumbnails (start visible, fade to invisible)
            for artist in source_bg_artists:
                artist.set_alpha(source_crop_alpha * (1.0 - fade_transition))

            # Fade in target thumbnails (start invisible, fade to visible)
            thumb_alpha_stage1 = fade_transition
            for cls in unique_classes:
                is_focus = (cls == focus_class)
                for thumb in thumb_by_class[cls]:
                    thumb.set_alpha(thumb_alpha_stage1 if is_focus else 0.22 * thumb_alpha_stage1)

            update_bars(alpha=0.0, visible=False)

        elif frame < whole_move_frames + zoom_frames:
            t = (frame - whole_move_frames) / max(1, zoom_frames - 1)

            subtitle.set_visible(False)
            legend.set_visible(False)
            ax.set_title("")

            set_visibility("zoom", show_source_thumbs=False, show_target_thumbs=True)
            ax.set_xlim(lerp(gxmin, fxmin, t), lerp(gxmax, fxmax, t))
            ax.set_ylim(lerp(gymin, fymin, t), lerp(gymax, fymax, t))

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                # Keep all prototypes at before position during zoom
                current_proto_xy[cls] = proto_before_xy[pidx]

            local_xspan = lerp(gxmax - gxmin, fxmax - fxmin, t)
            local_yspan = lerp(gymax - gymin, fymax - fymin, t)
            update_positions(current_proto_xy, local_xspan, local_yspan)

            # Keep source thumbnails hidden
            for artist in source_bg_artists:
                artist.set_alpha(0.0)

            # Show target thumbnails
            for cls in unique_classes:
                is_focus = (cls == focus_class)
                for thumb in thumb_by_class[cls]:
                    thumb.set_alpha(inactive_instance_alpha if not is_focus else 1.0)

            update_bars(alpha=0.0, visible=False)

        elif frame < whole_move_frames + zoom_frames + focus_move_frames:
            t = (frame - whole_move_frames - zoom_frames) / max(1, focus_move_frames - 1)

            subtitle.set_visible(True)
            subtitle.set_text(f"Focus on {focus_class_disp}")
            legend.set_visible(True)
            ax.set_title(f"{focus_class_disp}: local prototype motion", fontsize=18)
            ax.title.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.95))
            ax.title.set_zorder(1000)

            set_visibility("focus_move", show_source_thumbs=False, show_target_thumbs=True)
            ax.set_xlim(fxmin, fxmax)
            ax.set_ylim(fymin, fymax)

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                # All prototypes move from before to after during focus phase
                current_proto_xy[cls] = (1 - t) * proto_before_xy[pidx] + t * proto_after_xy[pidx]
            update_positions(current_proto_xy, fxmax - fxmin, fymax - fymin)

            # Keep source thumbnails hidden
            for artist in source_bg_artists:
                artist.set_alpha(0.0)

            # Show target thumbnails
            for cls in unique_classes:
                is_focus = (cls == focus_class)
                for thumb in thumb_by_class[cls]:
                    thumb.set_alpha(inactive_instance_alpha if not is_focus else 1.0)

            update_bars(alpha=t, visible=True)

        else:
            subtitle.set_visible(True)
            subtitle.set_text(f"{focus_class_disp} after mapping")
            legend.set_visible(True)
            ax.set_title(f"{focus_class_disp}: local prototype motion", fontsize=18)
            ax.title.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.95))
            ax.title.set_zorder(1000)

            set_visibility("final", show_source_thumbs=False, show_target_thumbs=True)
            ax.set_xlim(fxmin, fxmax)
            ax.set_ylim(fymin, fymax)

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                current_proto_xy[cls] = proto_after_xy[pidx]
            update_positions(current_proto_xy, fxmax - fxmin, fymax - fymin)

            # Keep source thumbnails hidden
            for artist in source_bg_artists:
                artist.set_alpha(0.0)

            # Show target thumbnails
            for cls in unique_classes:
                is_focus = (cls == focus_class)
                for thumb in thumb_by_class[cls]:
                    thumb.set_alpha(inactive_instance_alpha if not is_focus else 1.0)

            update_bars(alpha=1.0, visible=True)

        artists = [subtitle]
        if legend is not None:
            artists.append(legend)
        artists.extend(source_bg_artists)

        for cls in unique_classes:
            artists.extend([inst_scatters[cls], proto_before_sc[cls], proto_move_sc[cls], proto_after_sc[cls]])
            artists.extend(texts[cls])
            artists.extend(traj_lines[cls])
            artists.extend(thumb_by_class[cls])

        if show_distance_bars:
            artists.append(ax_bar)
            artists.append(avg_text_before)
            artists.append(avg_text_after)
            artists.append(vline_before)
            artists.append(vline_after)
            artists.extend(list(bars))

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--proto_before", type=str, required=True)
    parser.add_argument("--proto_after", type=str, required=True)
    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument("--proto_labels", type=str, required=True)
    parser.add_argument("--instance_labels", type=str, required=True)

    parser.add_argument("--source_instances", type=str, default=None,
                        help="Source instances .npy file")
    parser.add_argument("--source_instance_labels", type=str, default=None,
                        help="Source instance labels text file")
    parser.add_argument("--source_instance_paths", type=str, default=None,
                        help="Source instance image paths text file")
    parser.add_argument("--source_crop_zoom", type=float, default=0.5)
    parser.add_argument("--source_crop_alpha", type=float, default=0.18)

    parser.add_argument("--method", type=str, default="pca", choices=["pca", "mds"])
    parser.add_argument("--l2_normalize", action="store_true")

    parser.add_argument("--keep_classes", type=str, default=None,
                        help='Comma-separated class names to keep, e.g. "airplane,ship,swimming_pool"')

    parser.add_argument("--instance_image_paths", type=str, default=None,
                        help="Text file with one image path per line")

    parser.add_argument("--out_png", type=str, default="projection_plot.png")
    parser.add_argument("--out_csv", type=str, default="projection_coords.csv")
    parser.add_argument("--out_dist_csv", type=str, default="class_avg_distances.csv")

    parser.add_argument("--focus_zoom_animation", type=str, default=None,
                        help="Output path for focus zoom animation (.mp4 or .gif)")
    parser.add_argument("--focus_class", type=str, default=None,
                        help="Class name to focus on for focus zoom animation")

    parser.add_argument("--show_thumbnails", action="store_true")
    parser.add_argument("--thumbs_per_class", type=int, default=1)
    parser.add_argument("--thumbnail_zoom", type=float, default=0.80)

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=8,
                        help="Duration for standard / per-class animation")
    parser.add_argument("--seconds_per_class", type=int, default=3,
                        help="Seconds per class for staged animation")
    parser.add_argument("--final_seconds", type=int, default=3,
                        help="Final all-classes-together stage duration")

    parser.add_argument("--whole_move_seconds", type=int, default=3,
                        help="Whole move seconds for focus zoom animation (all categories move)")
    parser.add_argument("--zoom_seconds", type=int, default=2,
                        help="Zoom seconds for focus zoom animation")
    parser.add_argument("--focus_move_seconds", type=int, default=3,
                        help="Focus move seconds for focus zoom animation (focused category moves)")
    parser.add_argument("--final_hold_seconds", type=int, default=2,
                        help="Final hold seconds for focus zoom animation")
    parser.add_argument("--label_dx_ratio", type=float, default=0.012,
                        help="Label X offset ratio for focus zoom animation")
    parser.add_argument("--label_dy_ratio", type=float, default=0.025,
                        help="Label Y offset ratio for focus zoom animation")
    parser.add_argument("--show_distance_bars", action="store_true",
                        help="Show distance bars in focus zoom animation")
    parser.add_argument("--distance_bar_alpha", type=float, default=0.90,
                        help="Alpha transparency for distance bars in focus zoom animation")

    parser.add_argument("--inactive_instance_alpha", type=float, default=0.9)
    parser.add_argument("--inactive_proto_alpha", type=float, default=0.9)
    parser.add_argument("--active_instance_alpha", type=float, default=0.9)
    parser.add_argument("--active_proto_alpha", type=float, default=1.00)
    parser.add_argument("--final_instance_alpha", type=float, default=0.9)
    parser.add_argument("--final_proto_alpha", type=float, default=0.90)

    args = parser.parse_args()

    proto_before = np.load(args.proto_before, allow_pickle=True)
    proto_after = np.load(args.proto_after, allow_pickle=True)
    instances = np.load(args.instances, allow_pickle=True)

    proto_labels = load_labels(args.proto_labels)
    inst_labels = load_labels(args.instance_labels)
    instance_image_paths = read_txt(args.instance_image_paths)

    if proto_before.ndim != 2 or proto_after.ndim != 2 or instances.ndim != 2:
        raise ValueError("All feature files must be 2D arrays: [N, D]")

    if proto_before.shape != proto_after.shape:
        raise ValueError(
            f"Prototype before/after shapes differ: {proto_before.shape} vs {proto_after.shape}"
        )

    if len(proto_labels) != proto_before.shape[0]:
        raise ValueError(
            f"Number of prototype labels ({len(proto_labels)}) does not match "
            f"number of prototypes ({proto_before.shape[0]})"
        )

    if len(inst_labels) != instances.shape[0]:
        raise ValueError(
            f"Number of instance labels ({len(inst_labels)}) does not match "
            f"number of instances ({instances.shape[0]})"
        )

    ensure_same_dim(proto_before, proto_after, instances)

    if args.l2_normalize:
        proto_before = l2_normalize_np(proto_before)
        proto_after = l2_normalize_np(proto_after)
        instances = l2_normalize_np(instances)

    keep_classes = parse_keep_classes(args.keep_classes)
    proto_before, proto_after, proto_labels, instances, inst_labels, instance_image_paths = filter_by_classes(
        proto_before=proto_before,
        proto_after=proto_after,
        proto_labels=proto_labels,
        instances=instances,
        inst_labels=inst_labels,
        instance_image_paths=instance_image_paths,
        keep_classes=keep_classes,
        drop_proto_without_instances=True,
    )

    if len(proto_labels) == 0:
        raise ValueError("No prototype classes remain after filtering.")
    if len(inst_labels) == 0:
        raise ValueError("No instances remain after filtering.")

    proto_before_xy, proto_after_xy, inst_xy = split_projected_coords(
        proto_before, proto_after, instances, method=args.method
    )

    save_coords_csv(args.out_csv, proto_labels, proto_before_xy, proto_after_xy, inst_labels, inst_xy)

    class_avg_dists = compute_class_avg_distances(
        proto_before_xy=proto_before_xy,
        proto_after_xy=proto_after_xy,
        inst_xy=inst_xy,
        proto_labels=proto_labels,
        inst_labels=inst_labels,
    )
    save_class_avg_distances_csv(args.out_dist_csv, class_avg_dists)

    plot_projection(
        proto_labels=proto_labels,
        proto_before_xy=proto_before_xy,
        proto_after_xy=proto_after_xy,
        inst_labels=inst_labels,
        inst_xy=inst_xy,
        out_path=args.out_png,
        title=f"Shared 2D Projection ({args.method.upper()})",
    )
    print(f"Saved static plot to: {args.out_png}")
    print(f"Saved coordinates to: {args.out_csv}")
    print(f"Saved class average distances to: {args.out_dist_csv}")


    if args.focus_zoom_animation:
        if args.focus_class is None:
            raise ValueError("--focus_class is required when using --focus_zoom_animation")

        source_xy = None
        source_paths = None
        source_labels = None

        if args.source_instances and args.source_instance_labels:
            # Load source instances
            source_instances = np.load(args.source_instances, allow_pickle=True)
            source_labels = load_labels(args.source_instance_labels)
            source_paths = read_txt(args.source_instance_paths) if args.source_instance_paths else None

            if source_instances.ndim != 2:
                raise ValueError("Source instances must be 2D array: [N, D]")
            if len(source_labels) != source_instances.shape[0]:
                raise ValueError(
                    f"Number of source labels ({len(source_labels)}) does not match "
                    f"number of source instances ({source_instances.shape[0]})"
                )

            # Apply same normalization as target instances if needed
            if args.l2_normalize:
                source_instances = l2_normalize_np(source_instances)

            # Filter source instances by keep_classes if specified
            if keep_classes is not None:
                source_idx = [i for i, c in enumerate(source_labels) if c in keep_classes]
                source_instances = source_instances[source_idx]
                source_labels = [source_labels[i] for i in source_idx]
                if source_paths is not None:
                    source_paths = [source_paths[i] for i in source_idx]

            # Reproject everything including source instances
            proto_before_xy, proto_after_xy, inst_xy, source_xy = split_projected_coords_with_source(
                proto_before=proto_before,
                proto_after=proto_after,
                instances=instances,
                source_feats=source_instances,
                method=args.method,
            )

        make_focus_zoom_animation(
            proto_before_xy=proto_before_xy,
            proto_after_xy=proto_after_xy,
            inst_xy=inst_xy,
            proto_labels=proto_labels,
            inst_labels=inst_labels,
            focus_class=args.focus_class,
            out_path=args.focus_zoom_animation,
            instance_image_paths=instance_image_paths,
            show_thumbnails=args.show_thumbnails,
            thumbs_per_class=args.thumbs_per_class,
            thumbnail_zoom=args.thumbnail_zoom,
            fps=args.fps,
            whole_move_seconds=args.whole_move_seconds,
            zoom_seconds=args.zoom_seconds,
            focus_move_seconds=args.focus_move_seconds,
            final_hold_seconds=args.final_hold_seconds,
            title=f"Prototype motion before and after mapping",
            inactive_instance_alpha=args.inactive_instance_alpha,
            inactive_proto_alpha=args.inactive_proto_alpha,
            active_instance_alpha=args.active_instance_alpha,
            active_proto_alpha=args.active_proto_alpha,
            label_dx_ratio=args.label_dx_ratio,
            label_dy_ratio=args.label_dy_ratio,
            show_distance_bars=args.show_distance_bars,
            distance_bar_alpha=args.distance_bar_alpha,
            source_crop_xy=source_xy,
            source_crop_paths=source_paths,
            source_crop_labels=source_labels,
            source_crop_zoom=args.source_crop_zoom,
            source_crop_alpha=args.source_crop_alpha,
        )


if __name__ == "__main__":
    main()
