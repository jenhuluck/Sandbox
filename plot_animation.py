import os
import csv
import argparse
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


# -----------------------------
# Plot helpers
# -----------------------------
def class_color_map(class_names):
    unique_classes = list(dict.fromkeys(class_names))
    return {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}


def compute_axis_limits(proto_before_xy, proto_after_xy, inst_xy, pad_ratio=0.08):
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
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def add_thumbnail(ax, xy, img_path, zoom=0.30):
    if img_path is None or not os.path.exists(img_path):
        return None
    try:
        img = plt.imread(img_path)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, xy, frameon=False, pad=0.0)
        ax.add_artist(ab)
        return ab
    except Exception:
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


def add_thumbnails(ax, inst_xy, inst_labels, instance_image_paths, thumbs_per_class=1, thumbnail_zoom=0.30):
    artists = []
    if instance_image_paths is None:
        return artists
    selected_idx = select_thumbnail_indices(inst_xy, inst_labels, max_per_class=thumbs_per_class)
    for i in selected_idx:
        artist = add_thumbnail(ax, inst_xy[i], instance_image_paths[i], zoom=thumbnail_zoom)
        if artist is not None:
            artists.append(artist)
    return artists


def add_marker_legend(ax):
    marker_guide = [
        plt.Line2D([0], [0], marker='o', linestyle='', label='Target instances',
                   markersize=8, markerfacecolor='gray', markeredgecolor='none'),
        plt.Line2D([0], [0], marker='X', linestyle='', label='Prototype before',
                   markersize=10, markerfacecolor='gray', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='*', linestyle='', label='Prototype after / moving',
                   markersize=12, markerfacecolor='gray', markeredgecolor='black'),
    ]
    ax.legend(handles=marker_guide, loc="best")


# -----------------------------
# Static plot
# -----------------------------
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
            ax.text(xb, yb, f"{cls}\n(before)", fontsize=9, ha="right", va="bottom")
            ax.text(xa, ya, f"{cls}\n(after)", fontsize=9, ha="left", va="top")

    add_marker_legend(ax)
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


# -----------------------------
# Standard all-classes animation
# -----------------------------
def make_animation(
    proto_before_xy,
    proto_after_xy,
    inst_xy,
    proto_labels,
    inst_labels,
    out_path,
    instance_image_paths=None,
    show_thumbnails=False,
    thumbs_per_class=1,
    thumbnail_zoom=0.30,
    fps=20,
    seconds=8,
    title="Prototype Motion Before and After Mapping",
):
    colors = class_color_map(proto_labels)
    fig, ax = plt.subplots(figsize=(12, 10))

    setup_axes(ax, proto_before_xy, proto_after_xy, inst_xy, title)
    add_instance_scatter(ax, inst_xy, inst_labels, colors, alpha=0.65, size=50)

    if show_thumbnails and instance_image_paths is not None:
        add_thumbnails(
            ax,
            inst_xy,
            inst_labels,
            instance_image_paths,
            thumbs_per_class=thumbs_per_class,
            thumbnail_zoom=thumbnail_zoom,
        )

    before_static = ax.scatter(
        proto_before_xy[:, 0],
        proto_before_xy[:, 1],
        s=200,
        c=[colors[c] for c in proto_labels],
        marker="X",
        alpha=0.20,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    proto_scatter = ax.scatter(
        proto_before_xy[:, 0],
        proto_before_xy[:, 1],
        s=260,
        c=[colors[c] for c in proto_labels],
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    after_static = ax.scatter(
        proto_after_xy[:, 0],
        proto_after_xy[:, 1],
        s=240,
        c=[colors[c] for c in proto_labels],
        marker="*",
        alpha=0.0,
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )

    texts = []
    traj_lines = []
    for i, cls in enumerate(proto_labels):
        t = ax.text(
            proto_before_xy[i, 0],
            proto_before_xy[i, 1],
            cls,
            fontsize=9,
            ha="left",
            va="bottom",
            zorder=6,
        )
        texts.append(t)

        (line,) = ax.plot(
            [proto_before_xy[i, 0], proto_before_xy[i, 0]],
            [proto_before_xy[i, 1], proto_before_xy[i, 1]],
            linestyle="--",
            linewidth=1.8,
            color=colors[cls],
            alpha=0.9,
            zorder=4,
        )
        traj_lines.append(line)

    total_frames = fps * seconds
    hold_start = max(1, int(0.18 * total_frames))
    move_end = max(hold_start + 1, int(0.78 * total_frames))

    subtitle = ax.text(
        0.02, 0.98, "Before mapping",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
    )

    add_marker_legend(ax)

    def interpolate(alpha):
        return (1 - alpha) * proto_before_xy + alpha * proto_after_xy

    def update(frame):
        if frame <= hold_start:
            alpha = 0.0
            subtitle.set_text("Before mapping")
            after_static.set_alpha(0.0)
        elif frame <= move_end:
            alpha = (frame - hold_start) / max(1, (move_end - hold_start))
            subtitle.set_text("Prototype projection in progress")
            after_static.set_alpha(0.0)
        else:
            alpha = 1.0
            subtitle.set_text("After mapping")
            after_static.set_alpha(0.18)

        current_xy = interpolate(alpha)
        proto_scatter.set_offsets(current_xy)

        for i, txt in enumerate(texts):
            txt.set_position((current_xy[i, 0], current_xy[i, 1]))

        for i, line in enumerate(traj_lines):
            line.set_data(
                [proto_before_xy[i, 0], current_xy[i, 0]],
                [proto_before_xy[i, 1], current_xy[i, 1]],
            )

        return [before_static, proto_scatter, after_static, subtitle, *texts, *traj_lines]

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
# Per-class animations
# -----------------------------
def make_per_class_animations(
    proto_before_xy,
    proto_after_xy,
    inst_xy,
    proto_labels,
    inst_labels,
    out_dir,
    instance_image_paths=None,
    show_thumbnails=False,
    thumbs_per_class=1,
    thumbnail_zoom=0.30,
    fps=20,
    seconds=8,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    classes = [c for c in dict.fromkeys(proto_labels) if c in set(inst_labels)]

    for cls in classes:
        proto_idx = [i for i, c in enumerate(proto_labels) if c == cls]
        inst_idx = [i for i, c in enumerate(inst_labels) if c == cls]

        if not proto_idx or not inst_idx:
            continue

        pb = proto_before_xy[proto_idx]
        pa = proto_after_xy[proto_idx]
        ix = inst_xy[inst_idx]
        pl = [proto_labels[i] for i in proto_idx]
        il = [inst_labels[i] for i in inst_idx]
        ip = [instance_image_paths[i] for i in inst_idx] if instance_image_paths is not None else None

        out_path = os.path.join(out_dir, f"{cls}_animation.mp4")
        make_animation(
            proto_before_xy=pb,
            proto_after_xy=pa,
            inst_xy=ix,
            proto_labels=pl,
            inst_labels=il,
            out_path=out_path,
            instance_image_paths=ip,
            show_thumbnails=show_thumbnails,
            thumbs_per_class=thumbs_per_class,
            thumbnail_zoom=thumbnail_zoom,
            fps=fps,
            seconds=seconds,
            title=f"{cls}: Prototype Motion Before and After Mapping",
        )


# -----------------------------
# Staged category-by-category animation
# -----------------------------
def make_staged_animation(
    proto_before_xy,
    proto_after_xy,
    inst_xy,
    proto_labels,
    inst_labels,
    out_path,
    instance_image_paths=None,
    show_thumbnails=False,
    thumbs_per_class=1,
    thumbnail_zoom=0.30,
    fps=20,
    seconds_per_class=3,
    final_seconds=3,
    title="Prototype Motion by Category",
):
    unique_classes = [c for c in dict.fromkeys(proto_labels) if c in set(inst_labels)]
    colors = class_color_map(unique_classes)

    fig, ax = plt.subplots(figsize=(12, 10))
    setup_axes(ax, proto_before_xy, proto_after_xy, inst_xy, title)

    inst_scatters = {}
    proto_before_sc = {}
    proto_move_sc = {}
    proto_after_sc = {}
    texts = {}
    traj_lines = {}
    thumb_by_class = {}

    for cls in unique_classes:
        pidx = [i for i, c in enumerate(proto_labels) if c == cls]
        iidx = [i for i, c in enumerate(inst_labels) if c == cls]

        inst_pts = inst_xy[iidx]
        inst_scatters[cls] = ax.scatter(
            inst_pts[:, 0], inst_pts[:, 1],
            s=55, c=[colors[cls]], alpha=0.12, marker="o", edgecolors="none", zorder=2
        )

        if show_thumbnails and instance_image_paths is not None and len(iidx) > 0:
            sub_xy = inst_xy[iidx]
            sub_labels = [inst_labels[i] for i in iidx]
            sub_paths = [instance_image_paths[i] for i in iidx]
            selected_local = select_thumbnail_indices(sub_xy, sub_labels, max_per_class=min(thumbs_per_class, len(iidx)))
            cls_thumb_artists = []
            for local_i in selected_local:
                artist = add_thumbnail(ax, sub_xy[local_i], sub_paths[local_i], zoom=thumbnail_zoom)
                if artist is not None:
                    artist.set_alpha(0.18)
                    cls_thumb_artists.append(artist)
            thumb_by_class[cls] = cls_thumb_artists
        else:
            thumb_by_class[cls] = []

        pb = proto_before_xy[pidx]
        pa = proto_after_xy[pidx]

        proto_before_sc[cls] = ax.scatter(
            pb[:, 0], pb[:, 1],
            s=220, c=[colors[cls]], marker="X", alpha=0.08,
            edgecolors="black", linewidths=0.8, zorder=3
        )
        proto_move_sc[cls] = ax.scatter(
            pb[:, 0], pb[:, 1],
            s=260, c=[colors[cls]], marker="*", alpha=0.15,
            edgecolors="black", linewidths=1.0, zorder=5
        )
        proto_after_sc[cls] = ax.scatter(
            pa[:, 0], pa[:, 1],
            s=240, c=[colors[cls]], marker="*", alpha=0.0,
            edgecolors="black", linewidths=1.0, zorder=4
        )

        cls_texts = []
        cls_lines = []
        for j, i_proto in enumerate(pidx):
            t = ax.text(pb[j, 0], pb[j, 1], cls, fontsize=10, ha="left", va="bottom", alpha=0.12, zorder=6)
            cls_texts.append(t)

            (line,) = ax.plot(
                [pb[j, 0], pb[j, 0]],
                [pb[j, 1], pb[j, 1]],
                linestyle="--", linewidth=1.8, color=colors[cls], alpha=0.12, zorder=4
            )
            cls_lines.append(line)

        texts[cls] = cls_texts
        traj_lines[cls] = cls_lines

    subtitle = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
    )

    add_marker_legend(ax)

    frames_per_class = fps * seconds_per_class
    final_frames = fps * final_seconds
    total_frames = frames_per_class * len(unique_classes) + final_frames

    def set_class_visibility(cls, active=False, final=False):
        dim_inst = 0.10 if not final else 0.45
        act_inst = 0.80 if active else dim_inst

        dim_proto = 0.08 if not final else 0.22
        act_proto = 0.95 if active else dim_proto

        inst_scatters[cls].set_alpha(act_inst)
        proto_before_sc[cls].set_alpha(0.16 if active else (0.08 if not final else 0.12))
        proto_move_sc[cls].set_alpha(act_proto)
        proto_after_sc[cls].set_alpha(0.0 if active and not final else (0.14 if not final else 0.30))

        for t in texts[cls]:
            t.set_alpha(0.95 if active or final else 0.12)
        for line in traj_lines[cls]:
            line.set_alpha(0.95 if active or final else 0.12)
        for thumb in thumb_by_class[cls]:
            thumb.set_alpha(1.0 if active else (0.18 if not final else 0.55))

    def update(frame):
        if frame < frames_per_class * len(unique_classes):
            stage = frame // frames_per_class
            local = frame % frames_per_class
            active_cls = unique_classes[stage]
            subtitle.set_text(f"Category: {active_cls}")

            alpha = local / max(1, frames_per_class - 1)

            for cls in unique_classes:
                set_class_visibility(cls, active=(cls == active_cls), final=False)
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                if cls == active_cls:
                    current_xy = (1 - alpha) * proto_before_xy[pidx] + alpha * proto_after_xy[pidx]
                else:
                    current_xy = proto_before_xy[pidx]

                proto_move_sc[cls].set_offsets(current_xy)

                for j, i_proto in enumerate(pidx):
                    texts[cls][j].set_position((current_xy[j, 0], current_xy[j, 1]))
                    traj_lines[cls][j].set_data(
                        [proto_before_xy[i_proto, 0], current_xy[j, 0]],
                        [proto_before_xy[i_proto, 1], current_xy[j, 1]],
                    )
        else:
            subtitle.set_text("All categories together")
            for cls in unique_classes:
                set_class_visibility(cls, active=False, final=True)
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                current_xy = proto_after_xy[pidx]
                proto_move_sc[cls].set_offsets(current_xy)

                for j, i_proto in enumerate(pidx):
                    texts[cls][j].set_position((current_xy[j, 0], current_xy[j, 1]))
                    traj_lines[cls][j].set_data(
                        [proto_before_xy[i_proto, 0], current_xy[j, 0]],
                        [proto_before_xy[i_proto, 1], current_xy[j, 1]],
                    )

        artists = [subtitle]
        for cls in unique_classes:
            artists.extend([inst_scatters[cls], proto_before_sc[cls], proto_move_sc[cls], proto_after_sc[cls]])
            artists.extend(texts[cls])
            artists.extend(traj_lines[cls])
            artists.extend(thumb_by_class[cls])
        return artists

    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False, repeat=False
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

    parser.add_argument("--method", type=str, default="pca", choices=["pca", "mds"])
    parser.add_argument("--l2_normalize", action="store_true")

    parser.add_argument("--keep_classes", type=str, default=None,
                        help='Comma-separated class names to keep, e.g. "airplane,ship,swimming_pool"')

    parser.add_argument("--instance_image_paths", type=str, default=None,
                        help="Text file with one image path per line")

    parser.add_argument("--out_png", type=str, default="projection_plot.png")
    parser.add_argument("--out_csv", type=str, default="projection_coords.csv")
    parser.add_argument("--out_animation", type=str, default=None,
                        help="Standard all-classes animation (.mp4 or .gif)")
    parser.add_argument("--per_class_dir", type=str, default=None,
                        help="Output directory for one animation per category")
    parser.add_argument("--staged_animation", type=str, default=None,
                        help="Output path for staged category-by-category animation (.mp4 or .gif)")

    parser.add_argument("--show_thumbnails", action="store_true")
    parser.add_argument("--thumbs_per_class", type=int, default=1)
    parser.add_argument("--thumbnail_zoom", type=float, default=0.30)

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=8,
                        help="Duration for standard / per-class animation")
    parser.add_argument("--seconds_per_class", type=int, default=3,
                        help="Seconds per class for staged animation")
    parser.add_argument("--final_seconds", type=int, default=3,
                        help="Final all-classes-together stage duration")

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

    if args.out_animation:
        make_animation(
            proto_before_xy=proto_before_xy,
            proto_after_xy=proto_after_xy,
            inst_xy=inst_xy,
            proto_labels=proto_labels,
            inst_labels=inst_labels,
            out_path=args.out_animation,
            instance_image_paths=instance_image_paths,
            show_thumbnails=args.show_thumbnails,
            thumbs_per_class=args.thumbs_per_class,
            thumbnail_zoom=args.thumbnail_zoom,
            fps=args.fps,
            seconds=args.seconds,
            title="Prototype Motion Before and After Mapping",
        )
        print(f"Saved animation to: {args.out_animation}")

    if args.per_class_dir:
        make_per_class_animations(
            proto_before_xy=proto_before_xy,
            proto_after_xy=proto_after_xy,
            inst_xy=inst_xy,
            proto_labels=proto_labels,
            inst_labels=inst_labels,
            out_dir=args.per_class_dir,
            instance_image_paths=instance_image_paths,
            show_thumbnails=args.show_thumbnails,
            thumbs_per_class=args.thumbs_per_class,
            thumbnail_zoom=args.thumbnail_zoom,
            fps=args.fps,
            seconds=args.seconds,
        )
        print(f"Saved per-class animations to: {args.per_class_dir}")

    if args.staged_animation:
        make_staged_animation(
            proto_before_xy=proto_before_xy,
            proto_after_xy=proto_after_xy,
            inst_xy=inst_xy,
            proto_labels=proto_labels,
            inst_labels=inst_labels,
            out_path=args.staged_animation,
            instance_image_paths=instance_image_paths,
            show_thumbnails=args.show_thumbnails,
            thumbs_per_class=args.thumbs_per_class,
            thumbnail_zoom=args.thumbnail_zoom,
            fps=args.fps,
            seconds_per_class=args.seconds_per_class,
            final_seconds=args.final_seconds,
            title="Prototype Motion by Category",
        )
        print(f"Saved staged animation to: {args.staged_animation}")


if __name__ == "__main__":
    main()