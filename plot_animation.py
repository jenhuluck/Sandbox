import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import shutil
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def load_labels(path):
    if path.endswith(".npy"):
        labels = np.load(path, allow_pickle=True)
        return labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
    else:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]


def read_txt(path):
    """
    Read text file with one path per line.
    Returns list of paths, or None if path is None or file doesn't exist.
    """
    if path is None:
        return None

    if not os.path.exists(path):
        print(f"Warning: Image paths file not found: {path}")
        return None

    with open(path, "r") as f:
        paths = [line.strip() for line in f]

    return paths


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


def save_coords_csv(path, proto_labels, proto_before_xy, proto_after_xy, inst_labels, inst_xy):
    import csv

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


def add_thumbnail(ax, xy, img_path, zoom=0.18):
    if not os.path.exists(img_path):
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
        if len(idx) == 0:
            continue
        pts = inst_xy[idx]
        center = pts.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(pts - center, axis=1)
        order = np.argsort(dist)
        chosen = [idx[j] for j in order[:max_per_class]]
        selected.extend(chosen)
    return sorted(selected)


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
    fps=20,
    seconds=8,
    title="Prototype Motion Before and After Mapping",
):
    unique_classes = list(dict.fromkeys(proto_labels))
    color_map = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(12, 10))

    # Axis limits with padding
    all_xy = np.concatenate([proto_before_xy, proto_after_xy, inst_xy], axis=0)
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    padx = 0.08 * max(1e-6, xmax - xmin)
    pady = 0.08 * max(1e-6, ymax - ymin)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Plot fixed instances
    for cls in unique_classes:
        idx = [i for i, x in enumerate(inst_labels) if x == cls]
        if not idx:
            continue
        pts = inst_xy[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=50,
            c=[color_map[cls]],
            alpha=0.65,
            marker="o",
            edgecolors="none",
            zorder=2,
        )

    # Optional thumbnails
    thumb_artists = []
    if show_thumbnails and instance_image_paths is not None:
        selected_idx = select_thumbnail_indices(inst_xy, inst_labels, max_per_class=thumbs_per_class)
        for i in selected_idx:
            artist = add_thumbnail(ax, inst_xy[i], instance_image_paths[i], zoom=0.16)
            if artist is not None:
                thumb_artists.append(artist)

    # Static "before" markers, faint
    before_static = ax.scatter(
        proto_before_xy[:, 0],
        proto_before_xy[:, 1],
        s=200,
        c=[color_map[c] for c in proto_labels],
        marker="X",
        alpha=0.20,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    # Moving prototype markers
    proto_scatter = ax.scatter(
        proto_before_xy[:, 0],
        proto_before_xy[:, 1],
        s=260,
        c=[color_map[c] for c in proto_labels],
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    # Final after markers, initially invisible
    after_static = ax.scatter(
        proto_after_xy[:, 0],
        proto_after_xy[:, 1],
        s=240,
        c=[color_map[c] for c in proto_labels],
        marker="*",
        alpha=0.0,
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )

    # Labels that move with prototypes
    texts = []
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

    # Trajectory lines
    traj_lines = []
    for i, cls in enumerate(proto_labels):
        (line,) = ax.plot(
            [proto_before_xy[i, 0], proto_before_xy[i, 0]],
            [proto_before_xy[i, 1], proto_before_xy[i, 1]],
            linestyle="--",
            linewidth=1.8,
            color=color_map[cls],
            alpha=0.9,
            zorder=4,
        )
        traj_lines.append(line)

    # Phase timing
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

    marker_guide = [
        plt.Line2D([0], [0], marker='o', linestyle='', label='Target instances',
                   markersize=8, markerfacecolor='gray', markeredgecolor='none'),
        plt.Line2D([0], [0], marker='X', linestyle='', label='Prototype before',
                   markersize=10, markerfacecolor='gray', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='*', linestyle='', label='Moving / after prototype',
                   markersize=12, markerfacecolor='gray', markeredgecolor='black'),
    ]
    ax.legend(handles=marker_guide, loc="best")

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

        # update labels
        for i, txt in enumerate(texts):
            txt.set_position((current_xy[i, 0], current_xy[i, 1]))

        # update trajectory lines
        for i, line in enumerate(traj_lines):
            line.set_data(
                [proto_before_xy[i, 0], current_xy[i, 0]],
                [proto_before_xy[i, 1], current_xy[i, 1]],
            )

        return [proto_scatter, after_static, subtitle, *texts, *traj_lines]

    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    if out_path.lower().endswith(".mp4"):
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer, dpi=200)
    elif out_path.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=140)
    else:
        raise ValueError("Output must end with .mp4 or .gif")

    plt.close(fig)

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
    unique_classes = list(dict.fromkeys(proto_labels))
    color_map = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(12, 10))

    # plot instances
    for cls in unique_classes:
        idx = [i for i, x in enumerate(inst_labels) if x == cls]
        if len(idx) == 0:
            continue
        pts = inst_xy[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=55,
            alpha=alpha_instances,
            c=[color_map[cls]],
            marker="o",
            label=f"{cls} instances",
            edgecolors="none",
        )

    # plot prototype before / after and arrows
    for i, cls in enumerate(proto_labels):
        c = color_map[cls]
        xb, yb = proto_before_xy[i]
        xa, ya = proto_after_xy[i]

        ax.scatter(xb, yb, s=220, c=[c], marker="X", edgecolors="black", linewidths=1.0)
        ax.scatter(xa, ya, s=260, c=[c], marker="*", edgecolors="black", linewidths=1.0)

        ax.annotate(
            "",
            xy=(xa, ya),
            xytext=(xb, yb),
            arrowprops=dict(arrowstyle="->", linestyle="--", lw=1.8, color=c),
        )

        if show_text:
            ax.text(xb, yb, f"{cls}\n(before)", fontsize=9, ha="right", va="bottom")
            ax.text(xa, ya, f"{cls}\n(after)", fontsize=9, ha="left", va="top")

    # cleaner legend: one entry per class for instances + marker guide
    marker_guide = [
        plt.Line2D([0], [0], marker='o', linestyle='', label='Instances',
                   markersize=8, markerfacecolor='gray', markeredgecolor='none'),
        plt.Line2D([0], [0], marker='X', linestyle='', label='Prototype Before',
                   markersize=10, markerfacecolor='gray', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='*', linestyle='', label='Prototype After',
                   markersize=12, markerfacecolor='gray', markeredgecolor='black'),
    ]
    ax.legend(handles=marker_guide, loc="best")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto_before", type=str, required=True)
    parser.add_argument("--proto_after", type=str, required=True)
    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument("--proto_labels", type=str, required=True)
    parser.add_argument("--instance_labels", type=str, required=True)
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "mds"])
    parser.add_argument("--out_png", type=str, default="projection_plot.png",
                        help="Output path for static plot (PNG)")
    parser.add_argument("--out_animation", type=str, default=None,
                        help="Output path for animation (MP4 or GIF). If not provided, animation is skipped.")
    parser.add_argument("--out_csv", type=str, default="projection_coords.csv")
    parser.add_argument("--instance_image_paths", type=str, default=None,
                        help="Path to text file containing instance image paths (one per line)")
    parser.add_argument("--show_thumbnails", action="store_true",
                        help="Show thumbnail images in animation")
    parser.add_argument("--thumbs_per_class", type=int, default=1,
                        help="Number of thumbnails to show per class")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second for animation")
    parser.add_argument("--seconds", type=int, default=8,
                        help="Duration of animation in seconds")
    args = parser.parse_args()

    proto_before = np.load(args.proto_before)
    proto_after = np.load(args.proto_after)
    instances = np.load(args.instances)

    proto_labels = load_labels(args.proto_labels)
    inst_labels = load_labels(args.instance_labels)

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

    n_proto = proto_before.shape[0]
    n_inst = instances.shape[0]

    all_feats = np.concatenate([proto_before, proto_after, instances], axis=0)
    coords = project_features(all_feats, method=args.method)

    proto_before_xy = coords[:n_proto]
    proto_after_xy = coords[n_proto:2 * n_proto]
    inst_xy = coords[2 * n_proto:2 * n_proto + n_inst]

    save_coords_csv(
        args.out_csv,
        proto_labels,
        proto_before_xy,
        proto_after_xy,
        inst_labels,
        inst_xy,
    )

    # Generate static plot
    plot_projection(
        proto_labels,
        proto_before_xy,
        proto_after_xy,
        inst_labels,
        inst_xy,
        args.out_png,
        title=f"Shared 2D Projection ({args.method.upper()})",
    )
    print(f"Saved static plot to: {args.out_png}")

    # Generate animation if requested
    if args.out_animation:
        instance_image_paths = read_txt(args.instance_image_paths)

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
            fps=args.fps,
            seconds=args.seconds,
        )
        print(f"Saved animation to: {args.out_animation}")

    print(f"Saved coordinates to: {args.out_csv}")


if __name__ == "__main__":
    main()
