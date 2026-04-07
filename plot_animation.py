import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA


def load_labels(path):
    if path.endswith(".npy"):
        x = np.load(path, allow_pickle=True)
        return x.tolist() if isinstance(x, np.ndarray) else list(x)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def l2_normalize_np(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def maybe_project_to_2d(proto_before, proto_after, instances, method="pca"):
    # If already 2D, keep as-is
    if proto_before.shape[1] == 2 and proto_after.shape[1] == 2 and instances.shape[1] == 2:
        return proto_before, proto_after, instances

    all_feats = np.concatenate([proto_before, proto_after, instances], axis=0)

    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(all_feats)
    else:
        raise ValueError(f"Unsupported method: {method}")

    c = proto_before.shape[0]
    n = instances.shape[0]

    proto_before_xy = coords[:c]
    proto_after_xy = coords[c:2 * c]
    inst_xy = coords[2 * c:2 * c + n]
    return proto_before_xy, proto_after_xy, inst_xy


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto_before", type=str, required=True)
    parser.add_argument("--proto_after", type=str, required=True)
    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument("--proto_labels", type=str, required=True)
    parser.add_argument("--instance_labels", type=str, required=True)
    parser.add_argument("--out", type=str, default="prototype_motion.mp4")
    parser.add_argument("--project_method", type=str, default="pca")
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--instance_image_paths", type=str, default=None)
    parser.add_argument("--show_thumbnails", action="store_true")
    parser.add_argument("--thumbs_per_class", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=8)
    args = parser.parse_args()

    proto_before = np.load(args.proto_before, allow_pickle=True)
    proto_after = np.load(args.proto_after, allow_pickle=True)
    instances = np.load(args.instances, allow_pickle=True)

    proto_labels = load_labels(args.proto_labels)
    inst_labels = load_labels(args.instance_labels)

    if args.l2_normalize:
        proto_before = l2_normalize_np(proto_before)
        proto_after = l2_normalize_np(proto_after)
        instances = l2_normalize_np(instances)

    if proto_before.ndim != 2 or proto_after.ndim != 2 or instances.ndim != 2:
        raise ValueError("All input arrays must be 2D.")

    if proto_before.shape != proto_after.shape:
        raise ValueError(f"Prototype shape mismatch: {proto_before.shape} vs {proto_after.shape}")

    if len(proto_labels) != proto_before.shape[0]:
        raise ValueError("Number of prototype labels does not match number of prototypes.")

    if len(inst_labels) != instances.shape[0]:
        raise ValueError("Number of instance labels does not match number of instances.")

    if not (proto_before.shape[1] == proto_after.shape[1] == instances.shape[1]):
        # unless already 2D coordinates with all dims equal to 2
        if not (proto_before.shape[1] == proto_after.shape[1] == instances.shape[1] == 2):
            raise ValueError("Feature dimensions do not match.")

    proto_before_xy, proto_after_xy, inst_xy = maybe_project_to_2d(
        proto_before, proto_after, instances, method=args.project_method
    )

    instance_image_paths = None
    if args.instance_image_paths is not None:
        arr = np.load(args.instance_image_paths, allow_pickle=True)
        instance_image_paths = arr.tolist() if isinstance(arr, np.ndarray) else list(arr)

    make_animation(
        proto_before_xy=proto_before_xy,
        proto_after_xy=proto_after_xy,
        inst_xy=inst_xy,
        proto_labels=proto_labels,
        inst_labels=inst_labels,
        out_path=args.out,
        instance_image_paths=instance_image_paths,
        show_thumbnails=args.show_thumbnails,
        thumbs_per_class=args.thumbs_per_class,
        fps=args.fps,
        seconds=args.seconds,
    )

    print(f"Saved animation to: {args.out}")


if __name__ == "__main__":
    main()