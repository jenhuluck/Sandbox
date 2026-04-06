import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def load_labels(path):
    if path.endswith(".npy"):
        labels = np.load(path, allow_pickle=True)
        return labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
    else:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]


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
    parser.add_argument("--out_png", type=str, default="projection_plot.png")
    parser.add_argument("--out_csv", type=str, default="projection_coords.csv")
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

    plot_projection(
        proto_labels,
        proto_before_xy,
        proto_after_xy,
        inst_labels,
        inst_xy,
        args.out_png,
        title=f"Shared 2D Projection ({args.method.upper()})",
    )

    print(f"Saved plot to: {args.out_png}")
    print(f"Saved coordinates to: {args.out_csv}")


if __name__ == "__main__":
    main()