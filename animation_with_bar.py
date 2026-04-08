def make_focus_zoom_animation_twice(
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
    thumbnail_zoom=0.30,
    fps=20,
    whole_move_seconds=3,
    zoom_seconds=2,
    focus_move_seconds=3,
    final_hold_seconds=2,
    title=None,
    inactive_instance_alpha=0.10,
    inactive_proto_alpha=0.08,
    active_instance_alpha=0.90,
    active_proto_alpha=1.00,
    label_dx_ratio=0.020,
    label_dy_ratio=0.040,
    show_distance_bars=True,
    distance_bar_alpha=0.90,
):
    def fmt_label(s):
        s = str(s).replace("_", " ").strip()
        return s[:1].upper() + s[1:] if s else s

    focus_class_disp = fmt_label(focus_class)

    unique_classes = [c for c in dict.fromkeys(proto_labels) if c in set(inst_labels)]
    if focus_class not in unique_classes:
        raise ValueError(f"focus_class '{focus_class}' not found in filtered classes: {unique_classes}")

    color_map = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}

    # -----------------------------
    # layout: main plot + optional side bars
    # -----------------------------
    if show_distance_bars:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.4], wspace=0.18)
        ax = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax_bar = None

    # ----- global limits
    all_xy = np.concatenate([proto_before_xy, proto_after_xy, inst_xy], axis=0)
    gxmin, gymin = all_xy.min(axis=0)
    gxmax, gymax = all_xy.max(axis=0)
    gpadx = 0.08 * max(1e-6, gxmax - gxmin)
    gpady = 0.08 * max(1e-6, gymax - gymin)
    gxmin, gxmax = gxmin - gpadx, gxmax + gpadx
    gymin, gymax = gymin - gpady, gymax + gpady

    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymin, gymax)
    ax.set_xlabel("Feature space 1")
    ax.set_ylabel("Feature space 2")
    ax.set_title("" if title is None else fmt_label(title))
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

    # label offset in data units
    xspan_global = gxmax - gxmin
    yspan_global = gymax - gymin

    def label_offset(x, y, xspan, yspan):
        return x + label_dx_ratio * xspan, y + label_dy_ratio * yspan

    # -----------------------------
    # focused-class-only distances for side bars
    # -----------------------------
    focus_proto_before = proto_before_xy[pidx_focus].mean(axis=0)
    focus_proto_after = proto_after_xy[pidx_focus].mean(axis=0)
    focus_inst_xy = inst_xy[iidx_focus]

    d_before = np.linalg.norm(focus_inst_xy - focus_proto_before[None, :], axis=1)
    d_after = np.linalg.norm(focus_inst_xy - focus_proto_after[None, :], axis=1)

    # stable ordering
    order = np.argsort(d_before)[::-1]
    focus_inst_xy = focus_inst_xy[order]
    d_before = d_before[order]
    d_after = d_after[order]

    # simple labels: 1, 2, 3, ...
    bar_labels = [str(i + 1) for i in range(len(order))]

    max_bar_val = float(max(d_before.max(), d_after.max())) if len(d_before) > 0 else 1.0
    max_bar_val *= 1.15

    # -----------------------------
    # artists
    # -----------------------------
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
                fontsize=11 if is_focus else 10,
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
                artist = add_thumbnail(ax, sub_xy[local_i], sub_paths[local_i], zoom=thumbnail_zoom)
                if artist is not None:
                    artist.set_alpha(1.0 if is_focus else 0.22)
                    cls_thumb_artists.append(artist)
            thumb_by_class[cls] = cls_thumb_artists
        else:
            thumb_by_class[cls] = []

    subtitle = ax.text(
        0.02, 0.98, "All categories",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        zorder=10,
    )

    # legend: remove target instances if thumbnails are being used
    if show_thumbnails:
        marker_guide = [
            plt.Line2D([0], [0], marker='X', linestyle='', label='Prototype before',
                       markersize=10, markerfacecolor='gray', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='*', linestyle='', label='Moving / after prototype',
                       markersize=12, markerfacecolor='gray', markeredgecolor='black'),
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
    legend = ax.legend(handles=marker_guide, loc="best")

    # -----------------------------
    # side bar chart: focus class only
    # -----------------------------
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
        ax_bar.set_xlabel(f"Distance to {focus_class_disp} prototype")
        ax_bar.set_title(f"{focus_class_disp} instances only", fontsize=11)
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
            0.25,
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

    def set_visibility(mode):
        for cls in unique_classes:
            is_focus = (cls == focus_class)

            if mode == "whole_move":
                inst_a = 0.60 if not is_focus else active_instance_alpha
                proto_a = 0.45 if not is_focus else active_proto_alpha
                text_a = 0.45 if not is_focus else 0.95
                line_a = 0.25 if not is_focus else 0.95
                thumb_a = 0.22 if not is_focus else 1.0
            elif mode == "zoom":
                inst_a = inactive_instance_alpha if not is_focus else active_instance_alpha
                proto_a = inactive_proto_alpha if not is_focus else active_proto_alpha
                text_a = inactive_proto_alpha if not is_focus else 0.95
                line_a = inactive_proto_alpha if not is_focus else 0.95
                thumb_a = inactive_instance_alpha if not is_focus else 1.0
            elif mode == "focus_move":
                inst_a = inactive_instance_alpha if not is_focus else active_instance_alpha
                proto_a = inactive_proto_alpha if not is_focus else active_proto_alpha
                text_a = inactive_proto_alpha if not is_focus else 0.95
                line_a = inactive_proto_alpha if not is_focus else 0.95
                thumb_a = inactive_instance_alpha if not is_focus else 1.0
            else:
                inst_a = inactive_instance_alpha if not is_focus else active_instance_alpha
                proto_a = inactive_proto_alpha if not is_focus else active_proto_alpha
                text_a = inactive_proto_alpha if not is_focus else 0.95
                line_a = inactive_proto_alpha if not is_focus else 0.95
                thumb_a = inactive_instance_alpha if not is_focus else 1.0

            inst_scatters[cls].set_alpha(inst_a)
            proto_before_sc[cls].set_alpha(0.18 if is_focus else proto_a * 0.5)
            proto_move_sc[cls].set_alpha(proto_a)
            proto_after_sc[cls].set_alpha(0.16 if (mode == "final" and is_focus) else 0.0)

            for t in texts[cls]:
                t.set_alpha(text_a)
            for line in traj_lines[cls]:
                line.set_alpha(line_a)
            for thumb in thumb_by_class[cls]:
                thumb.set_alpha(thumb_a)

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
        avg_text_after.set_position((current_avg - 0.04 * max_bar_val, 0.25))
        avg_text_after.set_text(f"Avg after = {current_avg:.3f}")

        if alpha <= 0.0:
            avg_text_after.set_alpha(0.0)
            vline_after.set_alpha(0.0)
        else:
            avg_text_after.set_alpha(0.9)
            vline_after.set_alpha(0.85)

    def update(frame):
        # stage 1: whole figure, all prototypes move once
        if frame < whole_move_frames:
            t = frame / max(1, whole_move_frames - 1)
            subtitle.set_visible(True)
            subtitle.set_text("All categories")
            legend.set_visible(True)
            ax.set_title("" if title is None else fmt_label(title))

            set_visibility("whole_move")
            ax.set_xlim(gxmin, gxmax)
            ax.set_ylim(gymin, gymax)

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                current_proto_xy[cls] = (1 - t) * proto_before_xy[pidx] + t * proto_after_xy[pidx]
            update_positions(current_proto_xy, gxmax - gxmin, gymax - gymin)

            update_bars(alpha=0.0, visible=False)

        # stage 2: zoom transition, hide subtitle/legend/title, reset focus prototype to before
        elif frame < whole_move_frames + zoom_frames:
            t = (frame - whole_move_frames) / max(1, zoom_frames - 1)

            subtitle.set_visible(False)
            legend.set_visible(False)
            ax.set_title("")

            set_visibility("zoom")
            ax.set_xlim(lerp(gxmin, fxmin, t), lerp(gxmax, fxmax, t))
            ax.set_ylim(lerp(gymin, fymin, t), lerp(gymax, fymax, t))

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                if cls == focus_class:
                    current_proto_xy[cls] = proto_before_xy[pidx]
                else:
                    current_proto_xy[cls] = proto_after_xy[pidx]

            local_xspan = lerp(gxmax - gxmin, fxmax - fxmin, t)
            local_yspan = lerp(gymax - gymin, fymax - fymin, t)
            update_positions(current_proto_xy, local_xspan, local_yspan)

            update_bars(alpha=0.0, visible=False)

        # stage 3: zoomed-in, focused class moves again
        elif frame < whole_move_frames + zoom_frames + focus_move_frames:
            t = (frame - whole_move_frames - zoom_frames) / max(1, focus_move_frames - 1)

            subtitle.set_visible(True)
            subtitle.set_text(f"Focus on {focus_class_disp}")
            legend.set_visible(True)
            ax.set_title(f"{focus_class_disp}: local prototype motion")

            set_visibility("focus_move")
            ax.set_xlim(fxmin, fxmax)
            ax.set_ylim(fymin, fymax)

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                if cls == focus_class:
                    current_proto_xy[cls] = (1 - t) * proto_before_xy[pidx] + t * proto_after_xy[pidx]
                else:
                    current_proto_xy[cls] = proto_after_xy[pidx]
            update_positions(current_proto_xy, fxmax - fxmin, fymax - fymin)

            update_bars(alpha=t, visible=True)

        # stage 4: hold final zoomed view
        else:
            subtitle.set_visible(True)
            subtitle.set_text(f"{focus_class_disp} after mapping")
            legend.set_visible(True)
            ax.set_title(f"{focus_class_disp}: local prototype motion")

            set_visibility("final")
            ax.set_xlim(fxmin, fxmax)
            ax.set_ylim(fymin, fymax)

            current_proto_xy = {}
            for cls in unique_classes:
                pidx = [i for i, c in enumerate(proto_labels) if c == cls]
                current_proto_xy[cls] = proto_after_xy[pidx]
            update_positions(current_proto_xy, fxmax - fxmin, fymax - fymin)

            update_bars(alpha=1.0, visible=True)

        artists = [subtitle]
        if legend is not None:
            artists.append(legend)
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