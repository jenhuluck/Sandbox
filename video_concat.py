import argparse
from moviepy import VideoFileClip, CompositeVideoClip


def resize_with_padding(clip, target_w, target_h):
    """
    Resize clip to fit inside target size while keeping aspect ratio,
    then place it on a black canvas of target_w x target_h.
    """
    src_w, src_h = clip.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        # fit width
        new_w = target_w
        new_h = int(target_w / src_ratio)
    else:
        # fit height
        new_h = target_h
        new_w = int(target_h * src_ratio)

    resized = clip.resized((new_w, new_h))

    # put resized clip on black background canvas
    canvas = CompositeVideoClip(
        [resized.with_position(("center", "center"))],
        size=(target_w, target_h)
    ).with_duration(resized.duration)

    # keep original audio
    if clip.audio is not None:
        canvas = canvas.with_audio(resized.audio)

    return canvas


def build_timeline(clips, transition_duration):
    """
    Create a timeline with crossfade transitions.
    Each next clip starts before previous ends.
    """
    positioned = []
    current_start = 0.0

    for i, clip in enumerate(clips):
        if i == 0:
            positioned_clip = clip.with_start(current_start)
            current_start += clip.duration
        else:
            current_start -= transition_duration
            positioned_clip = clip.with_start(current_start).crossfadein(transition_duration)
            current_start += clip.duration

        positioned.append(positioned_clip)

    final_duration = max(c.start + c.duration for c in positioned)
    final = CompositeVideoClip(positioned, size=clips[0].size).with_duration(final_duration)
    return final


def main():
    parser = argparse.ArgumentParser(description="Concatenate 3 videos with resize and crossfade transitions.")
    parser.add_argument("--video1", type=str, required=True, help="Path to first video")
    parser.add_argument("--video2", type=str, required=True, help="Path to second video")
    parser.add_argument("--video3", type=str, required=True, help="Path to third video")
    parser.add_argument("--output", type=str, default="merged_output.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument("--transition", type=float, default=1.0, help="Crossfade duration in seconds")
    args = parser.parse_args()

    paths = [args.video1, args.video2, args.video3]

    raw_clips = [VideoFileClip(p) for p in paths]

    # Resize each clip to same padded output size
    processed_clips = [resize_with_padding(c, args.width, args.height) for c in raw_clips]

    # Build final timeline with crossfades
    final = build_timeline(processed_clips, args.transition)

    # Write output
    final.write_videofile(
        args.output,
        codec="libx264",
        audio_codec="aac",
        fps=args.fps
    )

    # Close clips
    final.close()
    for c in processed_clips:
        c.close()
    for c in raw_clips:
        c.close()


if __name__ == "__main__":
    main()