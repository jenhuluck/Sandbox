import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Optional but recommended:
# pip install rembg
USE_REMBG = True
try:
    from rembg import remove
except Exception:
    USE_REMBG = False

# ----------------------------
# China visa photo settings
# ----------------------------
OUT_W = 354   # accepted lower-bound width
OUT_H = 515   # fits accepted digital height range
HEAD_HEIGHT_RATIO = 0.60  # tune so head height is roughly 28-33mm of 48mm => 58%~69%
TOP_MARGIN_RATIO = 0.08   # small top space above hair
JPEG_QUALITY = 95

# ----------------------------
# Utilities
# ----------------------------
def remove_background_to_white(pil_img: Image.Image) -> Image.Image:
    """
    Remove background and replace with white.
    Uses rembg if available. Falls back to original image if not.
    """
    pil_img = pil_img.convert("RGBA")

    if USE_REMBG:
        cutout = remove(pil_img)  # RGBA with transparent background
        white_bg = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
        combined = Image.alpha_composite(white_bg, cutout)
        return combined.convert("RGB")

    # Fallback: no background removal
    return pil_img.convert("RGB")


def detect_face_bbox_rgb(rgb_np: np.ndarray):
    """
    Detect a single face using MediaPipe.
    Returns bbox in pixel coords: (x1, y1, x2, y2)
    """
    mp_face = mp.solutions.face_detection
    h, w = rgb_np.shape[:2]

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_np)

    if not results.detections:
        raise RuntimeError("No face detected. Please use a clearer frontal photo.")

    # Use the largest detected face
    best = None
    best_area = -1
    for det in results.detections:
        box = det.location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        bw = int(box.width * w)
        bh = int(box.height * h)
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)

    return best


def make_china_visa_photo(
    input_path: str,
    output_path: str,
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    head_height_ratio: float = HEAD_HEIGHT_RATIO,
    top_margin_ratio: float = TOP_MARGIN_RATIO,
):
    # Load image
    pil_img = Image.open(input_path).convert("RGB")

    # Background to white
    pil_white = remove_background_to_white(pil_img)

    # Convert to numpy RGB
    rgb = np.array(pil_white)
    img_h, img_w = rgb.shape[:2]

    # Detect face
    fx1, fy1, fx2, fy2 = detect_face_bbox_rgb(rgb)
    face_w = fx2 - fx1
    face_h = fy2 - fy1

    # Estimate head region from face box
    # MediaPipe face box is usually face-only, not entire head/hair.
    # Expand upward and sideways to approximate full head.
    head_x1 = max(0, int(fx1 - 0.25 * face_w))
    head_x2 = min(img_w, int(fx2 + 0.25 * face_w))
    head_y1 = max(0, int(fy1 - 0.35 * face_h))  # include forehead/hair
    head_y2 = min(img_h, int(fy2 + 0.10 * face_h))
    head_h = head_y2 - head_y1
    head_w = head_x2 - head_x1

    # Decide crop size so head occupies desired fraction of final photo height
    crop_h = int(head_h / head_height_ratio)
    crop_w = int(crop_h * out_w / out_h)

    # Put head near top with a small margin
    desired_top_margin = int(top_margin_ratio * crop_h)
    crop_y1 = head_y1 - desired_top_margin

    # Center horizontally on head
    head_cx = (head_x1 + head_x2) // 2
    crop_x1 = head_cx - crop_w // 2

    # Clamp crop box inside image
    crop_x1 = max(0, min(crop_x1, img_w - crop_w))
    crop_y1 = max(0, min(crop_y1, img_h - crop_h))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    # If crop is bigger than image, pad with white
    if crop_w > img_w or crop_h > img_h:
        pad_w = max(crop_w, img_w)
        pad_h = max(crop_h, img_h)
        canvas = np.full((pad_h, pad_w, 3), 255, dtype=np.uint8)
        off_x = (pad_w - img_w) // 2
        off_y = (pad_h - img_h) // 2
        canvas[off_y:off_y + img_h, off_x:off_x + img_w] = rgb
        rgb = canvas
        img_h, img_w = rgb.shape[:2]

        crop_x1 = max(0, min(crop_x1 + off_x, img_w - crop_w))
        crop_y1 = max(0, min(crop_y1 + off_y, img_h - crop_h))
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h

    cropped = rgb[crop_y1:crop_y2, crop_x1:crop_x2]

    # Final resize
    final_img = Image.fromarray(cropped).resize((out_w, out_h), Image.LANCZOS)

    # Save as JPEG
    final_img.save(output_path, format="JPEG", quality=JPEG_QUALITY)

    print(f"Saved visa photo to: {output_path}")
    print(f"Output size: {out_w}x{out_h}")
    print("Please visually verify:")
    print("- plain white background")
    print("- full front face")
    print("- head not too small or too large")
    print("- top of hair visible")
    print("- no heavy shadow / glare")


if __name__ == "__main__":
    input_path = "input.jpg"
    output_path = "china_visa_photo.jpg"
    make_china_visa_photo(input_path, output_path)