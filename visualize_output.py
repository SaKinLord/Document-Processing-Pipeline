import os
import sys
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

from src.utils.image import denoise_image, deskew_image


# Color scheme by element type
TYPE_COLORS = {
    "text":          (30, 144, 255),   # dodger blue
    "table":         (0, 100, 200),    # dark blue
    "figure":        (34, 139, 34),    # forest green
    "image":         (34, 139, 34),    # forest green
    "human face":    (34, 139, 34),    # forest green
    "signature":     (148, 0, 211),    # purple
    "logo":          (0, 180, 180),    # teal
    "seal":          (180, 0, 180),    # magenta
    "layout_region": (180, 180, 180),  # light gray
}

# TrOCR gets a distinct accent color
TROCR_COLOR = (255, 140, 0)  # orange

# Hallucination flag
HALLUCINATION_COLOR = (255, 50, 50)  # red


def _get_font(size=12):
    """Try to load a readable font, fall back to PIL default."""
    if sys.platform == "win32":
        candidates = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    else:  # Linux / Colab
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


def _truncate(text, max_len=35):
    """Truncate text for display."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("<br>", " ").strip()
    if len(text) > max_len:
        return text[:max_len - 1] + "\u2026"
    return text


def _draw_label(draw, bbox, label, color, font, bg_alpha=180):
    """Draw a text label above a bounding box with a semi-transparent background."""
    x1, y1 = int(bbox[0]), int(bbox[1])

    # Get text size
    try:
        text_bbox = font.getbbox(label)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
    except AttributeError:
        tw, th = len(label) * 7, 12

    # Position label above the bbox, or inside if no room above
    label_y = y1 - th - 4
    if label_y < 0:
        label_y = y1 + 2

    # Draw background rectangle for readability
    bg_rect = [x1, label_y, x1 + tw + 4, label_y + th + 2]
    draw.rectangle(bg_rect, fill=(0, 0, 0, bg_alpha))
    draw.text((x1 + 2, label_y), label, fill=color, font=font)


def draw_legend(draw, img_width, img_height, font, show_layout):
    """Draw a color legend in the top-right corner."""
    legend_items = [
        ("text (surya)", TYPE_COLORS["text"]),
        ("text (trocr)", TROCR_COLOR),
        ("table", TYPE_COLORS["table"]),
        ("signature", TYPE_COLORS["signature"]),
        ("logo", TYPE_COLORS["logo"]),
        ("seal", TYPE_COLORS["seal"]),
        ("figure/image", TYPE_COLORS["figure"]),
        ("hallucination", HALLUCINATION_COLOR),
    ]
    if show_layout:
        legend_items.append(("layout_region", TYPE_COLORS["layout_region"]))

    line_h = 16
    padding = 8
    legend_w = 160
    legend_h = len(legend_items) * line_h + padding * 2

    x0 = img_width - legend_w - 10
    y0 = 10

    # Background
    draw.rectangle([x0, y0, x0 + legend_w, y0 + legend_h], fill=(0, 0, 0, 200))

    for i, (name, color) in enumerate(legend_items):
        y = y0 + padding + i * line_h
        # Color swatch
        draw.rectangle([x0 + padding, y + 2, x0 + padding + 10, y + 12], fill=color)
        # Label
        draw.text((x0 + padding + 14, y), name, fill=(255, 255, 255), font=font)


def visualize_document(image_path, json_path, output_path,
                       show_layout=False, show_content=True, show_confidence=True):
    """Visualize a single document with detailed annotations."""
    try:
        img = Image.open(image_path).convert("RGB")
        # Apply the same preprocessing as the OCR pipeline so that
        # bbox coordinates (computed on the preprocessed image) align.
        img = denoise_image(img)
        img = deskew_image(img)
        img = img.convert("RGBA")
    except (OSError, IOError) as e:
        print(f"Error loading image {os.path.basename(image_path)}: {e}")
        return

    # Create an overlay for semi-transparent fills
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    font = _get_font(11)
    small_font = _get_font(9)

    counts = {"text": 0, "text_trocr": 0, "layout_region": 0,
              "table": 0, "signature": 0, "logo": 0, "other": 0}

    for page in data.get("pages", []):
        elements = page.get("elements", [])
        for element in elements:
            bbox = element.get("bbox")
            if not bbox:
                continue

            el_type = element.get("type", "").lower()
            source = element.get("source_model", "")
            confidence = element.get("confidence")
            content = element.get("content", "")
            is_hallucination = element.get("hallucination_score") is not None
            hall_score = element.get("hallucination_score", 0)

            # Skip layout_regions unless requested
            if el_type == "layout_region" and not show_layout:
                continue

            # Determine color
            if el_type == "text" and source == "trocr":
                color = TROCR_COLOR
                counts["text_trocr"] += 1
            elif el_type in TYPE_COLORS:
                color = TYPE_COLORS[el_type]
                if el_type == "text":
                    counts["text"] += 1
                elif el_type in counts:
                    counts[el_type] += 1
                else:
                    counts["other"] += 1
            else:
                color = (200, 200, 200)
                counts["other"] += 1

            # Override color for hallucination-flagged elements
            if is_hallucination and hall_score >= 0.40:
                color = HALLUCINATION_COLOR

            rect = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            width = 2 if el_type == "layout_region" else 3

            # Draw semi-transparent fill for non-text visual elements
            if el_type in ("signature", "logo", "seal"):
                overlay_draw.rectangle(rect, fill=color + (40,))

            # Draw outline
            draw.rectangle(rect, outline=color, width=width)

            # Build label for text elements
            if el_type == "text" and show_content:
                parts = []
                if content:
                    parts.append(_truncate(content))
                meta = []
                if source == "trocr":
                    meta.append("TrOCR")
                if show_confidence and confidence is not None:
                    meta.append(f"{confidence:.2f}")
                if is_hallucination:
                    meta.append(f"H:{hall_score:.2f}")
                if meta:
                    parts.append(f"[{' | '.join(meta)}]")
                if parts:
                    _draw_label(draw, bbox, " ".join(parts), color, small_font)

            # Label non-text elements with their type
            elif el_type != "layout_region":
                label = el_type
                if el_type == "table":
                    score = element.get("structure_score")
                    if score is not None:
                        label = f"table (score:{score:.0f})"
                _draw_label(draw, bbox, label, color, small_font)

    # Composite overlay
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # Draw legend
    draw = ImageDraw.Draw(img)
    draw_legend(draw, img.width, img.height, small_font, show_layout)

    img.save(output_path)
    text_total = counts["text"] + counts["text_trocr"]
    print(f"  {os.path.basename(image_path)}: "
          f"{text_total} text ({counts['text_trocr']} trocr), "
          f"{counts['layout_region']} layout, "
          f"{counts['table']} table, "
          f"{counts['signature']} sig, "
          f"{counts['logo']} logo")


def visualize_bboxes(input_dir, json_dir, output_dir,
                     show_layout=False, show_content=True, show_confidence=True):
    """Iterate through images and create visualized versions."""
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [f for f in os.listdir(input_dir)
             if os.path.splitext(f)[1].lower() in valid_extensions]

    if not files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(files)} images. Starting visualization...")

    for filename in files:
        image_path = os.path.join(input_dir, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(json_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"Skipping {filename}: JSON not found")
            continue

        output_path = os.path.join(output_dir, filename)
        try:
            visualize_document(
                image_path, json_path, output_path,
                show_layout=show_layout,
                show_content=show_content,
                show_confidence=show_confidence,
            )
        except (OSError, IOError, ValueError, KeyError) as e:
            print(f"Error processing {filename}: {e}")

    print(f"Visualization complete. Check {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pipeline output with detailed annotations.")
    parser.add_argument("--input_dir", type=str, default="input",
                        help="Directory containing source images")
    parser.add_argument("--json_dir", type=str, default="output",
                        help="Directory containing JSON output files")
    parser.add_argument("--output_dir", type=str, default="output/visualized",
                        help="Directory to save visualized images")
    parser.add_argument("--show-layout", action="store_true",
                        help="Show layout_region boxes (hidden by default)")
    parser.add_argument("--no-content", action="store_true",
                        help="Hide text content labels")
    parser.add_argument("--no-confidence", action="store_true",
                        help="Hide confidence scores")

    args = parser.parse_args()

    visualize_bboxes(
        args.input_dir, args.json_dir, args.output_dir,
        show_layout=args.show_layout,
        show_content=not args.no_content,
        show_confidence=not args.no_confidence,
    )


if __name__ == "__main__":
    main()
