import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

def visualize_bboxes(input_dir, json_dir, output_dir):
    """
    Iterates through images in input_dir, finds corresponding JSON in json_dir,
    draws bounding boxes, and saves to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(files)} images. Starting visualization...")

    for filename in files:
        image_path = os.path.join(input_dir, filename)
        
        # Determine expected JSON filename
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(json_dir, json_filename)
        
        if not os.path.exists(json_path):
            print(f"Skipping {filename}: JSON not found at {json_path}")
            continue
            
        try:
            # Load Image
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue

            draw = ImageDraw.Draw(img)
            
            # Load JSON
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Iterate through pages and elements
            pages = data.get("pages", [])
            draw_count = 0
            
            for page in pages:
                elements = page.get("elements", [])
                for element in elements:
                    bbox = element.get("bbox")
                    if bbox:
                        # bbox format assumed to be [x1, y1, x2, y2]
                        # Draw rectangle
                        # Color coding by type could be nice, currently Red for all
                        color = "red"
                        if element.get("type") == "table":
                            color = "blue"
                        elif element.get("type") == "figure" or element.get("type") == "image":
                            color = "green"
                            
                        draw.rectangle(bbox, outline=color, width=3)
                        draw_count += 1
            
            # Save result
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
            print(f"Saved visualization for {filename} ({draw_count} boxes)")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Visualization complete. Check {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes from JSON output on PNG images.")
    parser.add_argument("--input_dir", type=str, default="input", help="Directory containing source images")
    parser.add_argument("--json_dir", type=str, default="output", help="Directory containing JSON output files")
    parser.add_argument("--output_dir", type=str, default="output/visualized", help="Directory to save visualized images")
    
    args = parser.parse_args()
    
    visualize_bboxes(args.input_dir, args.json_dir, args.output_dir)

if __name__ == "__main__":
    main()
