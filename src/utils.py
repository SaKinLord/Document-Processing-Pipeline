
import os
from PIL import Image
import pypdfium2 as pdfium
import numpy as np
import io

def load_image(image_path):
    """Loads an image from a file path."""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def convert_pdf_to_images(pdf_path, dpi=200):
    """Converts a PDF file to a list of PIL Images."""
    images = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=dpi/72)
            pil_image = bitmap.to_pil()
            images.append(pil_image.convert("RGB"))
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
    return images

def crop_image(image, bbox):
    """Crops an image given a bounding box [x1, y1, x2, y2]."""
    return image.crop(bbox)

def normalize_bbox(bbox, width, height):
    """Normalizes bounding box coordinates to 0-1000 scale."""
    x1, y1, x2, y2 = bbox
    return [
        int(x1 / width * 1000),
        int(y1 / height * 1000),
        int(x2 / width * 1000),
        int(y2 / height * 1000)
    ]
