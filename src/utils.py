
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from spellchecker import SpellChecker
import re
import pypdfium2 as pdfium
import numpy as np
import io
import cv2

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


def estimate_noise(image_np):
    """
    Estimates noise level using a simple median blur difference method.
    Higher score means more noise.
    """
    try:
        # Convert to gray
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Use a small crop or resize for speed if image is huge, 
        # but for accuracy on full page, we can just run on full or resized.
        # Let's resize to a fixed width for consistent metric and speed
        h, w = gray.shape
        if w > 1000:
            scale = 1000 / w
            gray = cv2.resize(gray, (0,0), fx=scale, fy=scale)
            
        # Median blur removes salt-and-pepper noise
        median = cv2.medianBlur(gray, 3)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray, median)
        
        # Mean difference is our noise score
        # Clean docs (digital PDF) -> score ~0-0.5 (mostly text edges)
        # Noisy scans -> score > 1.0
        score = np.mean(diff)
        return score
    except Exception:
        return 999.0 # Fail safe: assume noisy

def denoise_image(image):
    """
    Applies simple denoising to an image using OpenCV if noise is detected.
    Intended for scanned documents with salt-and-pepper noise.
    """
    try:
        # Convert PIL to cv2 (OpenCV uses BGR)
        img_np = np.array(image)
        
        # Check noise level
        noise_score = estimate_noise(img_np)
        # Threshold heuristic: Digital text is clean, scanned text has grain.
        # 1.0 is a conservative threshold. Digital docs usually < 0.5.
        if noise_score < 0.8: 
            return image
            
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Apply Fast Non-Local Means Denoising
        # h=10 is a common strength for decent noise removal without blurring text too much
        # templateWindowSize=7, searchWindowSize=21 are standard
        denoised_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)

        # Convert back to RGB and then PIL
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(denoised_rgb)
    except Exception as e:
        print(f"Error denoising image: {e}")
        return image

def cluster_text_rows(elements, y_threshold=15):
    """
    Clusters text elements into rows based on Y-coordinate alignment.
    Sorts rows by Y, and elements within rows by X.
    Expected 'elements' is a list of dicts, each having a 'bbox' [x1, y1, x2, y2].
    """
    if not elements:
        return []

    # Filter only text elements for clustering (optional, but good practice if mixed content)
    # For now, we assume all passed elements are relevant text/content
    # Sort elements by Y1 (top) initially
    sorted_elements = sorted(elements, key=lambda e: e['bbox'][1])

    rows = []
    current_row = []

    for elem in sorted_elements:
        if not current_row:
            current_row.append(elem)
            continue
        
        # Check alignment with the current row's representative (first item or average)
        # Using the first item of the sorted row is a simple heuristic
        ref_elem = current_row[0]
        
        # Calculate vertical overlap or proximity
        # Simple proximity check: is the top of this element close to the top of the ref element?
        # A more robust check might be vertical center or IOA, but Y-threshold is standard for simple text
        if abs(elem['bbox'][1] - ref_elem['bbox'][1]) < y_threshold:
            current_row.append(elem)
        else:
            # Finish current row
            # Sort current row by X1
            current_row.sort(key=lambda e: e['bbox'][0])
            rows.append(current_row)
            # Start new row
            current_row = [elem]

    # Append last row
    if current_row:
        current_row.sort(key=lambda e: e['bbox'][0])
        rows.append(current_row)

    # Flatten logic: We might want to return 'rows' structure, 
    # but the current pipeline outputs a flat list of elements.
    # We can assign 'row_id' to elements or just re-order them in the flat list to ensure reading order.
    # Let's return the re-ordered flat list for now, but enriched with 'row_id' if helpful.
    
    ordered_elements = []
    for row_idx, row in enumerate(rows):
        for elem in row:
            elem['row_id'] = row_idx
            ordered_elements.append(elem)
            
            
    return ordered_elements

def is_bbox_too_large(bbox, width, height, label=None):
    """
    Checks if a bounding box covers too much of the page area.
    Uses per-type thresholds.
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    # Using rough approximation if width/height are not passed from pipeline cleanly,
    # but they should be.
    # However, pipeline uses PIL image, so pixel coords. 
    # bbox is also in pixel coords (from Florence) or 0-1000? 
    # Florence returns pixel coords usually when processed with image size.
    # Let's assume pixel coords matching width/height.
    
    total_area = width * height
    if total_area == 0:
        return False
        
    ratio = area / total_area
    
    # Per-type thresholds
    AREA_THRESHOLDS = {
        "human face": 0.10,
        "signature": 0.25,
        "logo": 0.30,
        "graphic": 0.50,
        "default": 0.50
    }
    
    threshold = AREA_THRESHOLDS.get(label.lower(), AREA_THRESHOLDS["default"]) if label else AREA_THRESHOLDS["default"]
    

    return ratio > threshold


class SpellCorrector:
    def __init__(self, domain_dict_path=None):
        self.spell = SpellChecker()
        self.domain_words = set()
        
        if domain_dict_path and os.path.exists(domain_dict_path):
            try:
                with open(domain_dict_path, 'r') as f:
                    content = f.read()
                    # Add to both local set and pyspellchecker
                    for line in content.splitlines():
                        word = line.strip()
                        if word:
                            self.spell.word_frequency.load_words([word])
                            self.domain_words.add(word.lower()) 
            except Exception as e:
                print(f"Warning: Could not load domain dictionary: {e}")

    def _preserve_case(self, original, corrected):
        if original.isupper():
            return corrected.upper()
        if original.istitle():
            return corrected.title()
        return corrected.lower()

    def correct_text(self, text):
        if not text:
            return text
            
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip logic:
            # 1. Short words (<4 chars) - User requested skip < 4
            if len(word) < 4:
                corrected_words.append(word)
                continue
                
            # 2. Contains digits (e.g. T07281A)
            if any(char.isdigit() for char in word):
                corrected_words.append(word)
                continue
                
            # Clean punctuation for checking
            clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
            if not clean_word:
                corrected_words.append(word)
                continue
            
            # 3. Starts with Uppercase (Skip proper nouns/Acronyms)
            if clean_word[0].isupper():
                corrected_words.append(word)
                continue

            # 4. Domain Dictionary Check
            if clean_word.lower() in self.domain_words:
                 corrected_words.append(word)
                 continue

            # 5. Generic Dictionary Check (pyspellchecker)
            if clean_word.lower() in self.spell:
                corrected_words.append(word)
                continue
            
            # Attempt correction for remaining words
            try:
                # Get candidates
                candidates = self.spell.candidates(clean_word)
                if not candidates:
                    corrected_words.append(word)
                    continue
                    
                best_candidate = self.spell.correction(clean_word)
                
                if best_candidate and best_candidate != clean_word:
                    # Apply correction
                    final_word = self._preserve_case(clean_word, best_candidate)
                    
                    # Re-attach punctuation
                    prefix = word[:word.find(clean_word)]
                    suffix = word[word.find(clean_word)+len(clean_word):]
                    corrected_words.append(prefix + final_word + suffix)
                else:
                    corrected_words.append(word)
            except Exception:
                 corrected_words.append(word)
                
        return " ".join(corrected_words)

def classify_document_type(image):
    """
    Classify document as typed or handwritten based on text line characteristics.
    Returns: "typed" or "handwritten"
    """
    try:
        # Convert to grayscale
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour characteristics
        angles = []
        sizes = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                # Fit ellipse to get angle
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    angles.append(ellipse[2])
                    sizes.append(cv2.contourArea(contour))
        
        if not angles:
            return "typed"
        
        # Handwriting tends to have high angle variance
        angle_variance = np.var(angles)
        # size_variance = np.var(sizes) / (np.mean(sizes) + 1)
        
        # Thresholds determined empirically (from Spec Report & Verification)
        # 1077 (Jittery Typed) < Threshold < 2024 (Handwritten)
        if angle_variance > 1200:
            return "handwritten"
        
        return "typed"
    except Exception as e:
        print(f"Warning: Document classification failed, defaulting to typed. Error: {e}")
        return "typed"

