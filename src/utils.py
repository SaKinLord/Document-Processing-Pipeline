
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


def pad_bbox(bbox, padding, image_width, image_height):
    """
    Pad a bounding box by a given number of pixels, ensuring it stays within image bounds.
    
    This is particularly important for handwritten text where Surya's tight cropping
    may cut off descenders (g, y, p, q) which TrOCR needs to see for accurate recognition.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Number of pixels to add on each side
        image_width: Width of the source image
        image_height: Height of the source image
        
    Returns:
        Padded bounding box [x1, y1, x2, y2] clamped to image bounds
    """
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding)
    ]


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
    Classify document as typed, handwritten, or mixed based on multiple features.
    
    Returns: tuple (doc_type: str, confidence: float)
        - doc_type: "typed", "handwritten", or "mixed"
        - confidence: 0.0 to 1.0
    
    Features used:
    1. Stroke width variance (typed = uniform, handwritten = variable)
    2. Line regularity (typed = horizontal, handwritten = slanted)
    3. Edge density patterns
    4. Contour angle variance
    5. Form structure detection (lines, boxes)
    6. Character uniformity
    7. Signature isolation (exclude bottom signature region from analysis)
    8. Fax/letterhead header detection
    """
    try:
        # Convert to grayscale
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Resize for consistent analysis
        h, w = gray.shape
        orig_h = h
        if w > 1000:
            scale = 1000 / w
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            h, w = gray.shape
        
        # NEW: Isolate signature region (bottom 20% of page) and analyze main body separately
        signature_region_start = int(h * 0.80)
        main_body = gray[:signature_region_start, :]
        signature_region = gray[signature_region_start:, :]
        
        # Check if signature region has handwriting characteristics
        has_signature = _detect_signature_in_region(signature_region)
        
        # NEW: Detect fax/letterhead header patterns in top 15% of page
        header_region = gray[:int(h * 0.15), :]
        has_fax_header = _detect_fax_header(header_region)
        
        # Use main body (excluding signature) for feature extraction
        analysis_region = main_body if has_signature else gray
        
        # Feature 1: Stroke Width Variance
        stroke_score = _calculate_stroke_variance(analysis_region)
        
        # Feature 2: Line Regularity (horizontal alignment)
        line_score = _calculate_line_regularity(analysis_region)
        
        # Feature 3: Contour Angle Variance
        angle_score = _calculate_angle_variance(analysis_region)
        
        # Feature 4: Edge density ratio
        edge_score = _calculate_edge_density(analysis_region)
        
        # Feature 5: Form structure detection (lines, boxes)
        form_score = _detect_form_structure(gray)  # Use full page for form detection
        
        # Feature 6: Character uniformity
        uniformity_score = _calculate_character_uniformity(analysis_region)
        
        # Weighted voting for "typed"
        # Higher score = more likely typed
        typed_score = 0.0
        
        # Stroke variance: low = typed
        if stroke_score < 0.3:
            typed_score += 0.20
        elif stroke_score < 0.5:
            typed_score += 0.10
        
        # Line regularity: high = typed
        if line_score > 0.7:
            typed_score += 0.12
        elif line_score > 0.5:
            typed_score += 0.06
        
        # Angle variance: low = typed, HIGH = handwritten (ENHANCED WEIGHTING)
        # High angle variance is a strong indicator of handwriting
        # BUT only penalize if not a clear form document (form_score < 0.5)
        if angle_score < 400:
            typed_score += 0.10  # Very uniform angles (likely typed)
        elif angle_score < 800:
            typed_score += 0.05  # Somewhat uniform (reduced from 0.15)
        elif angle_score > 1500 and form_score < 0.5:
            typed_score -= 0.15  # High variance + not a form = strongly handwritten
        elif angle_score > 1200 and form_score < 0.5:
            typed_score -= 0.08  # Moderate-high variance + not a form = likely handwritten
        
        # Edge density: moderate = typed
        if 0.02 < edge_score < 0.15:
            typed_score += 0.08
        elif edge_score < 0.02:
            typed_score += 0.04
        
        # BOOSTED: Form structure - strong indicator of typed/form document
        if form_score > 0.5:
            typed_score += 0.35  # Strong form structure (boosted from 0.25)
        elif form_score > 0.2:
            typed_score += 0.25  # Some form elements (boosted from 0.15)
        elif form_score > 0.1:
            typed_score += 0.15  # Minimal form elements
        
        # Character uniformity - typed has uniform heights
        if uniformity_score > 0.7:
            typed_score += 0.12
        elif uniformity_score > 0.4:
            typed_score += 0.06
        
        # NEW: Fax/letterhead header boost - strong indicator of typed document
        if has_fax_header:
            typed_score += 0.20
        
        # NEW: If document has isolated signature but strong form structure, boost typed
        if has_signature and form_score > 0.2:
            typed_score += 0.10  # Signature on a form = typed document with signature
        
        # TARGETED: Ruled paper handwriting detection
        # Conditions: high line_score (ruled paper), low form_score (not a form),
        # moderate angle_score (uniform but not machine-perfect)
        # This catches handwritten docs on lined paper with consistent slant
        is_ruled_paper_handwriting = (
            line_score >= 0.95 and  # Very high line regularity (ruled paper)
            form_score < 0.25 and   # Not a structured form
            300 < angle_score < 700  # Uniform handwriting slant (not typed-perfect <300)
        )
        if is_ruled_paper_handwriting:
            typed_score -= 0.12  # Bias toward handwritten
            print(f"    [DEBUG] Ruled paper handwriting detected: -0.12 adjustment")
        
        # DEBUG LOGGING: Output all scoring variables
        print(f"    [DEBUG] Classification Scores:")
        print(f"      - stroke_score: {stroke_score:.3f} (low=typed)")
        print(f"      - line_score: {line_score:.3f} (high=typed)")
        print(f"      - angle_score: {angle_score:.1f} (low=typed)")
        print(f"      - edge_score: {edge_score:.4f}")
        print(f"      - form_score: {form_score:.3f} (high=typed/form)")
        print(f"      - uniformity_score: {uniformity_score:.3f} (high=typed)")
        print(f"    [DEBUG] Detection Flags:")
        print(f"      - has_signature: {has_signature}")
        print(f"      - has_fax_header: {has_fax_header}")
        print(f"    [DEBUG] Final typed_score: {typed_score:.3f}")
        print(f"      - Thresholds: typed>=0.65, handwritten<=0.45, else mixed")
        
        # Determine classification with confidence
        # Threshold: typed >= 0.65, handwritten <= 0.45, else mixed
        # Adjusted thresholds for better separation with signature isolation
        if typed_score >= 0.65:
            return ("typed", min(typed_score, 0.95))
        elif typed_score <= 0.45:
            return ("handwritten", min(1.0 - typed_score, 0.95))
        else:
            # Uncertain - classify as mixed (will use ensemble OCR)
            return ("mixed", 0.5)
            
    except Exception as e:
        print(f"Warning: Document classification failed, defaulting to mixed. Error: {e}")
        return ("mixed", 0.3)


def _detect_signature_in_region(region):
    """
    Detect if a region (typically bottom 20% of page) contains a handwritten signature.
    Signatures have distinctive characteristics: curved strokes, varying thickness, isolated ink.
    Returns: bool
    """
    try:
        if region.shape[0] < 20 or region.shape[1] < 50:
            return False
            
        # Binarize
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Count ink pixels in signature region
        ink_pixels = np.count_nonzero(binary)
        total_pixels = region.shape[0] * region.shape[1]
        ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0
        
        # Signature typically has 0.5-5% ink coverage
        if ink_ratio < 0.003 or ink_ratio > 0.15:
            return False
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels < 2:  # Only background
            return False
        
        # Look for signature-like components (not too many, moderate size)
        sig_components = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Signature strokes: moderate area, wider than tall typically
            if 50 < area < total_pixels * 0.3 and width > 20:
                sig_components += 1
        
        # Signature typically has 1-10 main components
        return 1 <= sig_components <= 15
        
    except Exception:
        return False


def _detect_fax_header(header_region):
    """
    Detect fax/letterhead header patterns.
    Fax headers have: horizontal lines, dense text in header, specific patterns.
    Returns: bool
    """
    try:
        if header_region.shape[0] < 30 or header_region.shape[1] < 100:
            return False
        
        # Binarize
        _, binary = cv2.threshold(header_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Check for horizontal lines (fax separator lines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        h_line_pixels = np.count_nonzero(horizontal_lines)
        
        # Check ink density in header
        ink_pixels = np.count_nonzero(binary)
        total_pixels = header_region.shape[0] * header_region.shape[1]
        ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0
        
        # Fax headers typically have:
        # - Horizontal separator lines (h_line_pixels > 0)
        # - Moderate to high text density (3-20%)
        has_lines = h_line_pixels > (header_region.shape[1] * 0.1)  # At least 10% width line
        has_dense_text = 0.02 < ink_ratio < 0.25
        
        return has_lines and has_dense_text
        
    except Exception:
        return False


def _calculate_stroke_variance(gray):
    """
    Calculate stroke width variance using distance transform.
    Low variance = typed, high variance = handwritten.
    Returns: 0.0 (uniform) to 1.0 (variable)
    """
    try:
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Distance transform to find stroke widths
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Get non-zero stroke widths
        stroke_widths = dist[dist > 0]
        
        if len(stroke_widths) < 100:
            return 0.5  # Not enough data
        
        # Calculate coefficient of variation (normalized variance)
        mean_width = np.mean(stroke_widths)
        if mean_width < 0.1:
            return 0.5
            
        cv_score = np.std(stroke_widths) / mean_width
        
        # Normalize to 0-1 range (cv of 0.5+ is highly variable)
        return min(cv_score / 0.5, 1.0)
        
    except Exception:
        return 0.5


def _calculate_line_regularity(gray):
    """
    Calculate how horizontal/regular text lines are.
    High regularity = typed, low = handwritten.
    Returns: 0.0 (irregular) to 1.0 (regular)
    """
    try:
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            return 0.5
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(abs(angle))
        
        if not angles:
            return 0.5
        
        # Count near-horizontal lines (within 5 degrees)
        horizontal_count = sum(1 for a in angles if a < 5 or a > 175)
        regularity = horizontal_count / len(angles)
        
        return regularity
        
    except Exception:
        return 0.5


def _calculate_angle_variance(gray):
    """
    Original angle variance calculation from contour fitting.
    Returns raw variance value (lower = more typed).
    """
    try:
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) > 50 and len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angles.append(ellipse[2])
        
        if not angles:
            return 1000  # Default middle value
        
        return np.var(angles)
        
    except Exception:
        return 1000


def _calculate_edge_density(gray):
    """
    Calculate edge density ratio.
    Returns: ratio of edge pixels to total pixels.
    """
    try:
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        return edge_pixels / total_pixels if total_pixels > 0 else 0.0
        
    except Exception:
        return 0.1


def _detect_form_structure(gray):
    """
    Detect horizontal/vertical lines indicating form structure.
    Forms, faxes, and letterheads have distinctive line patterns.
    Returns: 0.0 (no structure) to 1.0 (strong form structure)
    """
    try:
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines using morphology
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines using morphology
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count line pixels
        h_line_pixels = np.count_nonzero(horizontal_lines)
        v_line_pixels = np.count_nonzero(vertical_lines)
        total_line_pixels = h_line_pixels + v_line_pixels
        
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Normalize: typical form has ~0.5-2% line coverage
        line_ratio = total_line_pixels / total_pixels if total_pixels > 0 else 0
        
        # Scale to 0-1 (0.01 = strong form indicator)
        form_score = min(line_ratio / 0.01, 1.0)
        
        return form_score
        
    except Exception:
        return 0.0


def _calculate_text_density(gray):
    """
    Calculate text/content density.
    Sparse content = likely a form template, Dense = handwritten or full typed doc.
    Returns: 0.0 (sparse) to 1.0 (dense)
    """
    try:
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Count text pixels
        text_pixels = np.count_nonzero(binary)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Typical document has 2-10% text coverage
        text_ratio = text_pixels / total_pixels if total_pixels > 0 else 0
        
        # Normalize: 0.10 = dense text
        density = min(text_ratio / 0.10, 1.0)
        
        return density
        
    except Exception:
        return 0.5


def _calculate_character_uniformity(gray):
    """
    Calculate character height uniformity.
    Typed text has uniform character heights; handwritten varies.
    Returns: 0.0 (variable = handwritten) to 1.0 (uniform = typed)
    """
    try:
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components (characters)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels < 10:
            return 0.5  # Not enough data
        
        # Get heights of components (excluding background and very small noise)
        heights = []
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # Filter: reasonable character size
            if 5 < h < gray.shape[0] // 3 and area > 20:
                heights.append(h)
        
        if len(heights) < 10:
            return 0.5
        
        # Calculate coefficient of variation (lower = more uniform)
        mean_height = np.mean(heights)
        std_height = np.std(heights)
        
        if mean_height < 1:
            return 0.5
            
        cv_score = std_height / mean_height
        
        # Normalize: cv of 0.3+ is highly variable (handwritten)
        # cv of 0.1 or less is very uniform (typed)
        uniformity = max(0, 1.0 - (cv_score / 0.4))
        
        return min(uniformity, 1.0)
        
    except Exception:
        return 0.5
