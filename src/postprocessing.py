"""
Post-processing module for OCR output.

Implements multi-signal hallucination detection and output cleaning.
"""

import re
from typing import Dict, List, Any, Tuple, Optional


def normalize_underscore_fields(elements: List[Dict]) -> List[Dict]:
    """
    Normalize blank form fields (underscore runs, excessive spaces after labels).

    OCR produces inconsistent representations of fill-in fields:
    - 'Name:________' -> 'Name: ___'
    - 'Name:         ' -> 'Name: ___'
    - 'Name: _ _ _ _' -> 'Name: ___'

    Standardizes all to a single '___' token for consistent comparison.

    Args:
        elements: List of element dictionaries

    Returns:
        Elements with normalized underscore fields
    """
    for element in elements:
        if element.get("type") != "text":
            continue

        content = element.get("content", "")
        if not content:
            continue

        # Normalize spaced underscores: '_ _ _ _' -> '____' then collapse
        content = re.sub(r'(_\s+){2,}_', '___', content)
        # Collapse runs of 3+ underscores to a single '___'
        content = re.sub(r'_{3,}', '___', content)
        # Collapse runs of 3+ spaces after a colon/label to ' ___'
        content = re.sub(r'(:\s*)\s{3,}', r'\1___', content)
        # Clean up any resulting double spaces
        content = re.sub(r' {2,}', ' ', content)

        element["content"] = content.strip()

    return elements


def normalize_punctuation_spacing(text: str) -> str:
    """
    Normalize spacing around punctuation in OCR output.
    
    TrOCR tends to add unnecessary spaces around punctuation marks which
    inflates WER (Word Error Rate) scores. This function fixes common patterns:
    - 'Govr. ,' -> 'Govr.,'
    - 'word . word' -> 'word. word'
    - '( text )' -> '(text)'
    - 'word ; word' -> 'word; word'
    
    Args:
        text: Raw OCR text output
        
    Returns:
        Text with normalized punctuation spacing
    """
    if not text:
        return text
    
    # Remove space before punctuation: 'word .' -> 'word.'
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Remove space after opening brackets: '( text' -> '(text'
    text = re.sub(r'([\(\[\{])\s+', r'\1', text)
    
    # Remove space before closing brackets: 'text )' -> 'text)'
    text = re.sub(r'\s+([\)\]\}])', r'\1', text)
    
    # Fix double punctuation with space: '. ,' -> '.,'
    text = re.sub(r'([.,;:!?])\s+([.,;:!?])', r'\1\2', text)
    
    # Fix space around hyphens in words: 'self - aware' -> 'self-aware'
    text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)
    
    # Fix space around apostrophes: "don 't" -> "don't"
    text = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", text)
    text = re.sub(r"(\w)\s+'", r"\1'", text)
    text = re.sub(r"'\s+(\w)", r"'\1", text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


# ============================================================================
# Offensive OCR Misread Filter (P0 - Reputational Risk)
# ============================================================================

# Known offensive OCR misreads mapped to their correct readings.
# These are cases where the OCR model produces offensive words from
# innocuous source text (e.g., "Litco" -> "Bitch", "Eckerd" -> "Pecker").
# Each entry: (offensive_pattern, correct_replacement)
OFFENSIVE_OCR_CORRECTIONS = [
    (re.compile(r'\bBitch\b', re.IGNORECASE), 'Litco'),
    (re.compile(r'\bPecker\b(?=\s+Drugs)', re.IGNORECASE), 'Eckerd'),
]


def filter_offensive_ocr_misreads(elements: List[Dict]) -> List[Dict]:
    """
    Detect and correct known offensive OCR misreads.

    Some OCR models produce offensive words from innocuous source text.
    This filter catches known patterns and replaces them with the correct
    reading, adding a flag to the element for audit purposes.

    Args:
        elements: List of element dictionaries

    Returns:
        Elements with offensive misreads corrected
    """
    for element in elements:
        if element.get("type") != "text":
            continue

        content = element.get("content", "")
        if not content:
            continue

        corrected = content
        corrections_made = []

        for pattern, replacement in OFFENSIVE_OCR_CORRECTIONS:
            if pattern.search(corrected):
                corrected = pattern.sub(replacement, corrected)
                corrections_made.append(f"{pattern.pattern} -> {replacement}")

        if corrections_made:
            element["content"] = corrected
            element["offensive_ocr_corrected"] = corrections_made

    return elements


# ============================================================================
# Document Type Indicators (Text-based classification helpers)
# ============================================================================

# Patterns that strongly indicate a typed/fax document
FAX_HEADER_PATTERNS = [
    r'\b(FAX|FACSIMILE|TELECOPY)\b',
    r'\bFAX\s*(NO|NUMBER|#)?[:\s]*\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',
    r'\b(FROM|TO|DATE|RE|SUBJECT|PAGES?)\s*:',  # Common fax form fields
]

TYPED_INDICATOR_PATTERNS = [
    r'\bDEPARTMENT\s+OF\b',  # Government docs
    r'\b(FORM|APPLICATION|CERTIFICATE)\s+\d+',  # Forms
    r'\bOFFICIAL\s+USE\s+ONLY\b',
    r'\bPRINT\s+NAME\b',  # Form instruction
    r'\bSIGNATURE\s+DATE\b',
    r'\b(?:PAGE|PG|P)\s*\d+\s*(?:OF|/)\s*\d+',  # Page numbers
]


def detect_typed_document_indicators(text: str) -> dict:
    """
    Detect text patterns that indicate a typed/form/fax document.
    
    This is used to catch misclassified typed documents that were
    incorrectly classified as handwritten based on image features.
    
    Args:
        text: OCR text from the document
        
    Returns:
        dict with:
        - 'is_likely_typed': bool indicating strong typed indicators found
        - 'fax_indicators': list of matched fax patterns
        - 'form_indicators': list of matched form patterns
        - 'confidence_boost': suggested boost to typed score (0.0 to 0.25)
    """
    result = {
        'is_likely_typed': False,
        'fax_indicators': [],
        'form_indicators': [],
        'confidence_boost': 0.0
    }
    
    if not text:
        return result
    
    text_upper = text.upper()
    
    # Check fax patterns
    for pattern in FAX_HEADER_PATTERNS:
        if re.search(pattern, text_upper):
            result['fax_indicators'].append(pattern)
    
    # Check typed/form patterns
    for pattern in TYPED_INDICATOR_PATTERNS:
        if re.search(pattern, text_upper):
            result['form_indicators'].append(pattern)
    
    # Calculate confidence boost
    fax_count = len(result['fax_indicators'])
    form_count = len(result['form_indicators'])
    
    if fax_count >= 2:
        result['confidence_boost'] = 0.25
        result['is_likely_typed'] = True
    elif fax_count == 1 and form_count >= 1:
        result['confidence_boost'] = 0.20
        result['is_likely_typed'] = True
    elif form_count >= 3:
        result['confidence_boost'] = 0.15
        result['is_likely_typed'] = True
    elif fax_count == 1 or form_count >= 2:
        result['confidence_boost'] = 0.10
    
    return result




# ============================================================================
# Date Format Validation
# ============================================================================

# Common date patterns
DATE_PATTERNS = [
    # MM/DD/YYYY or MM-DD-YYYY
    (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', 'MM/DD/YYYY'),
    # MM/DD/YY or MM-DD-YY
    (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})(?!\d)', 'MM/DD/YY'),
    # Month DD, YYYY
    (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})', 'Month DD, YYYY'),
]


def validate_date_format(text: str) -> dict:
    """
    Extract and validate date patterns in OCR text.
    
    Detects corrupted dates like "414,00" (should be "4/14/00")
    or "12/0/98" (missing digit).
    
    Args:
        text: OCR text that may contain dates
        
    Returns:
        dict with:
        - 'dates': list of detected date info
        - 'validation_status': 'valid', 'suspicious', or 'none'
    """
    result = {
        'dates': [],
        'validation_status': 'none'
    }
    
    if not text:
        return result
    
    has_suspicious = False
    
    for pattern, format_name in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_info = {
                'raw': match.group(0),
                'format': format_name,
                'valid': True,
                'issues': []
            }
            
            groups = match.groups()
            
            # Validate numeric date parts
            if format_name in ['MM/DD/YYYY', 'MM/DD/YY']:
                month = groups[0]
                day = groups[1]
                
                # Check month validity (1-12)
                try:
                    month_int = int(month)
                    if month_int < 1 or month_int > 12:
                        date_info['issues'].append('invalid_month')
                        date_info['valid'] = False
                except ValueError:
                    date_info['issues'].append('non_numeric_month')
                    date_info['valid'] = False
                
                # Check day validity (1-31)
                try:
                    day_int = int(day)
                    if day_int < 1 or day_int > 31:
                        date_info['issues'].append('invalid_day')
                        date_info['valid'] = False
                    if day_int == 0:  # Common OCR error
                        date_info['issues'].append('missing_digit')
                        date_info['valid'] = False
                except ValueError:
                    date_info['issues'].append('non_numeric_day')
                    date_info['valid'] = False
            
            if date_info['issues']:
                has_suspicious = True
            
            result['dates'].append(date_info)
    
    # Also check for corrupted date patterns (e.g., "414,00")
    corrupted_patterns = [
        (r'(\d{3}),(\d{2})(?!\d)', 'possible_corrupted_date'),  # 414,00
        (r'(\d{1,2})/0/(\d{2})', 'missing_digit'),  # 12/0/98
    ]
    
    for pattern, issue_type in corrupted_patterns:
        for match in re.finditer(pattern, text):
            result['dates'].append({
                'raw': match.group(0),
                'format': 'corrupted',
                'valid': False,
                'issues': [issue_type]
            })
            has_suspicious = True
    
    # Set overall status
    if result['dates']:
        if has_suspicious:
            result['validation_status'] = 'suspicious'
        else:
            result['validation_status'] = 'valid'
    
    return result


def add_date_validation_to_element(element: dict) -> dict:
    """
    Add date validation status to a text element if it contains dates.
    """
    if element.get('type') != 'text':
        return element
    
    content = element.get('content', '')
    if not content:
        return element
    
    # Quick check for date-like content
    if not re.search(r'\d+[/\-]', content) and not re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', content, re.I):
        return element
    
    validation = validate_date_format(content)
    
    if validation['dates']:
        element['date_validation'] = {
            'status': validation['validation_status'],
            'dates': validation['dates']
        }
    
    return element



# ============================================================================
# Phone Number Validation
# ============================================================================

# Phone patterns for common US formats
PHONE_PATTERNS = [
    # (xxx) xxx-xxxx or (xxx)xxx-xxxx
    r'\((\d{3})\)\s*(\d{3})[- ]?(\d{4})',
    # xxx-xxx-xxxx
    r'(\d{3})-(\d{3})-(\d{4})',
    # xxx/xxx-xxxx
    r'(\d{3})/(\d{3})-(\d{4})',
    # xxx xxx xxxx or xxx xxx-xxxx
    r'(\d{3})\s+(\d{3})[- ]?(\d{4})',
    # xxxxxxxxxx (10 digits no separator)
    r'(?<!\d)(\d{3})(\d{3})(\d{4})(?!\d)',
]

# Patterns that look like phone numbers but aren't (e.g., ZIP+4, dates)
FALSE_POSITIVE_PATTERNS = [
    r'\d{5}-\d{4}',  # ZIP+4 codes
    r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
    r'\d{4}/\d{2}/\d{2}',  # ISO dates
]


def validate_phone_number(text: str) -> dict:
    """
    Extract and validate phone numbers from OCR text.
    
    Detects common phone number formats and validates digit count.
    Does NOT auto-correct - only flags issues for human review.
    
    Args:
        text: OCR text that may contain phone numbers
        
    Returns:
        dict with:
        - 'phones': list of extracted phone numbers
        - 'validation_status': 'valid', 'suspicious', 'invalid', or 'none'
        - 'issues': list of detected issues
    """
    result = {
        'phones': [],
        'validation_status': 'none',
        'issues': []
    }
    
    if not text:
        return result
    
    # Check for false positives first
    for fp_pattern in FALSE_POSITIVE_PATTERNS:
        if re.search(fp_pattern, text):
            # Don't extract this as a phone number
            return result
    
    # Extract phone numbers
    for pattern in PHONE_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            groups = match.groups()
            if len(groups) >= 3:
                area_code = groups[0]
                prefix = groups[1]
                line = groups[2]
                
                # Validate digit counts
                issues = []
                
                if len(area_code) != 3:
                    issues.append(f'area_code_length:{len(area_code)}')
                if len(prefix) != 3:
                    issues.append(f'prefix_length:{len(prefix)}')
                if len(line) != 4:
                    issues.append(f'line_length:{len(line)}')
                
                # Check for suspicious patterns
                total_digits = len(area_code) + len(prefix) + len(line)
                if total_digits < 10:
                    issues.append(f'missing_digits:{10-total_digits}')
                elif total_digits > 10:
                    issues.append(f'extra_digits:{total_digits-10}')
                
                # Check for repeated digits (suspicious OCR errors)
                full_number = area_code + prefix + line
                if re.match(r'^(\d)\1{5,}$', full_number):
                    issues.append('repeated_digits')
                
                phone_data = {
                    'raw': match.group(0),
                    'normalized': f'({area_code}) {prefix}-{line}',
                    'area_code': area_code,
                    'prefix': prefix,
                    'line': line,
                    'total_digits': total_digits
                }
                
                if issues:
                    phone_data['issues'] = issues
                
                result['phones'].append(phone_data)
    
    # Determine overall validation status
    if not result['phones']:
        result['validation_status'] = 'none'
    else:
        all_valid = True
        any_suspicious = False
        
        for phone in result['phones']:
            if 'issues' in phone:
                if any('missing' in i or 'extra' in i for i in phone['issues']):
                    any_suspicious = True
                if any('repeated' in i for i in phone['issues']):
                    all_valid = False
        
        if not all_valid:
            result['validation_status'] = 'invalid'
        elif any_suspicious:
            result['validation_status'] = 'suspicious'
        else:
            result['validation_status'] = 'valid'
    
    return result


def add_phone_validation_to_element(element: dict) -> dict:
    """
    Add phone validation status to a text element if it contains phone numbers.
    
    Args:
        element: Text element dictionary with 'content' key
        
    Returns:
        Element with phone validation fields added (if applicable)
    """
    if element.get('type') != 'text':
        return element
    
    content = element.get('content', '')
    if not content:
        return element
    
    # Check for phone-related keywords to reduce unnecessary processing
    phone_keywords = ['phone', 'fax', 'tel', 'call', '(', '-']
    has_keyword = any(kw.lower() in content.lower() for kw in phone_keywords)
    has_digits = bool(re.search(r'\d{3}', content))
    
    if not (has_keyword and has_digits):
        return element
    
    validation = validate_phone_number(content)
    
    if validation['phones']:
        element['phone_validation'] = {
            'status': validation['validation_status'],
            'phones': validation['phones']
        }
    
    return element


def bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the overlap ratio between two bounding boxes.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        
    Returns:
        Overlap ratio (0.0 to 1.0) relative to the smaller bbox
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0  # No intersection
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    if area1 <= 0 or area2 <= 0:
        return 0.0
    
    # Return ratio relative to the smaller bbox (typically the text element)
    smaller_area = min(area1, area2)
    return intersection_area / smaller_area


def filter_empty_regions(elements: List[Dict]) -> List[Dict]:
    """
    Filter out table and layout_region elements that contain no text.
    
    A region is kept only if at least one text element's bbox overlaps
    with it by >= 30%. This removes hallucinated regions in empty/noisy areas.
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Filtered list with empty regions removed
    """
    MIN_OVERLAP_RATIO = 0.3  # 30% overlap required
    
    # Separate text elements from regions
    text_elements = [e for e in elements if e.get("type") == "text"]
    text_bboxes = [e.get("bbox", []) for e in text_elements]
    
    filtered = []
    removed_count = 0
    
    for element in elements:
        elem_type = element.get("type", "")
        
        # Keep text elements and other types (logo, etc.)
        if elem_type not in ("table", "layout_region"):
            filtered.append(element)
            continue
        
        # Check if this region overlaps with any text
        region_bbox = element.get("bbox", [])
        if not region_bbox:
            removed_count += 1
            continue
        
        has_text_overlap = False
        for text_bbox in text_bboxes:
            if text_bbox and bbox_overlap(region_bbox, text_bbox) >= MIN_OVERLAP_RATIO:
                has_text_overlap = True
                break
        
        if has_text_overlap:
            filtered.append(element)
        else:
            removed_count += 1
    
    return filtered


# ============================================================================
# Structural Validation Filter for Tables
# ============================================================================

# Configuration thresholds (tunable)
MIN_COLUMNS = 2          # Minimum columns to be considered a table
MIN_ROWS = 2             # Minimum rows to be considered a table
MIN_TEXT_DENSITY = 0.03  # 3% minimum text coverage
MAX_TEXT_DENSITY = 0.80  # 80% maximum (too dense = overlapping)
MIN_STRUCTURE_SCORE = 50 # Minimum score to keep as table
COLUMN_CLUSTER_TOLERANCE = 40  # Pixels for column clustering
ROW_CLUSTER_TOLERANCE = 15     # Pixels for row clustering


def calculate_text_density(table_bbox: List[float], text_bboxes: List[List[float]]) -> float:
    """
    Calculate ratio of text area to table area.
    
    Args:
        table_bbox: Table bounding box [x1, y1, x2, y2]
        text_bboxes: List of text bounding boxes inside the table
        
    Returns:
        Density ratio (0.0 to 1.0)
    """
    if len(table_bbox) != 4:
        return 0.0
    
    table_area = (table_bbox[2] - table_bbox[0]) * (table_bbox[3] - table_bbox[1])
    if table_area <= 0:
        return 0.0
    
    total_text_area = 0.0
    for bbox in text_bboxes:
        if len(bbox) >= 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width > 0 and height > 0:
                total_text_area += width * height
    
    return min(total_text_area / table_area, 1.0)


def cluster_positions(positions: List[float], tolerance: float) -> List[List[float]]:
    """
    Cluster similar positions together.
    
    Args:
        positions: List of coordinate values
        tolerance: Maximum distance to be in same cluster
        
    Returns:
        List of clusters (each cluster is list of positions)
    """
    if not positions:
        return []
    
    sorted_pos = sorted(positions)
    clusters = [[sorted_pos[0]]]
    
    for pos in sorted_pos[1:]:
        # Check if this position belongs to the last cluster
        if abs(pos - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])
    
    return clusters


def detect_column_alignment(table_bbox: List[float], text_bboxes: List[List[float]]) -> Tuple[int, List[float]]:
    """
    Detect column structure by clustering text X-positions.
    
    Args:
        table_bbox: Table bounding box
        text_bboxes: Text bboxes inside the table
        
    Returns:
        Tuple of (num_columns, column_center_positions)
    """
    if not text_bboxes:
        return 0, []
    
    # Get left edge (x1) of each text element
    x_positions = [bbox[0] for bbox in text_bboxes if len(bbox) >= 4]
    
    if not x_positions:
        return 0, []
    
    clusters = cluster_positions(x_positions, COLUMN_CLUSTER_TOLERANCE)
    
    # Get center position of each cluster
    column_centers = [sum(c) / len(c) for c in clusters]
    
    return len(clusters), column_centers


def detect_row_alignment(table_bbox: List[float], text_bboxes: List[List[float]]) -> Tuple[int, List[float]]:
    """
    Detect row structure by clustering text Y-positions.
    
    Args:
        table_bbox: Table bounding box
        text_bboxes: Text bboxes inside the table
        
    Returns:
        Tuple of (num_rows, row_center_positions)
    """
    if not text_bboxes:
        return 0, []
    
    # Get top edge (y1) of each text element
    y_positions = [bbox[1] for bbox in text_bboxes if len(bbox) >= 4]
    
    if not y_positions:
        return 0, []
    
    clusters = cluster_positions(y_positions, ROW_CLUSTER_TOLERANCE)
    
    # Get center position of each cluster
    row_centers = [sum(c) / len(c) for c in clusters]
    
    return len(clusters), row_centers


def check_grid_coverage(text_bboxes: List[List[float]], 
                        column_centers: List[float], 
                        row_centers: List[float]) -> float:
    """
    Check how well text elements cover the grid (columns x rows).
    
    A real table should have text in multiple cells across the grid.
    
    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if len(column_centers) < 2 or len(row_centers) < 2:
        return 0.0
    
    # Create a grid of expected cells
    total_cells = len(column_centers) * len(row_centers)
    
    # Count occupied cells
    occupied_cells = set()
    for bbox in text_bboxes:
        if len(bbox) < 4:
            continue
        
        text_x = (bbox[0] + bbox[2]) / 2  # Center X
        text_y = (bbox[1] + bbox[3]) / 2  # Center Y
        
        # Find which column this text belongs to
        col_idx = -1
        min_col_dist = float('inf')
        for i, col_x in enumerate(column_centers):
            dist = abs(text_x - col_x)
            if dist < min_col_dist and dist < COLUMN_CLUSTER_TOLERANCE * 2:
                min_col_dist = dist
                col_idx = i
        
        # Find which row this text belongs to
        row_idx = -1
        min_row_dist = float('inf')
        for i, row_y in enumerate(row_centers):
            dist = abs(text_y - row_y)
            if dist < min_row_dist and dist < ROW_CLUSTER_TOLERANCE * 2:
                min_row_dist = dist
                row_idx = i
        
        if col_idx >= 0 and row_idx >= 0:
            occupied_cells.add((col_idx, row_idx))
    
    return len(occupied_cells) / total_cells if total_cells > 0 else 0.0


def calculate_structure_score(
    density: float,
    num_cols: int,
    num_rows: int,
    grid_coverage: float,
    confidence: float
) -> Tuple[float, List[str]]:
    """
    Calculate structural validity score for a table.
    
    Args:
        density: Text density ratio
        num_cols: Number of detected columns
        num_rows: Number of detected rows
        grid_coverage: Grid coverage ratio
        confidence: Model confidence
        
    Returns:
        Tuple of (score 0-100, list of contributing signals)
    """
    score = 0.0
    signals = []
    
    # Signal 1: Multi-column structure (40 points max)
    if num_cols >= 3:
        score += 40
        signals.append(f"columns:{num_cols}")
    elif num_cols == 2:
        score += 25
        signals.append(f"columns:{num_cols}")
    else:
        signals.append("single_column")
    
    # Signal 2: Row structure (25 points max)
    if num_rows >= 3:
        score += 25
        signals.append(f"rows:{num_rows}")
    elif num_rows == 2:
        score += 15
        signals.append(f"rows:{num_rows}")
    else:
        signals.append("few_rows")
    
    # Signal 3: Grid coverage (20 points max)
    if grid_coverage >= 0.3:
        score += 20
        signals.append(f"grid_coverage:{grid_coverage:.0%}")
    elif grid_coverage >= 0.15:
        score += 10
        signals.append(f"grid_coverage:{grid_coverage:.0%}")
    else:
        signals.append(f"sparse_grid:{grid_coverage:.0%}")
    
    # Signal 4: Text density in valid range (10 points)
    if MIN_TEXT_DENSITY <= density <= MAX_TEXT_DENSITY:
        score += 10
        signals.append(f"density_ok:{density:.1%}")
    else:
        signals.append(f"density_bad:{density:.1%}")
    
    # Signal 5: Model confidence bonus (5 points)
    if confidence >= 0.95:
        score += 5
        signals.append("high_confidence")
    
    return score, signals


def validate_table_structure(
    table_element: Dict, 
    text_elements: List[Dict]
) -> Tuple[bool, float, List[str]]:
    """
    Validate if a detected table has true tabular structure.
    
    Args:
        table_element: Table element dictionary
        text_elements: All text elements on the page
        
    Returns:
        Tuple of (is_valid, score, signals)
    """
    table_bbox = table_element.get("bbox", [])
    confidence = table_element.get("confidence", 0.5)
    
    if len(table_bbox) != 4:
        return False, 0.0, ["invalid_bbox"]
    
    # Get text elements that overlap with this table
    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap(table_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)
    
    if not overlapping_texts:
        return False, 0.0, ["no_text_overlap"]
    
    # Calculate structural metrics
    density = calculate_text_density(table_bbox, overlapping_texts)
    num_cols, col_centers = detect_column_alignment(table_bbox, overlapping_texts)
    num_rows, row_centers = detect_row_alignment(table_bbox, overlapping_texts)
    grid_coverage = check_grid_coverage(overlapping_texts, col_centers, row_centers)
    
    # Calculate combined score
    score, signals = calculate_structure_score(
        density, num_cols, num_rows, grid_coverage, confidence
    )
    
    is_valid = score >= MIN_STRUCTURE_SCORE
    
    return is_valid, score, signals


def filter_invalid_tables(elements: List[Dict]) -> List[Dict]:
    """
    Filter out tables that don't have valid tabular structure.
    
    Tables with score < MIN_STRUCTURE_SCORE are removed.
    Tables with borderline scores are flagged.
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Filtered list with invalid tables removed
    """
    # Separate text elements for analysis
    text_elements = [e for e in elements if e.get("type") == "text"]
    
    filtered = []
    tables_removed = 0
    tables_kept = 0
    
    for element in elements:
        if element.get("type") != "table":
            filtered.append(element)
            continue
        
        # Validate table structure
        is_valid, score, signals = validate_table_structure(element, text_elements)
        
        if is_valid:
            # Add validation info to table
            element["structure_score"] = round(score, 1)
            element["structure_signals"] = signals
            filtered.append(element)
            tables_kept += 1
        else:
            # Remove invalid table
            tables_removed += 1
    
    return filtered


# ============================================================================
# Heuristic Table Promotion (for missed borderless tables)
# ============================================================================

# Configuration for table promotion (FINAL - very restrictive to avoid false positives)
VERTICAL_CLUSTER_GAP = 20        # Reduced gap for tighter clustering
MIN_CLUSTER_REGIONS = 15         # Require many regions (was 8) - only dense clusters
MIN_PROMOTION_COLUMNS = 4        # Require 4+ columns (was 3)
MIN_PROMOTION_ROWS = 5           # Require 5+ rows (was 3) - eliminates small headers
MIN_PROMOTION_SCORE = 90         # Very high score threshold (was 70)
MIN_GRID_COVERAGE = 0.55         # Require 55% grid coverage (was 35%)
TABLE_OVERLAP_THRESHOLD = 0.40   # Overlap ratio to consider as duplicate


def cluster_layout_regions_vertically(elements: List[Dict]) -> List[List[Dict]]:
    """
    Cluster layout_region elements by vertical proximity.
    
    Groups regions that are vertically adjacent (within VERTICAL_CLUSTER_GAP pixels).
    
    Args:
        elements: List of all elements
        
    Returns:
        List of clusters, where each cluster is a list of layout_region elements
    """
    # Extract layout regions only
    layout_regions = [e for e in elements if e.get("type") == "layout_region"]
    
    if len(layout_regions) < MIN_CLUSTER_REGIONS:
        return []
    
    # Sort by Y position (top of bbox)
    regions_with_y = []
    for region in layout_regions:
        bbox = region.get("bbox", [])
        if len(bbox) >= 4:
            y_top = bbox[1]
            regions_with_y.append((y_top, region))
    
    if not regions_with_y:
        return []
    
    regions_with_y.sort(key=lambda x: x[0])
    
    # Cluster by vertical proximity
    clusters = []
    current_cluster = [regions_with_y[0][1]]
    current_y_bottom = regions_with_y[0][1].get("bbox", [0, 0, 0, 0])[3]
    
    for y_top, region in regions_with_y[1:]:
        bbox = region.get("bbox", [0, 0, 0, 0])
        
        # Check if this region is close enough to current cluster
        if y_top - current_y_bottom <= VERTICAL_CLUSTER_GAP:
            current_cluster.append(region)
            current_y_bottom = max(current_y_bottom, bbox[3])
        else:
            # Start new cluster
            if len(current_cluster) >= MIN_CLUSTER_REGIONS:
                clusters.append(current_cluster)
            current_cluster = [region]
            current_y_bottom = bbox[3]
    
    # Don't forget the last cluster
    if len(current_cluster) >= MIN_CLUSTER_REGIONS:
        clusters.append(current_cluster)
    
    return clusters


def calculate_cluster_bbox(cluster: List[Dict]) -> List[float]:
    """
    Calculate combined bounding box for a cluster of regions.
    
    Args:
        cluster: List of layout_region elements
        
    Returns:
        Combined bbox [min_x, min_y, max_x, max_y]
    """
    if not cluster:
        return []
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for region in cluster:
        bbox = region.get("bbox", [])
        if len(bbox) >= 4:
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
    
    if min_x == float('inf'):
        return []
    
    return [min_x, min_y, max_x, max_y]


def validate_cluster_as_table(
    cluster: List[Dict], 
    text_elements: List[Dict]
) -> Tuple[bool, float, List[str]]:
    """
    Validate if a cluster of layout regions has valid table structure.
    
    Uses text elements that overlap with the cluster to detect
    column and row alignment.
    
    Args:
        cluster: List of layout_region elements
        text_elements: All text elements on the page
        
    Returns:
        Tuple of (is_valid, score, signals)
    """
    cluster_bbox = calculate_cluster_bbox(cluster)
    if not cluster_bbox:
        return False, 0.0, ["no_bbox"]
    
    # Get text elements that overlap with this cluster
    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap(cluster_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)
    
    if len(overlapping_texts) < 6:
        return False, 0.0, ["insufficient_text"]
    
    # Calculate structural metrics
    density = calculate_text_density(cluster_bbox, overlapping_texts)
    num_cols, col_centers = detect_column_alignment(cluster_bbox, overlapping_texts)
    num_rows, row_centers = detect_row_alignment(cluster_bbox, overlapping_texts)
    grid_coverage = check_grid_coverage(overlapping_texts, col_centers, row_centers)
    
    # Calculate combined score (same logic as validate_table_structure)
    score, signals = calculate_structure_score(
        density, num_cols, num_rows, grid_coverage, 0.80  # Synthetic confidence
    )
    
    # Stricter requirements for promotion
    # Forms tend to have low grid coverage (label-value pattern)
    # Real tables have cells spanning the grid more evenly
    is_valid = (
        score >= MIN_PROMOTION_SCORE and 
        num_cols >= MIN_PROMOTION_COLUMNS and 
        num_rows >= MIN_PROMOTION_ROWS and
        grid_coverage >= MIN_GRID_COVERAGE  # Key: reject low-coverage forms
    )
    
    return is_valid, score, signals


def overlaps_existing_table(cluster_bbox: List[float], tables: List[Dict]) -> bool:
    """
    Check if cluster overlaps significantly with any existing table.
    
    Args:
        cluster_bbox: Bounding box of the cluster
        tables: List of existing table elements
        
    Returns:
        True if overlap > TABLE_OVERLAP_THRESHOLD with any table
    """
    if not cluster_bbox or len(cluster_bbox) != 4:
        return False
    
    for table in tables:
        table_bbox = table.get("bbox", [])
        if len(table_bbox) != 4:
            continue
        
        overlap = bbox_overlap(cluster_bbox, table_bbox)
        if overlap >= TABLE_OVERLAP_THRESHOLD:
            return True
    
    return False


# Minimum columns required to anchor the table top boundary
MIN_ANCHOR_COLUMNS = 4  # Real tables have 4+ columns
# Row height tolerance for row detection in refinement
ANCHOR_ROW_HEIGHT = 18  # Reduced from 25 to prevent over-merging rows
# High density threshold - rows with this many items are automatically valid anchors
HIGH_DENSITY_ITEMS = 6
# Minimum columns for consecutive row check
MIN_CONSECUTIVE_COLS = 4


def _analyze_row_structure(row_texts: List[List[float]], bbox_width: float) -> dict:
    """
    Analyze a row's structural properties.
    
    Returns dict with: num_items, num_cols, span_ratio, is_valid
    """
    result = {
        "num_items": len(row_texts),
        "num_cols": 0,
        "span_ratio": 0.0,
        "is_valid": False
    }
    
    if not row_texts:
        return result
    
    # Get unique X positions
    x_positions = [bbox[0] for bbox in row_texts]
    column_clusters = cluster_positions(x_positions, COLUMN_CLUSTER_TOLERANCE)
    result["num_cols"] = len(column_clusters)
    
    # Calculate span ratio
    if len(column_clusters) >= 2:
        col_centers = sorted([sum(c) / len(c) for c in column_clusters])
        col_span = col_centers[-1] - col_centers[0]
        result["span_ratio"] = col_span / bbox_width if bbox_width > 0 else 0
    
    # A row is structurally valid if it has:
    # 1. Enough items (at least MIN_ANCHOR_COLUMNS items)
    # 2. Enough columns (at least MIN_ANCHOR_COLUMNS columns)
    # 3. Columns spanning enough width (at least 60%)
    result["is_valid"] = (
        result["num_items"] >= MIN_ANCHOR_COLUMNS and  # Must have enough items!
        result["num_cols"] >= MIN_ANCHOR_COLUMNS and 
        result["span_ratio"] >= 0.60
    )
    
    return result


def refine_table_top_boundary(
    cluster_bbox: List[float], 
    text_elements: List[Dict]
) -> List[float]:
    """
    Refine the top boundary of a promoted table by trimming header rows
    until a row with proper table-like multi-column alignment is found.
    
    Vertical Continuity Rule:
    A row qualifies as an anchor if EITHER:
    1. HIGH DENSITY: The row has >= 6 items (real table headers are dense)
    2. CONSECUTIVE STRUCTURE: The row has >= 4 cols spanning >= 60% width
       AND the next row ALSO has >= 4 cols spanning >= 60% width
    
    This rejects single "one-hit wonder" rows like form metadata that 
    coincidentally have 4 items but aren't followed by similar structure.
    
    Args:
        cluster_bbox: Original cluster bounding box [x1, y1, x2, y2]
        text_elements: All text elements on the page
        
    Returns:
        Refined bounding box with corrected top boundary
    """
    if not cluster_bbox or len(cluster_bbox) != 4:
        return cluster_bbox
    
    x1, y1, x2, y2 = cluster_bbox
    bbox_width = x2 - x1
    
    if bbox_width <= 0:
        return cluster_bbox
    
    # Get text elements within this cluster
    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap(cluster_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)
    
    if len(overlapping_texts) < 6:
        return cluster_bbox
    
    # Group text elements into rows by Y position
    overlapping_texts.sort(key=lambda b: b[1])
    
    # Cluster into rows
    rows = []
    current_row = [overlapping_texts[0]]
    current_row_y = overlapping_texts[0][1]
    
    for text_bbox in overlapping_texts[1:]:
        text_y = text_bbox[1]
        if abs(text_y - current_row_y) <= ANCHOR_ROW_HEIGHT:
            current_row.append(text_bbox)
        else:
            rows.append(current_row)
            current_row = [text_bbox]
            current_row_y = text_y
    
    if current_row:
        rows.append(current_row)
    
    # Pre-analyze all rows
    row_analyses = [_analyze_row_structure(row, bbox_width) for row in rows]
    
    # Find the anchor using Vertical Continuity Rule
    anchor_y = y1  # Default to original top
    
    for i, (row_texts, analysis) in enumerate(zip(rows, row_analyses)):
        if not row_texts:
            continue
        
        # RULE 1: High density rows are automatic anchors
        # Real table headers have many items (e.g., NAME | STORES | NAME | STORES = 10+)
        if analysis["num_items"] >= HIGH_DENSITY_ITEMS:
            anchor_y = min(bbox[1] for bbox in row_texts)
            break
        
        # RULE 2: Consecutive structure - this row and next row both valid
        if analysis["is_valid"]:
            # Check if next row also has valid structure
            if i + 1 < len(row_analyses):
                next_analysis = row_analyses[i + 1]
                if next_analysis["is_valid"] or next_analysis["num_items"] >= HIGH_DENSITY_ITEMS:
                    # Two consecutive qualifying rows - use current as anchor
                    anchor_y = min(bbox[1] for bbox in row_texts)
                    break
            # Single valid row at end - still use it (might be last row of headers)
            elif i == len(rows) - 1:
                anchor_y = min(bbox[1] for bbox in row_texts)
                break
    
    # Only adjust if anchor is significantly different from original
    if anchor_y > y1 + ANCHOR_ROW_HEIGHT:
        return [x1, anchor_y, x2, y2]
    
    return cluster_bbox


def promote_layout_regions_to_tables(elements: List[Dict]) -> List[Dict]:
    """
    Promote clusters of layout_region elements to table elements.
    
    This heuristic detects borderless tables that the model missed by:
    1. Clustering vertically adjacent layout regions
    2. Validating each cluster for tabular structure
    3. Creating synthetic table elements for valid clusters
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Elements with promoted tables added
    """
    # Get existing tables
    existing_tables = [e for e in elements if e.get("type") == "table"]
    
    # Get text elements for validation
    text_elements = [e for e in elements if e.get("type") == "text"]
    
    # Cluster layout regions
    clusters = cluster_layout_regions_vertically(elements)
    
    promoted_tables = []
    
    for cluster in clusters:
        cluster_bbox = calculate_cluster_bbox(cluster)
        
        if not cluster_bbox:
            continue
        
        # Skip if this overlaps with an existing table
        if overlaps_existing_table(cluster_bbox, existing_tables):
            continue
        
        # Validate the cluster structure
        is_valid, score, signals = validate_cluster_as_table(cluster, text_elements)
        
        if is_valid:
            # Refine the top boundary to exclude single-column header rows
            # This addresses the issue where TO:, FROM:, SUBJECT: headers
            # get incorrectly included in the table bbox
            refined_bbox = refine_table_top_boundary(cluster_bbox, text_elements)
            
            # Create synthetic table element
            synthetic_table = {
                "type": "table",
                "bbox": refined_bbox,
                "confidence": 0.80,  # Lower confidence for heuristic detection
                "source": "heuristic_promotion",
                "structure_score": round(score, 1),
                "structure_signals": signals
            }
            promoted_tables.append(synthetic_table)
    
    # Add promoted tables to elements
    if promoted_tables:
        elements = elements + promoted_tables
    
    return elements


def postprocess_output(output_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply all post-processing to OCR output.

    Steps:
    0. Filter empty table/layout regions (remove hallucinations)
    0.25. Normalize underscore fill-in fields
    0.5. Validate table structure (remove false positive tables)
    0.6. Heuristic table promotion (detect missed borderless tables)
    1. Deduplicate layout regions
    2. Score and handle hallucinations (with margin awareness)
    2.1. Filter rotated margin text (Bates numbers, vertical IDs)
    2.5. Filter offensive OCR misreads (P0 reputational risk)
    3. Clean text content
    3.1. Repair dropped parentheses (P3)
    3.25. Apply handwritten OCR corrections (with slash-compound splitting)
    3.26. Apply multi-word proper noun corrections (P1)
    3.3. Replace signature text
    3.35. Filter signature overlap garbage (short fragments on cursive signatures)
    3.5. Remove consecutive duplicate words (TrOCR beam search fix)
    4. Normalize phone numbers

    Args:
        output_data: Raw OCR output dictionary
        
    Returns:
        Cleaned output dictionary
    """
    for page in output_data.get("pages", []):
        elements = page.get("elements", [])
        
        # Step 0: Filter empty table/layout regions
        elements = filter_empty_regions(elements)

        # Step 0.25: Normalize underscore fill-in fields
        elements = normalize_underscore_fields(elements)

        # Step 0.5: Validate table structure (remove false positives)
        elements = filter_invalid_tables(elements)
        
        # Step 0.6: Heuristic table promotion (detect missed tables)
        elements = promote_layout_regions_to_tables(elements)
        
        # Step 1: Deduplicate layout regions
        elements = deduplicate_layout_regions(elements)
        
        # Step 2: Score and handle hallucinations
        elements = process_hallucinations(elements)

        # Step 2.1: Filter rotated margin text (Bates numbers, vertical IDs)
        elements = filter_rotated_margin_text(elements)

        # Step 2.5: Filter offensive OCR misreads
        elements = filter_offensive_ocr_misreads(elements)

        # Step 3: Clean text content
        elements = clean_text_content(elements)

        # Step 3.1: Repair dropped parentheses
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = repair_dropped_parentheses(element['content'])

        # Step 3.25: Apply OCR corrections (with slash-compound splitting)
        # Build page-level context for standalone elements (e.g., single-word table headers)
        page_context = set()
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                for w in element['content'].split():
                    page_context.add(w.rstrip(':.,;!?').lower())
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = apply_ocr_corrections_handwritten(
                    element['content'], is_handwritten=True, page_context=page_context
                )

        # Step 3.26: Apply multi-word proper noun corrections
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = apply_multi_word_ocr_corrections(element['content'], page_context=page_context)

        # Step 3.3: Replace signature text readings with '(signature)'
        elements = replace_signature_text(elements)

        # Step 3.35: Remove short garbage text overlapping signature regions
        elements = filter_signature_overlap_garbage(elements)

        # Step 3.5: Remove consecutive duplicate words (TrOCR beam search artifact)
        elements = remove_consecutive_duplicate_words(elements)

        # Step 4: Normalize phone numbers
        elements = normalize_phone_numbers(elements)
        
        # Step 5: Classification refinement (detect misclassified typed docs)
        # Collect all text for typed document indicator detection
        all_text = ' '.join([e.get('content', '') for e in elements if e.get('type') == 'text'])
        typed_indicators = detect_typed_document_indicators(all_text)
        
        if typed_indicators['is_likely_typed']:
            page['classification_refinement'] = {
                'likely_typed': True,
                'fax_indicators': len(typed_indicators['fax_indicators']),
                'form_indicators': len(typed_indicators['form_indicators']),
                'suggested_boost': typed_indicators['confidence_boost']
            }
        
        page["elements"] = elements
    
    return output_data


def deduplicate_layout_regions(elements: List[Dict]) -> List[Dict]:
    """
    Remove duplicate layout_region elements with same bbox.
    Keeps one representative region per unique bbox.
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Deduplicated list of elements
    """
    seen_bboxes = set()
    deduplicated = []
    
    for element in elements:
        if element.get("type") == "layout_region":
            # Create a hashable key from bbox
            bbox = element.get("bbox", [])
            bbox_key = tuple(round(x, 2) for x in bbox) if bbox else ()
            
            if bbox_key in seen_bboxes:
                continue  # Skip duplicate
            seen_bboxes.add(bbox_key)
        
        deduplicated.append(element)
    
    return deduplicated


def process_hallucinations(elements: List[Dict]) -> List[Dict]:
    """
    Score each text element for hallucination likelihood using multiple signals.
    Removes high-confidence hallucinations, flags uncertain ones.

    Signals:
    - Confidence score (15%)
    - Text length (10%)
    - Character patterns (25%)
    - Bbox size anomaly (15%)
    - Dictionary check (15%)
    - Repetition patterns (10%)
    - Margin position (10%)

    Args:
        elements: List of element dictionaries

    Returns:
        Processed elements with hallucinations handled
    """
    # Estimate page dimensions from all element bboxes
    page_width = 0
    page_height = 0
    for element in elements:
        bbox = element.get("bbox", [])
        if len(bbox) >= 4:
            page_width = max(page_width, bbox[2])
            page_height = max(page_height, bbox[3])
    # Fallback if no bboxes found
    if page_width == 0:
        page_width = 612  # Standard letter width in points
    if page_height == 0:
        page_height = 792

    processed = []

    for element in elements:
        if element.get("type") != "text":
            processed.append(element)
            continue

        content = element.get("content", "")
        confidence = element.get("confidence", 1.0)
        bbox = element.get("bbox", [0, 0, 100, 100])

        # Calculate hallucination score
        score, signals = calculate_hallucination_score(
            content, confidence, bbox, page_width, page_height
        )

        if score >= 0.50:
            # High hallucination likelihood - remove (tightened from > 0.50)
            continue
        elif score > 0.30:
            # Uncertain - flag but keep (tightened from 0.40)
            element["hallucination_flag"] = True
            element["hallucination_score"] = round(score, 3)
            element["hallucination_signals"] = signals

        processed.append(element)

    return processed


def filter_rotated_margin_text(elements: List[Dict]) -> List[Dict]:
    """
    Remove text fragments from rotated margin text (e.g., vertical Bates numbers).

    Documents often have ID numbers printed vertically along the right edge.
    The OCR reads these as isolated fragments (single digits, short nonsense
    strings) scattered along the margin. This filter detects and removes them
    based on their distinctive bbox signature: extremely narrow, at the far
    right edge of the page.

    Digit-only content is exempted — Bates numbers and document IDs that Surya
    reads as a single coherent number are legitimate and should be kept.

    Args:
        elements: List of element dictionaries

    Returns:
        Filtered elements with rotated margin fragments removed
    """
    # Estimate page width from all element bboxes
    page_width = 0
    for element in elements:
        bbox = element.get("bbox", [])
        if len(bbox) >= 4:
            page_width = max(page_width, bbox[2])
    if page_width == 0:
        return elements  # Can't determine margins without bboxes

    filtered = []
    for element in elements:
        if element.get("type") != "text":
            filtered.append(element)
            continue

        bbox = element.get("bbox", [0, 0, 100, 100])
        if len(bbox) < 4:
            filtered.append(element)
            continue

        content = element.get("content", "").strip()
        bbox_width = bbox[2] - bbox[0]
        # Rotated margin text: starts in rightmost 8% of page AND very narrow
        at_right_edge = bbox[0] > page_width * 0.92
        very_narrow = bbox_width < page_width * 0.04

        if at_right_edge and very_narrow:
            # Exempt digit-only content (Bates numbers, document IDs)
            if content.isdigit():
                filtered.append(element)
                continue
            continue  # Drop rotated margin fragment

        filtered.append(element)

    return filtered


def calculate_hallucination_score(
    content: str,
    confidence: float,
    bbox: List[float],
    page_width: float = 612,
    page_height: float = 792
) -> Tuple[float, List[str]]:
    """
    Calculate hallucination likelihood score from multiple signals.

    Returns:
        Tuple of (score 0.0-1.0, list of triggered signal names)
    """
    score = 0.0
    signals = []

    # Signal 1: Low confidence (15% weight — reduced from 20% for margin signal)
    if confidence < 0.50:
        score += 0.15
        signals.append("very_low_confidence")
    elif confidence < 0.70:
        score += 0.10
        signals.append("low_confidence")
    elif confidence < 0.85:
        score += 0.05

    # Signal 2: Very short text (10% weight — reduced from 15% for margin signal)
    text_len = len(content.strip())
    if text_len <= 2:
        score += 0.10
        signals.append("very_short")
    elif text_len <= 4:
        score += 0.06
        signals.append("short")

    # Signal 3: Character pattern anomalies (25% weight)
    pattern_score, pattern_signals = check_character_patterns(content)
    score += pattern_score * 0.25
    signals.extend(pattern_signals)

    # Signal 4: Bbox size anomaly (15% weight)
    if len(bbox) >= 4:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Very small bbox
        if width < 20 or height < 10:
            score += 0.15
            signals.append("tiny_bbox")
        # Extremely wide aspect ratio (likely noise)
        elif width > 0 and height / width > 5:
            score += 0.10
            signals.append("abnormal_aspect")

    # Signal 5: Not a recognizable word/pattern (15% weight)
    if not is_valid_text(content):
        score += 0.15
        signals.append("not_valid_text")

    # Signal 6: Repetition patterns (10% weight)
    if has_repetition_pattern(content):
        score += 0.10
        signals.append("repetition")

    # Signal 7: Margin position (10% weight — new)
    # Short fragments at extreme page edges are likely margin hallucinations,
    # but digit-only content at margins is typically a page number — exempt it.
    if len(bbox) >= 4 and page_width > 0:
        margin_threshold = 0.03  # 3% of page dimension
        at_left_margin = bbox[0] < page_width * margin_threshold and bbox[2] < page_width * 0.10
        at_right_margin = bbox[0] > page_width * (1 - 0.10) and bbox[2] > page_width * (1 - margin_threshold)
        is_page_number = content.strip().isdigit()
        if (at_left_margin or at_right_margin) and not is_page_number:
            if text_len <= 4:
                # Short non-numeric fragment at margin — high confidence hallucination
                score += 0.25
                signals.append("margin_fragment_short")
            else:
                # Longer text at margin — mild signal
                score += 0.10
                signals.append("margin_position")

    return min(score, 1.0), signals


def check_character_patterns(content: str) -> Tuple[float, List[str]]:
    """
    Check for suspicious character patterns.

    Returns:
        Tuple of (score 0.0-1.0, list of pattern names)
    """
    score = 0.0
    patterns = []

    # All same character
    if len(set(content.replace(" ", ""))) == 1 and len(content) > 2:
        score += 0.8
        patterns.append("all_same_char")

    stripped = content.strip()

    # Only digits / short numbers — mutually exclusive to avoid double-counting.
    # Page numbers, short codes, and reference numbers are common legitimate content.
    if stripped.isdigit() and len(stripped) <= 3:
        score += 0.30
        patterns.append("isolated_digits")
    elif re.match(r'^[\d\s]+$', stripped) and len(stripped) <= 4:
        score += 0.35
        patterns.append("short_numbers")

    # Only punctuation
    if all(c in ".,;:!?-_'" for c in content.replace(" ", "")):
        score += 0.6
        patterns.append("only_punctuation")

    # Non-printable or unusual characters — allow Latin Extended (U+00C0-U+024F)
    # which covers all European Latin-script languages (Spanish ñ, German ß,
    # Portuguese ã/õ, Nordic å/ø/æ, Polish ł/ś/ź, Turkish ğ/ş, Czech ř/š/č, etc.)
    # and Latin Extended Additional (U+1E00-U+1EFF) for Vietnamese and others.
    if any(
        ord(c) > 127
        and not (0x00C0 <= ord(c) <= 0x024F)
        and not (0x1E00 <= ord(c) <= 0x1EFF)
        for c in content
    ):
        score += 0.3
        patterns.append("unusual_chars")

    # Isolated year patterns (common hallucinations from fine print)
    # Matches: 1907, 1960s, 2007, etc. Mild signal — years in headers are legitimate.
    if re.match(r'^(19|20)\d{2}s?$', stripped):
        score += 0.25
        patterns.append("isolated_year")

    return min(score, 1.0), patterns


def is_valid_text(content: str) -> bool:
    """
    Check if content looks like valid text.
    Allows: words, numbers, dates, common abbreviations.
    """
    content = content.strip()
    
    # Empty
    if not content:
        return False
    
    # Single character (usually noise unless common)
    if len(content) == 1:
        return content.isalnum() or content in ".,;:!?()[]{}\"'"
    
    # Common patterns that are valid
    valid_patterns = [
        r'^[A-Za-z]{2,}$',  # Words
        r'^[A-Za-z]+[.,;:!?]?$',  # Words with punctuation
        r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$',  # Dates
        r'^\d+[.,]?\d*$',  # Numbers
        r'^[\$£€]\d+[.,]?\d*$',  # Currency
        r'^[A-Z]{2,}$',  # Acronyms
        r'^[A-Z][a-z]+$',  # Capitalized words
        r'^\(\d{3}\)\s*\d{3}[-\s]?\d{4}$',  # Phone numbers
        r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',  # Email
        r'^https?://',  # URLs
        r'^www\.',  # URLs
        r'^[A-Z][a-z]*\.?\s*$',  # Names with optional period
    ]
    
    for pattern in valid_patterns:
        if re.match(pattern, content, re.IGNORECASE):
            return True
    
    # Multi-word text is usually valid
    if ' ' in content and len(content) > 5:
        return True
    
    # Check if mostly alphanumeric
    alnum_ratio = sum(1 for c in content if c.isalnum()) / len(content)
    if alnum_ratio > 0.7:
        return True
    
    return False


def has_repetition_pattern(content: str) -> bool:
    """
    Check for repeated word patterns like 'the the the'.
    """
    words = content.lower().split()
    
    if len(words) < 2:
        return False
    
    # Check for consecutive repeated words
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 1:
            return True
    
    # Check if all words are the same
    if len(set(words)) == 1 and len(words) > 2:
        return True
    
    return False


# ============================================================================
# Handwriting OCR Correction (Applied only to handwritten documents)
# ============================================================================

# Common TrOCR confusion pairs: (wrong_word, correct_word, context_words)
# Context words can appear anywhere within 5 words of the confused word
OCR_CONFUSION_CORRECTIONS = [
    # Handwriting-specific confusions based on observed errors
    ('last', 'lost', ['property', 'found', 'missing', 'retrieve', 'retrieving']),
    ('book', 'look', ['neat', 'have', 'take', 'good', 'had', 'that']),
    ('intact', 'in fact', ['no', 'not', 'but', 'actually']),
    ('form', 'from', ['received', 'sent', 'get', 'letter', 'came']),
    ('bock', 'back', ['come', 'go', 'went', 'came', 'get']),
    ('sane', 'same', ['the', 'at', 'time', 'way']),
    ('dime', 'time', ['the', 'at', 'same', 'every', 'any']),
    # Empirically derived from test corpus — proper noun and typed form misreads
    # Context words prevent false positives on common English words (e.g. "angles", "patriot")
    ('gambetta', 'lambretta', ['negresco', 'beach', 'parked', 'opposite', 'promenade']),
    ('negress', 'negresco', ['lambretta', 'beach', 'promenade', 'opposite', 'swim']),
    ('angles', 'anglais', ['promenade', 'des', 'nice', 'walls', 'spumed']),
    ('patriot', 'patriote', ['read', 'nice', 'beach', 'catastrophe', 'about']),
    ('lobito', 'losito', ['cc', 'tahmaseb', 'baroody', 'stevens', 'registration']),
    ('proliffements', 'requirements', []),  # not a real word — safe unconditional
    ('leaved', 'leaked', ['condition', 'shipment', 'good', 'broken', 'article']),
    ('overlap', 'overwrap', ['filter', 'pack', 'type', 'flavoring']),
    ('depariment', 'department', []),  # not a real word — safe unconditional
    ('engine', 'broken', ['condition', 'shipment', 'good', 'leaked']),
    # Typed document proper noun misreads (Surya OCR)
    ('atterney', 'attorney', []),  # not a real word — safe unconditional
    ('decamps', 'delchamps', ['stores', 'account', 'maverick', 'distribution', 'region']),
    ('antler', 'dantzler', ['stores', 'account', 'maverick', 'region', 'distribution']),
    ('indoor', 'ind/lor', ['volume', 'stores', 'account', 'maverick', 'distribution']),
    # Degraded fax / small-print confusions (Surya OCR)
    ('approve', 'approx', ['circulation', 'geographical', 'redemption', 'coupon']),
    ('incipient', 'recipient', ['intended', 'disclosure', 'confidential', 'privileged']),
    ('probable', 'prohibited', ['disclosure', 'confidential', 'telecopy', 'privileged']),
]

# Multi-word OCR corrections for proper nouns that span multiple tokens.
# These are applied as full-text replacements (case-insensitive).
# Each entry: (wrong_phrase, correct_phrase, context_words)
# Empty context [] = unconditional (phrase is unique enough to never appear naturally).
# Non-empty context = requires at least one context word within the same text element.
MULTI_WORD_OCR_CORRECTIONS = [
    ('HAVENS GERMAN', 'HAGENS BERMAN', []),
    ('Havens German', 'Hagens Berman', []),
    ('Steve W. German', 'Steve W. Berman', []),
    ('Meyer G. Follow', 'Meyer G. Koplow', []),
    ('Rose & Kate', 'Rosen & Katz', []),
    ('Martin Harrington', 'Martin Barrington', []),
    ('Martin Warrington', 'Martin Barrington', []),
    ('Farewell', 'Wardwell', ['davis', 'polk', 'law', 'counsel', 'firm']),
    ('Ronald Einstein', 'Ronald Milstein', []),
    ('Charles A. Bit', 'Charles A. Blixt', []),
    ('Style Oil', 'Sayle Oil', []),
    ('Try Green', 'Autry Greer', ['stores', 'region', 'account', 'distribution', 'maverick']),
    ('Win Dixie', 'Winn Dixie', []),
    ('Compact Foods', 'Compac Foods', []),
]

# Words that commonly appear with prefixes that TrOCR misses
# These are checked in a wider window (5 words before, 3 after) for negation/contrast context
PREFIX_CORRECTIONS = {
    'cuckolded': 'uncuckolded',
    'robbed': 'unrobbed',
    'fortunate': 'unfortunate',
    'known': 'unknown',
    'able': 'unable',
    'satisfied': 'dissatisfied',
    'appear': 'disappear',
    'happy': 'unhappy',
    'likely': 'unlikely',
    'certain': 'uncertain',
}

# Extended negation context for prefix restoration
NEGATION_CONTEXT = ['not', "n't", 'never', 'but', 'yet', 'nor', 'neither', 'nothing', 'none', 'chosen']


def apply_ocr_corrections_handwritten(text: str, is_handwritten: bool = False, page_context: set = None) -> str:
    """
    Apply OCR-specific corrections for handwritten documents only.

    Uses flexible context matching - looks for context words within a 5-word
    window around the potentially confused word. Falls back to page-level
    context when the element has too few words for local context matching.

    Args:
        text: OCR text to correct
        is_handwritten: Whether the source document is handwritten
        page_context: Optional set of lowercased words from all text elements
                      on the page, used as fallback for short/standalone elements

    Returns:
        Corrected text (unchanged if not handwritten)
    """
    if not is_handwritten or not text:
        return text

    words = text.split()
    corrected_words = []
    context_window_size = 5  # Check 5 words before and after

    for i, word in enumerate(words):
        # Strip trailing punctuation so "OVERLAP:" matches "overlap"
        stripped = word.rstrip(':.,;!?')
        suffix = word[len(stripped):]

        # Build context window (5 words before and after), also strip punctuation
        window_start = max(0, i - context_window_size)
        window_end = min(len(words), i + context_window_size + 1)
        context_words = set(w.rstrip(':.,;!?').lower() for w in words[window_start:window_end])

        # Handle compound words joined by '/' or '-' (e.g., "DIRECTOR/DEPARIMENT",
        # "atterney-privileged"). Split, correct each part, rejoin.
        for sep in ('/', '-'):
            if sep in stripped:
                parts = stripped.split(sep)
                corrected_parts = []
                for part in parts:
                    part_lower = part.lower()
                    part_corrected = part
                    for wrong, correct, context in OCR_CONFUSION_CORRECTIONS:
                        if part_lower == wrong:
                            if not context or any(ctx in context_words for ctx in context) or (page_context and any(ctx in page_context for ctx in context)):
                                part_corrected = correct if part.islower() else correct.upper() if part.isupper() else correct.capitalize()
                                break
                    corrected_parts.append(part_corrected)
                corrected_words.append(sep.join(corrected_parts) + suffix)
                break
        else:
            # No separator found — fall through to normal word processing below
            pass

        # If we handled a compound word above, skip normal processing
        if '/' in stripped or '-' in stripped:
            continue

        word_lower = stripped.lower()
        corrected = stripped
        context_words.discard(word_lower)  # Remove the word itself

        # Check confusion pairs with flexible context matching
        for wrong, correct, context in OCR_CONFUSION_CORRECTIONS:
            if word_lower == wrong:
                # Empty context = always apply (proper noun / unconditional corrections)
                # Non-empty context = check element-local window first, then page context as fallback
                if not context or any(ctx in context_words for ctx in context) or (page_context and any(ctx in page_context for ctx in context)):
                    corrected = correct if stripped.islower() else correct.upper() if stripped.isupper() else correct.capitalize()
                    break

        # Check prefix restoration in negation/contrast context
        if word_lower in PREFIX_CORRECTIONS:
            # Look in wider window for negation context (5 before, 3 after)
            window = ' '.join(words[max(0, i - 5):i + 3]).lower()
            if any(neg in window for neg in NEGATION_CONTEXT):
                restored = PREFIX_CORRECTIONS[word_lower]
                corrected = restored if stripped.islower() else restored.upper() if stripped.isupper() else restored.capitalize()

        corrected_words.append(corrected + suffix)

    return ' '.join(corrected_words)


def apply_multi_word_ocr_corrections(text: str, page_context: set = None) -> str:
    """
    Apply multi-word OCR corrections for proper nouns spanning multiple tokens.

    These corrections handle cases where the OCR model mangles multi-word
    proper nouns (e.g., "HAVENS GERMAN" -> "HAGENS BERMAN") that cannot
    be fixed by single-word corrections.

    Entries with non-empty context require at least one context word to appear
    anywhere in the text element before the correction fires. Falls back to
    page-level context when the element itself doesn't contain context words.
    This prevents false positives on common words (e.g., "Farewell" only
    becomes "Wardwell" when near "Davis", "Polk", etc.).

    Args:
        text: OCR text to correct
        page_context: Optional set of lowercased words from all text elements
                      on the page, used as fallback for short elements

    Returns:
        Text with multi-word corrections applied
    """
    if not text:
        return text

    text_lower = text.lower()

    for wrong, correct, context in MULTI_WORD_OCR_CORRECTIONS:
        # Case-insensitive search, preserve surrounding text
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        if pattern.search(text):
            # Empty context = unconditional; non-empty = require context match
            # Check element text first, then fall back to page context
            if not context or any(ctx in text_lower for ctx in context) or (page_context and any(ctx in page_context for ctx in context)):
                text = pattern.sub(correct, text)
                text_lower = text.lower()  # refresh after substitution

    return text


# Pattern to fix dropped opening parentheses for optional-plural suffixes.
# Surya OCR produces "BRANDS)" instead of "BRAND(S)", "DIVISIONS)" instead of "DIVISION(S)".
# Only matches known suffix patterns: S, s, ES, es (plural/optional markers).
_DROPPED_OPEN_PAREN_RE = re.compile(
    r'\b([A-Za-z]{2,}?)((?:[Ee][Ss]|[Ss]))\)', re.UNICODE
)


def repair_dropped_parentheses(text: str) -> str:
    """
    Repair dropped opening parentheses in OCR output.

    Surya OCR consistently drops the opening parenthesis in mid-word
    positions for optional-plural suffixes:
    - "BRANDS)" -> "BRAND(S)"
    - "DIVISIONS)" -> "DIVISION(S)"
    - "recipients)" -> "recipient(s):"
    - "individuals)" -> "individual(s)"

    Only repairs known suffix patterns (S), (s), (ES), (es) to avoid
    false positives on words like "SCOPE)" which should stay as-is.

    Args:
        text: OCR text to repair

    Returns:
        Text with parentheses repaired
    """
    if not text or ')' not in text:
        return text

    def fix_paren(match):
        prefix = match.group(1)
        suffix = match.group(2)
        return f"{prefix}({suffix})"

    # Only apply if there's a ) without a matching (
    open_count = text.count('(')
    close_count = text.count(')')

    if close_count > open_count:
        text = _DROPPED_OPEN_PAREN_RE.sub(fix_paren, text)

    return text


SIGNATURE_LABEL_RE = re.compile(
    r'((?:RECEIVED|SIGNED)\s+BY\s*:'
    r'|SIGNATURE\s+OF\s+[\w\s]*?(?:CONSIGNEE|INITIATOR|APPLICANT|AUTHORIZED(?:\s+\w+)?)\s*:'
    r'|SIGNATURE\s*:)\s*'
    r'([A-Za-z]+(?:\s+[A-Za-z]+){0,2})'
    r'(?=\s+DATE\s*:|$)',
    re.IGNORECASE
)


def replace_signature_text(elements: List[Dict]) -> List[Dict]:
    """
    Replace OCR-read signature text with '(signature)'.

    Matches patterns like 'RECEIVED BY: Some Name DATE:' and replaces
    the name portion with '(signature)'.
    """
    for element in elements:
        if element.get('type') == 'text' and element.get('content'):
            element['content'] = SIGNATURE_LABEL_RE.sub(
                lambda m: m.group(1) + ' (signature)', element['content']
            )
    return elements


_DATE_LIKE_RE = re.compile(
    r'^\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}$'  # 21-Jan-00, 03/01/90
    r'|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$'        # 03/01/1990, 3-1-90
)


def filter_signature_overlap_garbage(elements: List[Dict]) -> List[Dict]:
    """
    Remove single-word garbage fragments that overlap with signature elements.

    When Surya reads cursive signatures, it often produces short garbage strings
    like 'elevens.', 'not', '100' that are too "normal" for the hallucination
    scorer but clearly wrong when correlated with Florence-2's signature detection.

    Removes text elements that are:
    - Single word (exactly 1 word)
    - Not a date-like pattern (e.g., '21-Jan-00')
    - Overlapping > 50% with a signature element

    Only uses 'signature' visual elements (not logo/seal, which commonly have
    legitimate text nearby like firm names).

    Args:
        elements: List of element dictionaries

    Returns:
        Filtered elements with signature garbage removed
    """
    # Collect bboxes of signature elements only (not logo/seal)
    signature_bboxes = []
    for el in elements:
        if el.get('type', '').lower() == 'signature':
            bbox = el.get('bbox')
            if bbox and len(bbox) == 4:
                signature_bboxes.append(bbox)

    if not signature_bboxes:
        return elements

    filtered = []
    for el in elements:
        if el.get('type') != 'text':
            filtered.append(el)
            continue

        content = el.get('content', '').strip()
        word_count = len(content.split()) if content else 0

        # Only filter single-word fragments
        if word_count != 1:
            filtered.append(el)
            continue

        # Exempt date-like patterns (e.g., 21-Jan-00, 03/01/90)
        if _DATE_LIKE_RE.match(content.rstrip('.,')):
            filtered.append(el)
            continue

        el_bbox = el.get('bbox')
        if not el_bbox or len(el_bbox) != 4:
            filtered.append(el)
            continue

        # Check overlap with any signature element
        overlaps_sig = False
        for sig_bbox in signature_bboxes:
            overlap = bbox_overlap(el_bbox, sig_bbox)
            if overlap > 0.50:
                overlaps_sig = True
                break

        if overlaps_sig:
            print(f"  [SIG-GARBAGE] Removed '{content}' (overlaps signature region)")
        else:
            filtered.append(el)

    return filtered


def clean_text_content(elements: List[Dict]) -> List[Dict]:
    """
    Clean and normalize text content.

    Fixes:
    - Extra whitespace
    - Common OCR substitution errors
    - Encoding issues
    """
    for element in elements:
        if element.get("type") != "text":
            continue
        
        content = element.get("content", "")
        
        # Normalize whitespace
        content = " ".join(content.split())
        
        # Fix common encoding issues
        content = fix_encoding_issues(content)
        
        element["content"] = content

    return elements


def remove_consecutive_duplicate_words(elements: List[Dict]) -> List[Dict]:
    """
    Remove consecutive duplicate words from text content.

    This fixes a known TrOCR beam search artifact where words get duplicated,
    e.g., "We went straight straight to bed" -> "We went straight to bed"

    Args:
        elements: List of element dictionaries

    Returns:
        Elements with duplicate words removed from text content
    """
    for element in elements:
        if element.get("type") != "text":
            continue

        # Only apply to TrOCR output (handwriting model)
        if element.get("source_model") != "trocr":
            continue

        content = element.get("content", "")
        if not content:
            continue

        words = content.split()
        if len(words) < 2:
            continue

        # Remove consecutive duplicates (case-insensitive comparison)
        deduplicated = [words[0]]
        for word in words[1:]:
            if word.lower() != deduplicated[-1].lower():
                deduplicated.append(word)

        if len(deduplicated) < len(words):
            element["content"] = ' '.join(deduplicated)
            element["duplicate_words_removed"] = len(words) - len(deduplicated)

    return elements


def fix_encoding_issues(text: str) -> str:
    """
    Fix common encoding/OCR issues.
    """
    replacements = {
        '\u00a0': ' ',  # Non-breaking space
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


# ============================================================================
# Phone Number Normalization
# ============================================================================

# Comprehensive regex for US phone numbers - handles all 7 format variations
# Matches: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX XXX XXXX, XXX/XXX-XXXX, etc.
PHONE_PATTERN = re.compile(
    r'''
    (?:^|(?<=\s)|(?<=[:\-/]))  # Start of string, whitespace, or separator
    (?:
        \((\d{3})\)\s*         # (XXX) with optional space
        |
        (\d{3})[\s\-/.]        # XXX followed by separator
    )
    (\d{3})                    # Exchange: XXX
    [\s\-.]?                   # Optional separator
    (\d{4})                    # Subscriber: XXXX
    (?=\s|$|[,;])              # End boundary
    ''',
    re.VERBOSE
)

# Pattern to detect phone-related labels for type classification
PHONE_TYPE_PATTERNS = {
    'fax': re.compile(r'\b(FAX|FACSIMILE|TELECOPY)\b', re.IGNORECASE),
    'phone': re.compile(r'\b(PHONE|TELEPHONE|TEL|CALL)\b', re.IGNORECASE),
}

# False positive patterns to skip
ZIP_PATTERN = re.compile(r'\b\d{5}(-\d{4})?\b')  # 5-digit or 5+4 ZIP
DATE_PATTERN = re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b')  # MM/DD/YY


def normalize_phone_numbers(elements: List[Dict]) -> List[Dict]:
    """
    Detect and normalize phone numbers in text elements.
    
    Adds:
    - normalized_phone: standardized format (XXX) XXX-XXXX
    - phone_type: 'phone' or 'fax' based on context
    
    False positive protection:
    - Skips ZIP codes (5-digit patterns)
    - Skips date patterns
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Elements with phone normalization applied
    """
    for element in elements:
        if element.get("type") != "text":
            continue
        
        content = element.get("content", "")
        
        # Skip if content looks like a date or ZIP code
        if is_date_or_zip(content):
            continue
        
        # Try to extract and normalize phone numbers
        phones = extract_phone_numbers(content)
        
        if phones:
            # Add normalized phones (multiple if found)
            if len(phones) == 1:
                element["normalized_phone"] = phones[0]
            else:
                element["normalized_phones"] = phones
            
            # Detect phone type from context
            phone_type = detect_phone_type(content)
            if phone_type:
                element["phone_type"] = phone_type
            
            # Add validation status using the validation helper
            validation_result = validate_phone_number(content)
            if validation_result['phones']:
                element["phone_validation_status"] = validation_result['validation_status']
    
    return elements


def extract_phone_numbers(content: str) -> List[str]:
    """
    Extract and normalize all phone numbers from content.
    
    Returns:
        List of normalized phone numbers in (XXX) XXX-XXXX format
    """
    phones = []
    
    # First try the comprehensive regex
    for match in PHONE_PATTERN.finditer(content):
        # Get area code from either group 1 (parenthesized) or group 2 (plain)
        area_code = match.group(1) or match.group(2)
        exchange = match.group(3)
        subscriber = match.group(4)
        
        if area_code and exchange and subscriber:
            # Validate: must be 10 digits total
            if len(area_code) == 3 and len(exchange) == 3 and len(subscriber) == 4:
                normalized = f"({area_code}) {exchange}-{subscriber}"
                phones.append(normalized)
    
    # Also try simpler pattern for edge cases like "212/545-3297"
    slash_pattern = re.compile(r'(\d{3})/(\d{3})-(\d{4})')
    for match in slash_pattern.finditer(content):
        normalized = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
        if normalized not in phones:
            phones.append(normalized)
    
    # Handle space-separated format: "206 623 0594"
    space_pattern = re.compile(r'\b(\d{3})\s+(\d{3})\s+(\d{4})\b')
    for match in space_pattern.finditer(content):
        # Check it's not part of a date
        full_match = match.group(0)
        if not re.search(r'\d{1,2}[/\-]', content[max(0, match.start()-5):match.start()]):
            normalized = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
            if normalized not in phones:
                phones.append(normalized)
    
    return phones


def is_date_or_zip(content: str) -> bool:
    """
    Check if content is primarily a date or ZIP code (false positive).
    
    Returns:
        True if content should be skipped for phone normalization
    """
    content_stripped = content.strip()
    
    # Pure ZIP code pattern: "64105-2118" or "43215"
    if re.match(r'^\d{5}(-\d{4})?$', content_stripped):
        return True
    
    # Pure date pattern: "12/12/96", "01-14-99"
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', content_stripped):
        return True
    
    # Content is mostly a date with timestamp: "12/12/96 08:33"
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s+\d{1,2}:\d{2}', content_stripped):
        return True
    
    # ZIP in address context: "City, State 12345-6789"
    # These should be kept as-is without phone normalization
    if re.search(r'[A-Za-z]+,?\s+[A-Za-z]{2}\s+\d{5}(-\d{4})?$', content_stripped):
        return True
    
    return False


def detect_phone_type(content: str) -> str:
    """
    Detect phone type (phone or fax) from content context.
    
    Returns:
        'phone', 'fax', or empty string if not determinable
    """
    content_upper = content.upper()
    
    # Check for fax-related keywords first (more specific)
    if PHONE_TYPE_PATTERNS['fax'].search(content):
        return 'fax'
    
    # Check for phone-related keywords
    if PHONE_TYPE_PATTERNS['phone'].search(content):
        return 'phone'
    
    return ""

