"""
Post-processing module for OCR output.

Implements multi-signal hallucination detection and output cleaning.
"""

import re
from typing import Dict, List, Any, Tuple


def postprocess_output(output_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply all post-processing to OCR output.
    
    Steps:
    1. Deduplicate layout regions
    2. Score and flag potential hallucinations
    3. Clean text content
    
    Args:
        output_data: Raw OCR output dictionary
        
    Returns:
        Cleaned output dictionary
    """
    for page in output_data.get("pages", []):
        elements = page.get("elements", [])
        
        # Step 1: Deduplicate layout regions
        elements = deduplicate_layout_regions(elements)
        
        # Step 2: Score and handle hallucinations
        elements = process_hallucinations(elements)
        
        # Step 3: Clean text content
        elements = clean_text_content(elements)
        
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
    - Confidence score (20%)
    - Text length (15%)
    - Character patterns (25%)
    - Bbox size anomaly (15%)
    - Dictionary check (15%)
    - Repetition patterns (10%)
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Processed elements with hallucinations handled
    """
    processed = []
    
    for element in elements:
        if element.get("type") != "text":
            processed.append(element)
            continue
        
        content = element.get("content", "")
        confidence = element.get("confidence", 1.0)
        bbox = element.get("bbox", [0, 0, 100, 100])
        
        # Calculate hallucination score
        score, signals = calculate_hallucination_score(content, confidence, bbox)
        
        if score > 0.70:
            # High hallucination likelihood - remove
            continue
        elif score > 0.40:
            # Uncertain - flag but keep
            element["hallucination_flag"] = True
            element["hallucination_score"] = round(score, 3)
            element["hallucination_signals"] = signals
        
        processed.append(element)
    
    return processed


def calculate_hallucination_score(
    content: str, 
    confidence: float, 
    bbox: List[float]
) -> Tuple[float, List[str]]:
    """
    Calculate hallucination likelihood score from multiple signals.
    
    Returns:
        Tuple of (score 0.0-1.0, list of triggered signal names)
    """
    score = 0.0
    signals = []
    
    # Signal 1: Low confidence (20% weight)
    if confidence < 0.50:
        score += 0.20
        signals.append("very_low_confidence")
    elif confidence < 0.70:
        score += 0.12
        signals.append("low_confidence")
    elif confidence < 0.85:
        score += 0.05
    
    # Signal 2: Very short text (15% weight)
    text_len = len(content.strip())
    if text_len <= 2:
        score += 0.15
        signals.append("very_short")
    elif text_len <= 4:
        score += 0.08
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
    
    # Only digits when surrounded by text context is suspicious
    if content.strip().isdigit() and len(content.strip()) <= 3:
        score += 0.4
        patterns.append("isolated_digits")
    
    # Only punctuation
    if all(c in ".,;:!?-_'" for c in content.replace(" ", "")):
        score += 0.6
        patterns.append("only_punctuation")
    
    # Non-printable or unusual characters
    if any(ord(c) > 127 and c not in "éèêëàâäùûüôöîïç" for c in content):
        score += 0.3
        patterns.append("unusual_chars")
    
    # Mostly numbers with random letters (like "160", "000")
    if re.match(r'^[\d\s]+$', content.strip()) and len(content.strip()) <= 4:
        score += 0.5
        patterns.append("short_numbers")
    
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
