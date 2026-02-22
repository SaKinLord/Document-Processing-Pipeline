"""
Phone number and date validation/normalization for OCR output.

Handles detection, validation, and normalization of phone numbers and dates
in OCR text elements.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ============================================================================
# Date Format Validation
# ============================================================================

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

            if format_name in ['MM/DD/YYYY', 'MM/DD/YY']:
                month = groups[0]
                day = groups[1]

                try:
                    month_int = int(month)
                    if month_int < 1 or month_int > 12:
                        date_info['issues'].append('invalid_month')
                        date_info['valid'] = False
                except ValueError:
                    date_info['issues'].append('non_numeric_month')
                    date_info['valid'] = False

                try:
                    day_int = int(day)
                    if day_int < 1 or day_int > 31:
                        date_info['issues'].append('invalid_day')
                        date_info['valid'] = False
                    if day_int == 0:
                        date_info['issues'].append('missing_digit')
                        date_info['valid'] = False
                except ValueError:
                    date_info['issues'].append('non_numeric_day')
                    date_info['valid'] = False

            if date_info['issues']:
                has_suspicious = True

            result['dates'].append(date_info)

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
# Phone Number Detection (single pattern set used by both validation and
# normalization to avoid silent disagreements between parallel systems)
# ============================================================================

# Ordered by specificity — more explicit formats first so they match before
# the loose 10-consecutive-digit fallback.
PHONE_PATTERNS = [
    r'\((\d{3})\)\s*(\d{3})[- ]?(\d{4})',          # (XXX) XXX-XXXX
    r'(\d{3})-(\d{3})-(\d{4})',                      # XXX-XXX-XXXX
    r'(\d{3})/(\d{3})-(\d{4})',                      # XXX/XXX-XXXX
    r'(\d{3})\s+(\d{3})\s+(\d{4})',                  # XXX XXX XXXX (space-separated)
    r'(\d{3})[\s\-/.](\d{3})[\s\-.]?(\d{4})',        # XXX.XXX.XXXX and mixed separators
    r'(?<!\d)(\d{3})(\d{3})(\d{4})(?!\d)',            # XXXXXXXXXX (10 consecutive digits)
]

FALSE_POSITIVE_PATTERNS = [
    r'\d{5}-\d{4}',              # ZIP+4 codes
    r'\d{1,2}/\d{1,2}/\d{2,4}', # Dates
    r'\d{4}/\d{2}/\d{2}',       # ISO dates
]


def _collect_false_positive_spans(text: str):
    """Return list of (start, end) spans that are dates or ZIP codes."""
    spans = []
    for fp_pattern in FALSE_POSITIVE_PATTERNS:
        for fp_match in re.finditer(fp_pattern, text):
            spans.append((fp_match.start(), fp_match.end()))
    return spans


def _find_phone_matches(text: str):
    """Find all phone matches in *text* using the shared PHONE_PATTERNS.

    Returns a list of dicts with ``raw``, ``area_code``, ``prefix``,
    ``line``, and ``normalized`` keys.  Duplicates (same normalized
    number) are suppressed.
    """
    if not text:
        return []

    fp_spans = _collect_false_positive_spans(text)

    # Also skip if the entire element is a date or ZIP
    text_stripped = text.strip()
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', text_stripped):
        return []

    seen_normalized = set()
    matches = []

    for pattern in PHONE_PATTERNS:
        for match in re.finditer(pattern, text):
            # Skip matches overlapping a false-positive span
            m_start, m_end = match.start(), match.end()
            if any(m_start < fp_end and m_end > fp_start
                   for fp_start, fp_end in fp_spans):
                continue

            groups = match.groups()
            if len(groups) < 3:
                continue

            area_code, prefix, line = groups[0], groups[1], groups[2]
            if not (area_code and prefix and line):
                continue

            normalized = f"({area_code}) {prefix}-{line}"
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)

            matches.append({
                'raw': match.group(0),
                'area_code': area_code,
                'prefix': prefix,
                'line': line,
                'normalized': normalized,
            })

    return matches


def validate_phone_number(text: str) -> dict:
    """
    Extract and validate phone numbers from OCR text.
    Does NOT auto-correct -- only flags issues for human review.
    """
    result = {
        'phones': [],
        'validation_status': 'none',
        'issues': []
    }

    if not text:
        return result

    for m in _find_phone_matches(text):
        area_code = m['area_code']
        prefix = m['prefix']
        line = m['line']

        issues = []

        if len(area_code) != 3:
            issues.append(f'area_code_length:{len(area_code)}')
        if len(prefix) != 3:
            issues.append(f'prefix_length:{len(prefix)}')
        if len(line) != 4:
            issues.append(f'line_length:{len(line)}')

        total_digits = len(area_code) + len(prefix) + len(line)
        if total_digits < 10:
            issues.append(f'missing_digits:{10 - total_digits}')
        elif total_digits > 10:
            issues.append(f'extra_digits:{total_digits - 10}')

        full_number = area_code + prefix + line
        if re.match(r'^(\d)\1{5,}$', full_number):
            issues.append('repeated_digits')

        phone_data = {
            'raw': m['raw'],
            'normalized': m['normalized'],
            'area_code': area_code,
            'prefix': prefix,
            'line': line,
            'total_digits': total_digits,
        }

        if issues:
            phone_data['issues'] = issues

        result['phones'].append(phone_data)

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
    """
    if element.get('type') != 'text':
        return element

    content = element.get('content', '')
    if not content:
        return element

    phone_keywords = ['phone', 'fax', 'tel', 'call']
    has_keyword = any(kw in content.lower() for kw in phone_keywords)
    # Also trigger on parenthesized area codes like "(555)" but not bare parens
    has_paren_digits = bool(re.search(r'\(\d{3}\)', content))
    has_digits = bool(re.search(r'\d{3}', content))

    if not ((has_keyword or has_paren_digits) and has_digits):
        return element

    validation = validate_phone_number(content)

    if validation['phones']:
        element['phone_validation'] = {
            'status': validation['validation_status'],
            'phones': validation['phones']
        }

    return element


# ============================================================================
# Phone Number Normalization
# ============================================================================

PHONE_TYPE_PATTERNS = {
    'fax': re.compile(r'\b(FAX|FACSIMILE|TELECOPY)\b', re.IGNORECASE),
    'phone': re.compile(r'\b(PHONE|TELEPHONE|TEL|CALL)\b', re.IGNORECASE),
}


def normalize_phone_numbers(elements: List[Dict]) -> List[Dict]:
    """
    Detect and normalize phone numbers in text elements.
    """
    for element in elements:
        if element.get("type") != "text":
            continue

        content = element.get("content", "")

        if is_date_or_zip(content):
            continue

        phones = extract_phone_numbers(content)

        if phones:
            if len(phones) == 1:
                element["normalized_phone"] = phones[0]
            else:
                element["normalized_phones"] = phones

            phone_type = detect_phone_type(content)
            if phone_type:
                element["phone_type"] = phone_type

            validation_result = validate_phone_number(content)
            if validation_result['phones']:
                element["phone_validation_status"] = validation_result['validation_status']

    return elements


def extract_phone_numbers(content: str) -> List[str]:
    """
    Extract and normalize all phone numbers from content.

    Uses the shared ``_find_phone_matches`` helper (same patterns as
    ``validate_phone_number``) so both functions always agree on what
    constitutes a phone number.
    """
    return [m['normalized'] for m in _find_phone_matches(content)]


def is_date_or_zip(content: str) -> bool:
    """
    Check if content is primarily a date or ZIP code (false positive).
    """
    content_stripped = content.strip()

    if re.match(r'^\d{5}(-\d{4})?$', content_stripped):
        return True

    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', content_stripped):
        return True

    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s+\d{1,2}:\d{2}', content_stripped):
        return True

    if re.search(r'[A-Za-z]+,?\s+[A-Za-z]{2}\s+\d{5}(-\d{4})?$', content_stripped):
        return True

    return False


def detect_phone_type(content: str) -> str:
    """
    Detect phone type (phone or fax) from content context.
    """
    if PHONE_TYPE_PATTERNS['fax'].search(content):
        return 'fax'

    if PHONE_TYPE_PATTERNS['phone'].search(content):
        return 'phone'

    return ""
