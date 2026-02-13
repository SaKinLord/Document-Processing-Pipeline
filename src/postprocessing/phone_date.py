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

    # Check for corrupted date patterns
    corrupted_patterns = [
        (r'(\d{3}),(\d{2})(?!\d)', 'possible_corrupted_date'),
        (r'(\d{1,2})/0/(\d{2})', 'missing_digit'),
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
# Phone Number Validation
# ============================================================================

PHONE_PATTERNS = [
    r'\((\d{3})\)\s*(\d{3})[- ]?(\d{4})',
    r'(\d{3})-(\d{3})-(\d{4})',
    r'(\d{3})/(\d{3})-(\d{4})',
    r'(\d{3})\s+(\d{3})[- ]?(\d{4})',
    r'(?<!\d)(\d{3})(\d{3})(\d{4})(?!\d)',
]

FALSE_POSITIVE_PATTERNS = [
    r'\d{5}-\d{4}',      # ZIP+4 codes
    r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
    r'\d{4}/\d{2}/\d{2}',        # ISO dates
]


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

    for fp_pattern in FALSE_POSITIVE_PATTERNS:
        if re.search(fp_pattern, text):
            return result

    for pattern in PHONE_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            groups = match.groups()
            if len(groups) >= 3:
                area_code = groups[0]
                prefix = groups[1]
                line = groups[2]

                issues = []

                if len(area_code) != 3:
                    issues.append(f'area_code_length:{len(area_code)}')
                if len(prefix) != 3:
                    issues.append(f'prefix_length:{len(prefix)}')
                if len(line) != 4:
                    issues.append(f'line_length:{len(line)}')

                total_digits = len(area_code) + len(prefix) + len(line)
                if total_digits < 10:
                    issues.append(f'missing_digits:{10-total_digits}')
                elif total_digits > 10:
                    issues.append(f'extra_digits:{total_digits-10}')

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


# ============================================================================
# Phone Number Normalization
# ============================================================================

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

PHONE_TYPE_PATTERNS = {
    'fax': re.compile(r'\b(FAX|FACSIMILE|TELECOPY)\b', re.IGNORECASE),
    'phone': re.compile(r'\b(PHONE|TELEPHONE|TEL|CALL)\b', re.IGNORECASE),
}

ZIP_PATTERN = re.compile(r'\b\d{5}(-\d{4})?\b')
DATE_PATTERN = re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b')


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
    """
    phones = []

    for match in PHONE_PATTERN.finditer(content):
        area_code = match.group(1) or match.group(2)
        exchange = match.group(3)
        subscriber = match.group(4)

        if area_code and exchange and subscriber:
            if len(area_code) == 3 and len(exchange) == 3 and len(subscriber) == 4:
                normalized = f"({area_code}) {exchange}-{subscriber}"
                phones.append(normalized)

    # Slash-separated format
    slash_pattern = re.compile(r'(\d{3})/(\d{3})-(\d{4})')
    for match in slash_pattern.finditer(content):
        normalized = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
        if normalized not in phones:
            phones.append(normalized)

    # Space-separated format
    space_pattern = re.compile(r'\b(\d{3})\s+(\d{3})\s+(\d{4})\b')
    for match in space_pattern.finditer(content):
        if not re.search(r'\d{1,2}[/\-]', content[max(0, match.start()-5):match.start()]):
            normalized = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
            if normalized not in phones:
                phones.append(normalized)

    return phones


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
