"""
Text processing utilities: spell correction and text row clustering.
"""

import os
import re
import logging

from spellchecker import SpellChecker

logger = logging.getLogger(__name__)


class SpellCorrector:
    def __init__(self, domain_dict_path=None):
        self.spell = SpellChecker()
        self.domain_words = set()

        if domain_dict_path and os.path.exists(domain_dict_path):
            try:
                with open(domain_dict_path, 'r') as f:
                    content = f.read()
                    for line in content.splitlines():
                        word = line.strip()
                        if word:
                            self.spell.word_frequency.load_words([word])
                            self.domain_words.add(word.lower())
            except Exception as e:
                logger.warning("Could not load domain dictionary: %s", e)

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
            # Skip short words (<4 chars)
            if len(word) < 4:
                corrected_words.append(word)
                continue

            # Skip words with digits
            if any(char.isdigit() for char in word):
                corrected_words.append(word)
                continue

            # Clean punctuation for checking
            clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
            if not clean_word:
                corrected_words.append(word)
                continue

            # Skip pure acronyms (all uppercase, 2-5 chars)
            if clean_word.isupper() and 2 <= len(clean_word) <= 5:
                corrected_words.append(word)
                continue

            # Domain dictionary check (case-insensitive)
            if clean_word.lower() in self.domain_words:
                corrected_words.append(word)
                continue

            # Generic dictionary check
            if clean_word.lower() in self.spell:
                corrected_words.append(word)
                continue

            # Attempt correction
            try:
                candidates = self.spell.candidates(clean_word)
                if not candidates:
                    corrected_words.append(word)
                    continue

                best_candidate = self.spell.correction(clean_word)

                if best_candidate and best_candidate != clean_word:
                    final_word = self._preserve_case(clean_word, best_candidate)

                    prefix = word[:word.find(clean_word)]
                    suffix = word[word.find(clean_word)+len(clean_word):]
                    corrected_words.append(prefix + final_word + suffix)
                else:
                    corrected_words.append(word)
            except Exception:
                corrected_words.append(word)

        return " ".join(corrected_words)


def cluster_text_rows(elements, y_threshold=15):
    """
    Clusters text elements into rows based on Y-coordinate alignment.
    Sorts rows by Y, and elements within rows by X.
    Expected 'elements' is a list of dicts, each having a 'bbox' [x1, y1, x2, y2].
    """
    if not elements:
        return []

    sorted_elements = sorted(elements, key=lambda e: e['bbox'][1])

    rows = []
    current_row = []

    for elem in sorted_elements:
        if not current_row:
            current_row.append(elem)
            continue

        ref_elem = current_row[0]

        if abs(elem['bbox'][1] - ref_elem['bbox'][1]) < y_threshold:
            current_row.append(elem)
        else:
            current_row.sort(key=lambda e: e['bbox'][0])
            rows.append(current_row)
            current_row = [elem]

    if current_row:
        current_row.sort(key=lambda e: e['bbox'][0])
        rows.append(current_row)

    ordered_elements = []
    for row_idx, row in enumerate(rows):
        for elem in row:
            elem['row_id'] = row_idx
            ordered_elements.append(elem)

    return ordered_elements
