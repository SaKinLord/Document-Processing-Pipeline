"""
OCR Pipeline Automated Test Framework
Runs regression tests after each optimization phase.
Compares WER/CER against baseline and target thresholds.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Allow imports from the project root (parent of tests/)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flexible_evaluation import FlexibleEvaluator
from src.postprocessing.normalization import normalize_underscores

# Feature flag for flexible evaluation
FLEXIBLE_EVALUATION_ENABLED = True


def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    distance = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words) if ref_words else 0
    return wer


def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate."""
    ref_chars = reference.lower().replace(" ", "")
    hyp_chars = hypothesis.lower().replace(" ", "")
    distance = levenshtein_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars) if ref_chars else 0
    return cer


# Metadata patterns to filter out (these appear in images but not in ground truth)
METADATA_PATTERNS = [
    r'^name:\s*$',  # Empty name field (only when no content follows)
    r'^\s*$',  # Empty lines
]

def extract_ocr_text(json_path):
    """Extract all text content from OCR output JSON, filtering metadata."""
    import re
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_elements = []
    for page in data.get('pages', []):
        for element in page.get('elements', []):
            if element.get('type') == 'text':
                content = element.get('content', '')
                # Filter out metadata lines not in ground truth
                is_metadata = any(re.match(p, content.lower().strip()) for p in METADATA_PATTERNS)
                if not is_metadata:
                    text_elements.append(content)

    return normalize_underscores(' '.join(text_elements))


def get_doc_metadata(json_path):
    """Get document type and classification confidence from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pages = data.get('pages', [])
    if not pages:
        return 'unknown', 0
    doc_type = pages[0].get('document_type', 'unknown')
    doc_conf = pages[0].get('classification_confidence', 0)
    return doc_type, doc_conf


# Target thresholds (after all optimizations)
# Handwritten docs have higher tolerance since TrOCR has inherent variance
# Typed forms have higher tolerance due to blank fields and table structures
THRESHOLDS = {
    'handwritten': {'wer': 0.20, 'cer': 0.10},  # 20% WER, 10% CER max
    'typed': {'wer': 0.40, 'cer': 0.30},  # 40% WER, 30% CER for forms with tables
    'mixed': {'wer': 0.40, 'cer': 0.20},  # 40% WER, 20% CER for mixed content
}


def run_tests(verbose=True):
    """Run all accuracy tests and return pass/fail status."""
    base_dir = Path(__file__).parent.parent  # Go up from tests/ to project root
    ground_truth_path = base_dir / 'ground_truth.json'
    output_dir = base_dir / 'output'

    if not ground_truth_path.exists():
        print("ERROR: ground_truth.json not found")
        return False, {}

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # Initialize flexible evaluator
    evaluator = FlexibleEvaluator(ignore_punctuation=True) if FLEXIBLE_EVALUATION_ENABLED else None

    results = []
    all_passed = True

    for doc_id, gt_data in ground_truth.items():
        json_file = output_dir / f"{doc_id}.json"

        if not json_file.exists():
            if verbose:
                print(f"  SKIP: {doc_id} - output file not found")
            continue

        ocr_text = extract_ocr_text(json_file)
        gt_text = normalize_underscores(gt_data['ground_truth'])
        doc_type, doc_conf = get_doc_metadata(json_file)

        # Calculate strict metrics (original behavior)
        wer = calculate_wer(gt_text, ocr_text)
        cer = calculate_cer(gt_text, ocr_text)

        # Calculate flexible metrics if enabled
        flexible_wer = None
        flexible_cer = None
        formatting_errors = 0
        content_errors = 0

        if evaluator:
            eval_result = evaluator.evaluate(gt_text, ocr_text)
            flexible_wer = eval_result.flexible_wer
            flexible_cer = eval_result.flexible_cer
            formatting_errors = eval_result.formatting_errors
            content_errors = eval_result.content_errors

        # Check against thresholds (using flexible metrics for pass/fail)
        # Use ground truth document_type for thresholds — the test should measure
        # OCR accuracy against the actual document content, not the pipeline's
        # classification which may vary across environments
        gt_doc_type = gt_data.get('document_type', doc_type)
        threshold = THRESHOLDS.get(gt_doc_type, THRESHOLDS['mixed'])
        effective_wer = flexible_wer if flexible_wer is not None else wer
        effective_cer = flexible_cer if flexible_cer is not None else cer
        wer_pass = effective_wer <= threshold['wer']
        cer_pass = effective_cer <= threshold['cer']
        passed = wer_pass and cer_pass

        if not passed:
            all_passed = False

        result_entry = {
            'doc_id': doc_id,
            'doc_type': doc_type,
            'confidence': doc_conf,
            'wer': wer,
            'cer': cer,
            'wer_threshold': threshold['wer'],
            'cer_threshold': threshold['cer'],
            'wer_pass': wer_pass,
            'cer_pass': cer_pass,
            'passed': passed
        }

        # Add flexible metrics if available
        if flexible_wer is not None:
            result_entry['flexible_wer'] = flexible_wer
            result_entry['flexible_cer'] = flexible_cer
            result_entry['formatting_errors'] = formatting_errors
            result_entry['content_errors'] = content_errors

        results.append(result_entry)

        if verbose:
            status = "PASS" if passed else "FAIL"
            if FLEXIBLE_EVALUATION_ENABLED and flexible_wer is not None:
                print(f"{status} {doc_id:<12} {doc_type:<12} WER:{wer*100:>5.1f}% (flex:{flexible_wer*100:>5.1f}%) CER:{cer*100:>5.1f}%")
            else:
                print(f"{status} {doc_id:<12} {doc_type:<12} WER:{wer*100:>5.1f}% CER:{cer*100:>5.1f}%")
    
    # Min-doc guard: fail if no results or fewer than half of ground truth docs tested
    if len(results) == 0:
        if verbose:
            print("FAIL: No output files found — 0 documents tested")
        return False, {}
    if len(results) < len(ground_truth) / 2:
        if verbose:
            print(f"FAIL: Only {len(results)}/{len(ground_truth)} ground truth docs tested (minimum: {len(ground_truth) // 2 + 1})")
        return False, {}

    # Summary
    if verbose and results:
        passed_count = sum(1 for r in results if r['passed'])
        print()
        print("=" * 60)
        print(f"RESULTS: {passed_count}/{len(results)} tests passed")

        avg_wer = sum(r['wer'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        print(f"Average WER (strict): {avg_wer*100:.1f}% | Average CER (strict): {avg_cer*100:.1f}%")

        # Show flexible metrics if available
        if FLEXIBLE_EVALUATION_ENABLED and results[0].get('flexible_wer') is not None:
            avg_flex_wer = sum(r.get('flexible_wer', 0) for r in results) / len(results)
            avg_flex_cer = sum(r.get('flexible_cer', 0) for r in results) / len(results)
            total_formatting = sum(r.get('formatting_errors', 0) for r in results)
            total_content = sum(r.get('content_errors', 0) for r in results)
            print(f"Average WER (flexible): {avg_flex_wer*100:.1f}% | Average CER (flexible): {avg_flex_cer*100:.1f}%")
            print(f"Error breakdown: {total_formatting} formatting, {total_content} content")
        print("=" * 60)

    # Build return data
    return_data = {
        'timestamp': datetime.now().isoformat(),
        'total': len(results),
        'passed': sum(1 for r in results if r['passed']),
        'avg_wer': sum(r['wer'] for r in results) / len(results) if results else 0,
        'avg_cer': sum(r['cer'] for r in results) / len(results) if results else 0,
        'results': results
    }

    # Add flexible averages if available
    if FLEXIBLE_EVALUATION_ENABLED and results and results[0].get('flexible_wer') is not None:
        return_data['avg_flexible_wer'] = sum(r.get('flexible_wer', 0) for r in results) / len(results)
        return_data['avg_flexible_cer'] = sum(r.get('flexible_cer', 0) for r in results) / len(results)

    return all_passed, return_data


def save_baseline(results, filename='test_baseline.json'):
    """Save current results as baseline for future comparison."""
    base_dir = Path(__file__).parent
    with open(base_dir / filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Baseline saved to {filename}")


def compare_to_baseline(current_results, baseline_file='test_baseline.json'):
    """Compare current results to baseline and show improvements."""
    base_dir = Path(__file__).parent
    baseline_path = base_dir / baseline_file
    
    if not baseline_path.exists():
        print("No baseline found. Run with --save-baseline first.")
        return
    
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    
    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINE")
    print("=" * 60)
    
    wer_delta = (baseline['avg_wer'] - current_results['avg_wer']) * 100
    cer_delta = (baseline['avg_cer'] - current_results['avg_cer']) * 100
    
    wer_icon = "📈" if wer_delta > 0 else "📉" if wer_delta < 0 else "➡️"
    cer_icon = "📈" if cer_delta > 0 else "📉" if cer_delta < 0 else "➡️"
    
    print(f"WER: {baseline['avg_wer']*100:.1f}% → {current_results['avg_wer']*100:.1f}% {wer_icon} ({wer_delta:+.1f}%)")
    print(f"CER: {baseline['avg_cer']*100:.1f}% → {current_results['avg_cer']*100:.1f}% {cer_icon} ({cer_delta:+.1f}%)")
    print(f"Pass rate: {baseline['passed']}/{baseline['total']} → {current_results['passed']}/{current_results['total']}")


if __name__ == "__main__":
    print("=" * 60)
    print("OCR PIPELINE REGRESSION TEST")
    print("=" * 60)
    print()
    
    all_passed, results = run_tests(verbose=True)
    
    if '--save-baseline' in sys.argv:
        save_baseline(results)
    
    if '--compare' in sys.argv:
        compare_to_baseline(results)
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if all_passed else 1)
