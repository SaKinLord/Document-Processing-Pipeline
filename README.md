# Document Processing Pipeline (Hybrid Intelligent System)

A local pipeline for extracting structured data from complex, multi-format documents. It uses a **Hybrid Architecture** that dynamically switches between specialized models based on document content (Typed, Handwritten, or Mixed), with **per-line handwriting routing** that detects and re-routes individual handwritten lines on otherwise typed pages.

## Key Features

*   **Intelligent 8-Feature Classification:** Automatically detects document type using stroke width variance, line regularity, contour angle variance, edge density, form structure detection, character uniformity, signature region isolation, and fax/letterhead header detection. Includes ruled paper handwriting detection and sparse form safeguards.
*   **Per-Line Handwriting Routing:** Florence-2 phrase grounding detects signature and handwritten text regions independently before OCR. Lines overlapping these regions are forced into TrOCR ensemble regardless of page-level classification, with a strict 0.90 confidence gate for signature regions and a normal confidence gate for handwriting regions. This catches handwritten fill-ins, names, and annotations on otherwise typed pages — even when the page-level classifier misidentifies the document type.
*   **Multi-Model Intelligence:**
    *   **Surya OCR:** High-precision text detection and recognition for typed documents
    *   **TrOCR:** Specialized handwriting recognition with beam search (4 beams), 12px bbox padding for descenders, and punctuation spacing normalization
    *   **Table Transformer (DETR):** Table detection and structural extraction with validation scoring
    *   **Florence-2 (VLM):** Vision-language model for object detection, captioning, and phrase grounding (signatures, handwriting, logos, seals)
*   **Multi-Stage Post-Processing:**
    *   7-signal hallucination detection and removal
    *   Generalizable non-word OCR correction (spell checker + OCR confusion matrix)
    *   Decimal-dash confusion repair with page-level context detection
    *   Parenthesis repair for Surya artifacts
    *   Signature text replacement and garbage filtering
    *   Structural table validation and cell-level content extraction
    *   Phone number normalization and date validation
    *   Rotated margin text filtering (vertical Bates numbers)
*   **Robust Pre-processing:** Adaptive denoising and deskewing via OpenCV with resolution-adaptive parameters

## Accuracy & Performance

Based on regression testing with 32 ground truth documents (typed, handwritten, and mixed):

| Metric | Value |
|--------|-------|
| Regression Tests | **30/32 passing** |
| Average Flex WER | **~11.3%** |
| Average Flex CER | **~7.9%** |

Flexible evaluation ignores punctuation and formatting differences. Pass/fail requires **both** flex WER and flex CER to be within thresholds.

**Per-document thresholds:**

| Document Type | Max WER | Max CER |
|---------------|---------|---------|
| Handwritten | 20% | 10% |
| Typed | 40% | 30% |
| Mixed | 40% | 20% |

## Architecture

```
                          INGESTION
                       Load Image/PDF
                              |
                              v
                        PRE-PROCESSING
                  Denoise + Deskew + Resize
                              |
                              v
                     8-FEATURE CLASSIFIER
    Stroke Variance | Line Regularity | Angle Variance
    Edge Density | Form Structure | Character Uniformity
         Signature Isolation | Fax Header
                              |
                              v
                  FLORENCE-2 PHRASE GROUNDING
              Detect "signature" + "handwritten text"
              regions independently (pixel bboxes)
                              |
            +-----------------+-----------------+
            v                 v                 v
       TYPED (>=0.65)   MIXED (0.45-0.65)  HANDWRITTEN (<=0.45)
            |                 |                 |
            v                 v                 v
        Surya OCR        Ensemble           TrOCR
                        Surya+TrOCR        + padding
            |                 |                 |
            +--------+--------+-----------------+
                     |
                     v  (per-line override)
         Lines in Florence-2 detected regions
           -> forced ensemble regardless of page type
         Signature region  -> 0.90 TrOCR gate
         Handwriting region -> normal gate
         Both overlap       -> signature gate wins
                     |
                     v
              LAYOUT ANALYSIS
       Table Transformer + Florence-2 OD
                     |
                     v
             POST-PROCESSING (17 stages)
                     |
                     v
               JSON OUTPUT
```

### Post-Processing Pipeline

The post-processing pipeline in `postprocess_output()` applies a sanitize pre-step followed by 17 sequential stages:

| # | Stage | Description |
|---|-------|-------------|
| 0 | Sanitize elements | Validate and coerce element fields (type, content, bbox) at the pipeline boundary |
| 1 | Filter empty regions | Remove tables/layout regions with <30% text overlap |
| 2 | Normalize underscores | Standardize form fields (`Name:____` -> `Name: ___`) |
| 3 | Validate table structure | Score tables 0-100; remove if score < 50 |
| 4 | Deduplicate layout regions | Remove duplicate regions by bbox |
| 5 | Score hallucinations | 7-signal scoring; remove >= 0.50, flag > 0.30 |
| 6 | Filter rotated margin text | Remove vertical Bates numbers at page edges |
| 7 | Clean text content | Whitespace normalization, Unicode fixes, HTML/math markup stripping |
| 8 | Classification override | If >=50% of text elements sourced from TrOCR, override page type to handwritten |
| 9 | Strip TrOCR trailing periods | Remove spurious periods from TrOCR output (typed/mixed docs only, abbreviation-safe) |
| 10 | Repair decimal-dash confusion | Page-level context detection; fix `.88` misread as `-88` |
| 11 | Repair parentheses | Fix Surya's `BRANDS)` -> `BRAND(S)` pattern |
| 12 | Non-word OCR correction | Generalizable spell checker + OCR character confusion matrix |
| 13 | Replace signature text | Pattern-matched signature labels -> `(signature)` |
| 14 | Filter signature garbage | Remove single-word text overlapping >50% with signature bboxes |
| 15 | Remove duplicate words | Fix TrOCR beam search artifacts (`straight straight` -> `straight`) |
| 16 | Extract table cells | Assign OCR text to Table Transformer's row/column grid |
| 17 | Normalize phone numbers | Format phone numbers to `(xxx) xxx-xxxx`; detect fax vs phone |

### Enhanced Document Classification

The classifier uses 8 weighted signals to determine document type:

| Signal | Max Weight | Logic |
|--------|-----------|-------|
| Stroke Variance | +0.20 | Low variance = typed |
| Line Regularity | +0.12 | High regularity = typed |
| Angle Variance | -0.15 to +0.10 | Low = typed, High = handwritten penalty |
| Edge Density | +0.08 | Moderate density = typed |
| Form Structure | +0.30 | Forms/tables = typed (largest weight) |
| Character Uniformity | +0.12 | Uniform heights = typed |
| Signature Isolation | +0.10 | Signatures on forms = typed |
| Fax Header Detection | +0.20 | Fax/letterhead text = typed |

**Special adjustments:**
- **Ruled paper handwriting:** line_score >= 0.95, form_score < 0.25, moderate angle variance -> -0.12 (bias toward handwritten)
- **High angle variance penalty:** angle_score > 1200, low form_score -> -0.08 to -0.15
- **Sparse form safeguard:** Low edge density with form structure forces typed_score to 0.65 minimum

### Hallucination Detection

Text elements are scored using 7 weighted signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Confidence | 15% | Low OCR confidence increases score |
| Text Length | 10% | Very short text (1-2 chars) is suspicious |
| Character Patterns | 25% | Repeating chars, isolated digits, punctuation-only, unusual chars |
| Bbox Size | 15% | Tiny bboxes or abnormal aspect ratios |
| Valid Text Check | 15% | Not matching word/date/email/URL patterns |
| Repetition | 10% | Consecutive or all-same repeated words |
| Margin Position | 10% | Text at extreme left/right page edges |

**Decision logic:** Score >= 0.50 -> **Remove**, Score > 0.30 -> **Flag**, Score <= 0.30 -> **Keep**

### Table Validation

Detected tables are validated for true tabular structure:

| Signal | Max Points | Description |
|--------|------------|-------------|
| Columns | 40 | 3+ columns = 40 pts, 2 columns = 25 pts |
| Rows | 25 | 3+ rows = 25 pts, 2 rows = 15 pts |
| Grid Coverage | 20 | How well text fills the column x row grid |
| Text Density | 10 | Ratio of text area to table area (3-80%) |
| Confidence | 5 | Model confidence >= 95% |

Tables with structure_score < 50 are removed.

### Phone Number Normalization

| Input Format | Normalized Output |
|--------------|-------------------|
| `614-466-5087` | `(614) 466-5087` |
| `212/545-3297` | `(212) 545-3297` |
| `Fax:614-466-5087` | `(614) 466-5087` |
| `206 623 0594` | `(206) 623-0594` |

Adds `normalized_phone`, `phone_type` (`"fax"` or `"phone"`), and validation status. ZIP codes and dates are excluded.

### OCR Correction System

A single generalizable correction mechanism handles OCR errors without corpus-specific dictionaries:

| Component | Scope | Logic |
|-----------|-------|-------|
| `pyspellchecker` | Detect non-words | Words not in English dictionary (min 5 chars, skips acronyms/digits) |
| OCR confusion matrix | Filter candidates | Only accept corrections differing by visually confusable characters (e.g., `c`↔`e`, `l`↔`1`, `n`↔`h`) |
| Frequency ranking | Disambiguation | When multiple OCR-plausible candidates exist, pick the most common English word |

This approach is fully generalizable — it works on any English document without manual dictionary updates.

## Setup

### Prerequisites
*   **OS:** Windows / Linux (primarily tested on Google Colab with GPU runtime)
*   **Hardware:** NVIDIA GPU (8GB+ VRAM recommended)
*   **Python:** 3.10+

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/SaKinLord/Document-Processing-Pipeline.git
    cd Document-Processing-Pipeline
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Ensure PyTorch is installed with CUDA support. `transformers` is pinned to <5.0.0 for Florence-2 compatibility.

## Usage

### 1. Run the Pipeline
Place document images (PNG, JPG, PDF) in the `input/` directory:

```bash
python main.py --input_dir input --output_dir output
```

### 2. Visualize Results
Generate annotated images with color-coded bounding boxes:

```bash
python visualize_output.py --input_dir input --json_dir output --output_dir output/visualized
```

**Color coding:**
| Color | Element |
|-------|---------|
| Blue | Text (Surya) |
| Orange | Text (TrOCR) |
| Dark Blue | Table |
| Purple | Signature |
| Teal | Logo |
| Green | Figure/Image |
| Red | Hallucination (score >= 0.40) |

**Flags:**
- `--show-layout` — Show layout region boxes (hidden by default)
- `--no-content` — Hide text content labels
- `--no-confidence` — Hide confidence scores

### 3. Run Accuracy Tests
```bash
python tests/test_accuracy.py
python tests/test_accuracy.py --save-baseline   # save current metrics
python tests/test_accuracy.py --compare          # compare against baseline
```

### 4. Output Format
For each input file, a JSON is generated:

```json
{
  "filename": "doc.png",
  "pages": [
    {
      "page_number": 1,
      "document_type": "typed",
      "classification_confidence": 0.78,
      "language": "en",
      "elements": [
        {
          "type": "text",
          "content": "FAX NO. (614) 466-5087",
          "bbox": [100, 200, 500, 250],
          "confidence": 0.98,
          "source_model": "surya",
          "row_id": 5,
          "normalized_phone": "(614) 466-5087",
          "phone_type": "fax"
        },
        {
          "type": "text",
          "content": "Leonard H Jones",
          "bbox": [300, 800, 600, 850],
          "confidence": 0.85,
          "source_model": "trocr",
          "in_signature_region": true
        },
        {
          "type": "table",
          "bbox": [50, 300, 700, 600],
          "confidence": 0.97,
          "structure_score": 85.0,
          "structure_signals": ["columns:5", "rows:12", "grid_coverage:53%"],
          "num_rows": 12,
          "num_columns": 5,
          "cells": [
            {"row": 0, "col": 0, "bbox": [50, 300, 180, 330], "content": "Date", "is_header": true},
            {"row": 0, "col": 1, "bbox": [180, 300, 310, 330], "content": "Amount", "is_header": true},
            {"row": 1, "col": 0, "bbox": [50, 330, 180, 360], "content": "01/15/1999", "is_header": false}
          ]
        },
        {
          "type": "signature",
          "bbox": [300, 750, 600, 850],
          "description": "Detected signature"
        }
      ]
    }
  ]
}
```

**Element types:** `text`, `table`, `signature`, `logo`, `seal`, `figure`, `image`, `human face`

**Text element fields:** `content`, `bbox`, `confidence` (Surya's), `source_model` (`surya`/`trocr`), `row_id`, optional `in_signature_region`, `hallucination_score`, `hallucination_signals`, `normalized_phone`, `phone_type`, `date_validation`

**Table element fields:** `bbox`, `confidence`, `structure_score`, `structure_signals`, optional `num_rows`, `num_columns`, `cells` (array of `{row, col, bbox, content, is_header}`).

## Project Structure

```
Document-Processing-Pipeline/
├── main.py                          # Entry point
├── visualize_output.py              # Visualization with color-coded annotations
├── pyproject.toml                   # Project metadata, ruff linting, pytest config
├── requirements.txt
├── src/
│   ├── config.py                    # Centralized pipeline thresholds (routing, confidence, hallucination, table, signature)
│   ├── processing_pipeline.py       # DocumentProcessor: model loading, per-page OCR routing
│   ├── postprocessing/              # Post-processing pipeline (sanitize + 17 stages)
│   │   ├── pipeline.py              # Orchestrator: postprocess_output() with per-step fault isolation
│   │   ├── hallucination.py         # 7-signal hallucination scoring and removal
│   │   ├── ocr_corrections.py       # Generalizable spell checker + OCR confusion matrix
│   │   ├── normalization.py         # Underscore normalization, text cleaning, parenthesis repair
│   │   ├── signatures.py            # Signature text replacement and garbage filtering
│   │   ├── table_validation.py      # Table structure scoring and cell extraction
│   │   └── phone_date.py            # Phone number normalization and date validation
│   ├── utils/                       # Shared utilities (modular package)
│   │   ├── classification.py        # 8-feature document type classifier
│   │   ├── image.py                 # Pre-processing: denoising, deskewing, resizing
│   │   ├── bbox.py                  # Bounding box overlap and intersection utilities
│   │   └── text.py                  # Text row clustering for reading order
│   └── models/
│       ├── handwriting.py           # TrOCR wrapper (beam search, pooler disabled)
│       └── table.py                 # Table Transformer (detection + structure)
├── input/                           # Place documents here
├── output/                          # JSON results + visualized/ subdirectory
└── tests/
    ├── test_accuracy.py             # WER/CER regression tests (32-doc validation set)
    ├── test_postprocessing.py       # Unit tests for postprocessing functions
    └── flexible_evaluation.py       # Punctuation-insensitive evaluator
```

## Models

| Model | Source | Purpose |
|-------|--------|---------|
| Florence-2 | microsoft/Florence-2-large-ft | Phrase grounding (signatures, handwriting), object detection, captioning |
| Surya | surya-ocr (>=0.4.0) | Primary OCR for typed text |
| TrOCR | microsoft/trocr-large-handwritten | Handwriting recognition (beam search, 4 beams) |
| Table Transformer | microsoft/table-transformer-detection + structure-recognition | Table detection (0.7 threshold) and structure extraction (0.5 threshold) |

All models load onto GPU if available, falling back to CPU.

## Dependencies

*   `torch` (with CUDA support)
*   `torchvision`
*   `transformers` (>=4.38.0, <5.0.0)
*   `surya-ocr`
*   `Pillow`
*   `numpy`
*   `opencv-python-headless`
*   `scikit-learn`
*   `langdetect`
*   `pyspellchecker`
*   `pypdfium2`
*   `accelerate`
*   `einops`, `timm`, `scipy`

## Known Limitations

*   **English/US-only:** Spell correction, phone normalization, and date parsing assume English language and US formats. International documents would require additional locale support.
*   **Florence-2 bbox precision:** Florence-2 phrase grounding returns coarse bounding boxes. In mixed documents, a handwriting region may extend over adjacent typed text, causing those lines to be routed through TrOCR unnecessarily. The impact is minor (TrOCR may lowercase typed text), and the net effect is still positive for handwriting recall.
*   **Dense form layouts:** Documents with complex grid structures (many small fields, checkboxes, and lines) produce fragmented OCR elements. The two currently failing test documents (33% and 30% flex WER) are both dense Hazleton Laboratories project sheet forms.
*   **Classification edge cases:** Some fully handwritten documents on clean white paper with uniform line spacing are misclassified as "typed" by the image-based 8-feature classifier. A post-OCR classification override corrects this: if >=50% of text elements were sourced from TrOCR, the page is reclassified as "handwritten," preserving legitimate sentence-ending periods that would otherwise be stripped.

## License

MIT License
