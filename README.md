# Document Processing Pipeline (Hybrid Intelligent System)

A local pipeline for extracting structured data from complex, multi-format documents. It uses a **Hybrid Architecture** that dynamically switches between specialized models based on document content (Typed, Handwritten, or Mixed), with **per-line handwriting routing** that detects and re-routes individual handwritten lines on otherwise typed pages.

## Key Features

*   **Intelligent 8-Feature Classification:** Automatically detects document type using stroke width variance, line regularity, contour angle variance, edge density, form structure detection, character uniformity, signature region isolation, and fax/letterhead header detection. Includes ruled paper handwriting detection and sparse form safeguards.
*   **Per-Line Handwriting Routing:** Florence-2 phrase grounding detects signature and handwritten text regions before OCR. Lines overlapping these regions are routed to TrOCR ensemble regardless of page-level classification, catching handwritten fill-ins, names, and annotations on typed forms.
*   **Multi-Model Intelligence:**
    *   **Surya OCR:** High-precision text detection and recognition for typed documents
    *   **TrOCR:** Specialized handwriting recognition with beam search (4 beams), 12px bbox padding for descenders, and punctuation spacing normalization
    *   **Table Transformer (DETR):** Table detection and structural extraction with validation scoring
    *   **LayoutLMv3:** Document layout semantic segmentation (headers, titles, figures)
    *   **Florence-2 (VLM):** Vision-language model for object detection, captioning, and phrase grounding (signatures, handwriting, logos, seals)
*   **Multi-Stage Post-Processing:**
    *   7-signal hallucination detection and removal
    *   Offensive OCR misread filter with audit trail
    *   Context-aware OCR corrections (single-word and multi-word)
    *   Parenthesis repair for Surya artifacts
    *   Slash/hyphen compound word splitting before corrections
    *   Signature text replacement and garbage filtering
    *   Structural table validation and heuristic borderless table promotion
    *   Phone number normalization and date validation
    *   Domain-specific spell correction
    *   Rotated margin text filtering (vertical Bates numbers)
*   **Robust Pre-processing:** Adaptive denoising and deskewing via OpenCV with resolution-adaptive parameters

## Accuracy & Performance

Based on regression testing with 15 ground truth documents (5 handwritten, 10 typed):

| Metric | Value |
|--------|-------|
| Regression Tests | **15/15 passing** |
| Average Flex WER | **~8.5%** |
| Average Flex CER | **~5.7%** |

Flexible evaluation ignores punctuation and formatting differences. Pass/fail is determined by flex CER.

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
    Signature Isolation | Fax Header | Ruled Paper Detection
                              |
                              v
                  FLORENCE-2 PHRASE GROUNDING
              Detect "signature" + "handwritten text"
               regions BEFORE OCR (pixel bboxes)
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
         Lines in Florence-2 handwritten regions
           -> forced ensemble regardless of page type
         Signature overlap -> 0.90 TrOCR gate
         Handwriting overlap -> normal gate
                     |
                     v
              LAYOUT ANALYSIS
       LayoutLMv3 + Table Transformer + Florence-2 OD
                     |
                     v
             POST-PROCESSING (17 stages)
                     |
                     v
               JSON OUTPUT
```

### Post-Processing Pipeline

The post-processing pipeline in `postprocess_output()` applies 17 sequential stages:

| # | Stage | Description |
|---|-------|-------------|
| 1 | Filter empty regions | Remove tables/layout regions with <30% text overlap |
| 2 | Normalize underscores | Standardize form fields (`Name:____` -> `Name: ___`) |
| 3 | Validate table structure | Score tables 0-100; remove if score < 50 |
| 4 | Promote borderless tables | Detect missed tables via layout region clustering |
| 5 | Deduplicate layout regions | Remove duplicate regions by bbox |
| 6 | Score hallucinations | 7-signal scoring; remove >= 0.50, flag > 0.30 |
| 7 | Filter rotated margin text | Remove vertical Bates numbers at page edges |
| 8 | Filter offensive misreads | Regex-based reputational-risk corrections with audit trail |
| 9 | Clean text content | Whitespace normalization, Unicode fixes |
| 10 | Repair parentheses | Fix Surya's `BRANDS)` -> `BRAND(S)` pattern |
| 11 | OCR corrections | Context-aware single-word fixes with slash/hyphen splitting |
| 12 | Multi-word corrections | Proper noun phrase replacements (e.g., `HAVENS GERMAN` -> `HAGENS BERMAN`) |
| 13 | Replace signature text | Pattern-matched signature labels -> `(signature)` |
| 14 | Filter signature garbage | Remove single-word text overlapping >50% with signature bboxes |
| 15 | Remove duplicate words | Fix TrOCR beam search artifacts (`straight straight` -> `straight`) |
| 16 | Normalize phone numbers | Extract and format to `(XXX) XXX-XXXX` with type detection |
| 17 | Classification refinement | Detect fax/form indicators in text to refine document type |

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

### Heuristic Table Promotion

Detects borderless tables missed by the Table Transformer by analyzing clusters of layout regions:

- Clusters vertically adjacent layout regions (max 20px gap)
- Requires: 15+ regions, 4+ columns, 5+ rows, 55% grid coverage, score >= 90
- Refines top boundary via vertical continuity rule to trim non-tabular headers
- Promoted tables tagged with `source: "heuristic_promotion"` and `confidence: 0.8`

### Phone Number Normalization

| Input Format | Normalized Output |
|--------------|-------------------|
| `614-466-5087` | `(614) 466-5087` |
| `212/545-3297` | `(212) 545-3297` |
| `Fax:614-466-5087` | `(614) 466-5087` |
| `206 623 0594` | `(206) 623-0594` |

Adds `normalized_phone`, `phone_type` (`"fax"` or `"phone"`), and validation status. ZIP codes and dates are excluded.

### OCR Correction System

Four correction dictionaries handle different error types:

| Dictionary | Scope | Context Rules |
|------------|-------|---------------|
| `OFFENSIVE_OCR_CORRECTIONS` | Regex-based reputational-risk misreads | Unconditional with audit trail |
| `OCR_CONFUSION_CORRECTIONS` | Single-word substitutions (~25 entries) | Empty context `[]` = unconditional (non-real words only); non-empty = requires context word in 5-word window |
| `MULTI_WORD_OCR_CORRECTIONS` | Proper noun phrases (~15 entries) | Same context rules; real English phrases require context |
| `PREFIX_CORRECTIONS` | TrOCR dropped prefixes (un-/dis-) | Handwriting mode only; checks negation context window |

Compound words split on `/` or `-` before matching (e.g., `DIRECTOR/DEPARIMENT` -> `DIRECTOR/DEPARTMENT`).

## Setup

### Prerequisites
*   **OS:** Windows / Linux
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
          "structure_signals": ["columns:5", "rows:12", "grid_coverage:53%"]
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

**Element types:** `text`, `table`, `layout_region`, `signature`, `logo`, `seal`, `figure`, `image`, `human face`

**Text element fields:** `content`, `bbox`, `confidence` (Surya's), `source_model` (`surya`/`trocr`), `row_id`, optional `in_signature_region`, `hallucination_score`, `hallucination_signals`, `normalized_phone`, `phone_type`, `date_validation`, `offensive_ocr_corrected`

## Project Structure

```
Document-Processing-Pipeline/
├── main.py                          # Entry point
├── visualize_output.py              # Visualization with color-coded annotations
├── requirements.txt
├── src/
│   ├── processing_pipeline.py       # DocumentProcessor: model loading, per-page OCR routing
│   ├── postprocessing.py            # 17-stage post-processing pipeline (~2400 lines)
│   ├── utils.py                     # Classification, preprocessing, spell correction, bbox utils
│   ├── resources/
│   │   └── domain_dictionary.txt    # Domain vocabulary (legal, pharma terms)
│   └── models/
│       ├── handwriting.py           # TrOCR wrapper (beam search, pooler disabled)
│       ├── table.py                 # Table Transformer (detection + structure)
│       └── layout.py               # LayoutLMv3 (token classification, bbox normalization)
├── input/                           # Place documents here
├── output/                          # JSON results + visualized/ subdirectory
└── tests/
    ├── test_accuracy.py             # WER/CER regression tests
    └── flexible_evaluation.py       # Punctuation-insensitive evaluator
```

## Models

| Model | Source | Purpose |
|-------|--------|---------|
| Florence-2 | microsoft/Florence-2-large-ft | Phrase grounding (signatures, handwriting), object detection, captioning |
| Surya | surya v0.17 API | Primary OCR for typed text |
| TrOCR | microsoft/trocr-large-handwritten | Handwriting recognition (beam search, 4 beams) |
| Table Transformer | microsoft/table-transformer-detection + structure-recognition | Table detection (0.7 threshold) and structure extraction (0.5 threshold) |
| LayoutLMv3 | microsoft/layoutlmv3-large | Document layout classification (bbox normalized to 0-1000 scale) |

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

## License

MIT License
