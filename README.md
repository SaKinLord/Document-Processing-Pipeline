# Document Processing Pipeline (Hybrid Intelligent System)

A state-of-the-art local pipeline for extracting structured data from complex, multi-format documents. It uses a **Hybrid Architecture** that dynamically switches between specialized models based on document content (Typed, Handwritten, or Mixed), ensuring high fidelity for technical forms while maintaining readability for cursive notes.

## 🚀 Key Features

*   **Intelligent 8-Feature Classification:** Automatically detects document type using:
    *   Stroke width variance
    *   Line regularity (horizontal alignment)
    *   Contour angle variance (with handwriting penalty for high variance)
    *   Edge density
    *   **Form structure detection** (horizontal/vertical lines)
    *   **Character uniformity**
    *   **Signature region isolation** (bottom 20% of page)
    *   **Fax/letterhead header detection** (top 15%)
    *   **Ruled paper handwriting detection** (biases lined paper toward handwritten)
*   **Three-Way Routing:** Documents classified as `typed`, `handwritten`, or `mixed` (ensemble OCR)
    *   Thresholds: typed ≥ 0.65, handwritten ≤ 0.45, else mixed
*   **Multi-Model Intelligence:**
    *   **Surya OCR (SegFormer):** High-precision text detection and recognition for typed documents.
    *   **TrOCR (Transformer OCR):** Specialized attention-based recognition for handwritten lines with:
        *   Beam search decoding (num_beams=4) for improved accuracy
        *   12px bounding box padding to preserve descenders (g, y, p, q)
        *   Punctuation spacing normalization
    *   **Table Transformer (DETR):** Accurately detects tables and extracts structural bounding boxes.
    *   **LayoutLMv3:** Understands document layout (Headers, Titles, Figures) for semantic segmentation.
    *   **Florence-2 (VLM):** Vision-Language Model for captioning figures and detecting logos/signatures.
*   **Multi-Signal Post-Processing:**
    *   Hallucination detection using 6 weighted signals
    *   Layout region deduplication
    *   Text content cleaning
    *   **Empty region filtering** (removes hallucinated tables/regions)
    *   **Structural validation filter** (removes false positive tables)
    *   **Heuristic table promotion** (detects borderless tables missed by the model)
    *   **Phone number normalization** (extracts & standardizes phone numbers)
    *   **Punctuation spacing normalization** (fixes TrOCR's extra spaces around punctuation)
*   **Robust Pre-processing:** Adaptive denoising using OpenCV to clean scanned artifacts.

## 📊 Accuracy & Performance

Based on regression testing with 12 ground truth documents (Feb 2026):

| Document Type | Count | WER | CER | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Typed / Structured** | 7 | **0.0%** | **0.0%** | Forms, Invoices, Technical Specs, Faxes |
| **Handwritten** | 5 | **1.6-16.4%** | **1.7-5.3%** | Letters, Notes, Cursive Annotations |
| **Mixed** | 1 | **6.4%** | **2.5%** | Forms with handwritten fill-ins |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Regression Tests | **12/12 passing** |
| Average WER | **3.4%** |
| Average CER | **1.8%** |
| Processing Time (15 docs) | ~450 seconds |
| Classification Accuracy | **100%** (0 misclassifications) |

## 🛠️ Architecture

The pipeline follows a sequential flow with intelligent branching:

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION                                │
│                    Load Image/PDF                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PRE-PROCESSING                              │
│              Denoise + Resize + Grayscale                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 8-FEATURE CLASSIFIER                             │
│  Stroke Variance │ Line Regularity │ Angle Variance (weighted)   │
│  Edge Density    │ Form Structure  │ Character Uniformity        │
│  Signature Isolation (bottom 20%) │ Fax Header Detection (top 15%)│
│            Ruled Paper Handwriting Detection                     │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      ┌──────────┐      ┌──────────┐      ┌──────────┐
      │  TYPED   │      │  MIXED   │      │HANDWRITTEN│
      │ (≥0.65)  │      │(0.45-0.65)│     │ (≤0.45)  │
      └────┬─────┘      └────┬─────┘      └────┬─────┘
           │                 │                  │
           ▼                 ▼                  ▼
      ┌─────────┐      ┌───────────┐      ┌─────────┐
      │  Surya  │      │ Ensemble  │      │  TrOCR  │
      │   OCR   │      │Surya+TrOCR│      │ +padding│
      └────┬────┘      └─────┬─────┘      └────┬────┘
           │                 │                  │
           └─────────────────┼──────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LAYOUT ANALYSIS                              │
│           LayoutLMv3 + Table Transformer + Florence-2            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                               │
│  Empty Filter │ Table Validation │ Hallucination │ Punctuation   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       JSON OUTPUT                                │
└─────────────────────────────────────────────────────────────────┘
```

### Enhanced Document Classification

The classifier uses 8 weighted signals to determine document type:

| Signal | Contribution | Logic |
|--------|-------------|-------|
| Stroke Variance | +0.10 to +0.20 | Low variance = typed |
| Line Regularity | +0.06 to +0.12 | High regularity = typed |
| Angle Variance | -0.15 to +0.10 | Low = typed, High = handwritten penalty |
| Edge Density | +0.04 to +0.08 | Moderate density = typed |
| Form Structure | +0.15 to +0.35 | Forms/tables = typed document |
| Character Uniformity | +0.06 to +0.12 | Uniform heights = typed |
| Signature Isolation | +0.10 | Signatures on forms = typed |
| Fax Header Detection | +0.15 | Fax/letterhead text = typed |

**Special Adjustments:**

| Adjustment | Conditions | Effect |
|------------|------------|--------|
| **Ruled Paper Handwriting** | line_score ≥ 0.95(lined paper), form_score < 0.25, angle_score 300-700 | -0.12 (bias toward handwritten) |
| **High Angle Variance Penalty** | angle_score > 1200, form_score < 0.5 | -0.08 to -0.15 (bias toward handwritten) |

**Thresholds:**
- typed_score ≥ 0.65 → **Typed** (use Surya OCR)
- typed_score ≤ 0.45 → **Handwritten** (use TrOCR)
- Otherwise → **Mixed** (use ensemble OCR)

### Post-Processing: Multi-Signal Hallucination Detection

Text elements are scored for hallucination likelihood using 6 signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Confidence | 20% | Low OCR confidence increases score |
| Text Length | 15% | Very short text (1-3 chars) is suspicious |
| Character Patterns | 25% | Repeating chars, only digits, gibberish |
| Bbox Size | 15% | Abnormally small/large bounding boxes |
| Dictionary Check | 15% | Non-word that's not a number/date |
| Repetition | 10% | Repeated words ("the the the") |

**Decision Logic:**
- Score > 0.70 → **Remove** (likely hallucination)
- Score 0.40-0.70 → **Flag** (keep but mark uncertain)
- Score ≤ 0.40 → **Keep** (likely valid)

### Empty Region Filtering

Removes hallucinated table and layout regions that don't overlap with actual text content. This prevents false table detections in empty areas of documents.

### Structural Validation Filter

Validates that detected tables have true tabular structure (not just forms or aligned lists). Uses multiple signals:

| Signal | Max Points | Description |
|--------|------------|-------------|
| Columns | 40 | 3+ columns = 40 pts, 2 columns = 25 pts |
| Rows | 25 | 3+ rows = 25 pts, 2 rows = 15 pts |
| Grid Coverage | 20 | How well text fills the column×row grid |
| Text Density | 10 | Ratio of text area to table area (3-80%) |
| Confidence | 5 | Model confidence ≥ 95% |

**Decision Logic:**
- Score ≥ 50 → **Keep** (valid table structure)
- Score < 50 → **Remove** (likely form or false positive)

**Output Fields Added:**
- `structure_score`: 0-100 validation score
- `structure_signals`: Array of detected structural features

### Phone Number Normalization

Extracts and normalizes phone numbers from text elements:

| Input Format | Normalized Output |
|--------------|-------------------|
| `614-466-5087` | `(614) 466-5087` |
| `212/545-3297` | `(212) 545-3297` |
| `Fax:614-466-5087` | `(614) 466-5087` |
| `206 623 0594` | `(206) 623-0594` |

**Adds to JSON:**
- `normalized_phone` or `normalized_phones` (for multiple)
- `phone_type`: `"fax"`, `"phone"`, or empty

**False Positive Protection:** ZIP codes (e.g., `64105-2118`) and dates are NOT normalized.

### Heuristic Table Promotion

Detects borderless tables that the Table Transformer model misses by analyzing clusters of `layout_region` elements for tabular structure.

**How it works:**
1. Clusters vertically adjacent layout regions (max gap: 20px)
2. Validates cluster structure using column/row alignment
3. Promotes valid clusters to synthetic `table` elements

**Validation Criteria (must meet ALL):**

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| Regions | ≥15 | Minimum layout regions in cluster |
| Columns | ≥4 | Detected column alignment |
| Rows | ≥5 | Detected row alignment |
| Structure Score | ≥90 | Combined structural validation |
| Grid Coverage | ≥55% | Text elements filling the grid |

**Boundary Refinement:**

Trimms non-tabular header rows from the top using the "Vertical Continuity Rule":
- A row qualifies as anchor if it has **≥6 items** (high density)
- OR if it has **≥4 columns spanning ≥60% width** AND the next row also qualifies

**Output Fields Added:**
- `source: "heuristic_promotion"` - Indicates table was promoted (not model-detected)
- `confidence: 0.8` - Lower confidence than model-detected tables

## ⚙️ Setup

### Prerequisites
*   **OS:** Windows / Linux
*   **Hardware:** NVIDIA GPU (8GB+ VRAM recommended for full pipeline).
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
    *(Note: Ensure you have PyTorch installed with CUDA support)*

## 🖥️ Usage

### 1. Run the Pipeline
Place your document images (PNG, JPG, PDF) in the `input/` directory and run:

```bash
python main.py --input_dir input --output_dir output
```

**Arguments:**
*   `--input_dir`: Source directory (Default: `input`)
*   `--output_dir`: Destination for JSON files (Default: `output`)

### 2. Output Format
For each file `input/doc.png`, an `output/doc.json` is generated:

```json
{
  "filename": "doc.png",
  "pages": [
    {
      "page_number": 1,
      "document_type": "mixed",
      "classification_confidence": 0.5,
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
          "type": "layout_region",
          "region_type": "LABEL_0",
          "bbox": [100, 200, 500, 250]
        },
        {
          "type": "table",
          "bbox": [...],
          "confidence": 0.97,
          "structure_score": 100.0,
          "structure_signals": ["columns:5", "rows:12", "grid_coverage:53%"]
        }
      ]
    }
  ]
}
```

### 3. Visualization
To audit the results visually (draws bounding boxes on images):

```bash
python visualize_output.py --input_dir input --json_dir output --output_dir output/visualized
```

## 📁 Project Structure

```
Document-Processing-Pipeline/
├── main.py                    # Entry point
├── src/
│   ├── processing_pipeline.py # Main DocumentProcessor class
│   ├── postprocessing.py      # Hallucination detection & cleaning
│   ├── utils.py               # Classifier & utilities
│   └── models/
│       ├── handwriting.py     # TrOCR wrapper
│       ├── table.py           # Table Transformer wrapper
│       └── layout.py          # LayoutLMv3 wrapper
├── input/                     # Place documents here
├── output/                    # JSON results saved here
└── requirements.txt
```

## 🧩 Dependencies
*   `surya-ocr`
*   `transformers` (Hugging Face)
*   `torch`
*   `opencv-python`
*   `pillow`
*   `scikit-learn`
*   `numpy`

## 📜 License

MIT License
