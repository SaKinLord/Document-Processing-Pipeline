# Document Processing Pipeline (Hybrid Intelligent System)

A state-of-the-art local pipeline for extracting structured data from complex, multi-format documents. It uses a **Hybrid Architecture** that dynamically switches between specialized models based on document content (Typed, Handwritten, or Mixed), ensuring high fidelity for technical forms while maintaining readability for cursive notes.

## 🚀 Key Features

*   **Intelligent 6-Feature Classification:** Automatically detects document type using:
    *   Stroke width variance
    *   Line regularity (horizontal alignment)
    *   Contour angle variance
    *   Edge density
    *   **Form structure detection** (horizontal/vertical lines)
    *   **Character uniformity**
*   **Three-Way Routing:** Documents classified as `typed`, `handwritten`, or `mixed` (ensemble OCR)
*   **Multi-Model Intelligence:**
    *   **Surya OCR (SegFormer):** High-precision text detection and recognition for typed documents.
    *   **TrOCR (Transformer OCR):** Specialized attention-based recognition for handwritten lines.
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
*   **Robust Pre-processing:** Adaptive denoising using OpenCV to clean scanned artifacts.

## 📊 Accuracy & Performance

Based on testing with 15 diverse documents (Jan 2026):

| Document Type | Count | Accuracy | Best For |
| :--- | :--- | :--- | :--- |
| **Typed / Structured** | 8 | **95-99%** | Forms, Invoices, Technical Specs, Faxes |
| **Handwritten** | 2 | **80-90%** | Letters, Notes, Cursive Annotations |
| **Mixed** | 5 | **90-95%** | Forms with handwritten fill-ins |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Processing Time (15 docs) | ~230 seconds |
| Avg JSON file size | ~20KB |
| Classification Accuracy | 100% (0 misclassifications) |

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
│                 6-FEATURE CLASSIFIER                             │
│  Stroke Variance │ Line Regularity │ Angle Variance              │
│  Edge Density    │ Form Structure  │ Character Uniformity        │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      ┌──────────┐      ┌──────────┐      ┌──────────┐
      │  TYPED   │      │  MIXED   │      │HANDWRITTEN│
      │ (≥0.70)  │      │(0.30-0.70)│     │ (≤0.30)  │
      └────┬─────┘      └────┬─────┘      └────┬─────┘
           │                 │                  │
           ▼                 ▼                  ▼
      ┌─────────┐      ┌───────────┐      ┌─────────┐
      │  Surya  │      │ Ensemble  │      │  TrOCR  │
      │   OCR   │      │Surya+TrOCR│      │   OCR   │
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
│  Empty Filter │ Table Validation │ Hallucination │ Phone Norm    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       JSON OUTPUT                                │
└─────────────────────────────────────────────────────────────────┘
```

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
