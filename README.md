# Document Processing Pipeline

A local, deep-learning-based pipeline for extracting structured data from multi-format documents (PDFs, images, handwriting, forms).

## Features
- **Multi-Format Support:** Handles scanned forms, fax covers, and handwritten notes.
- **Robust Pre-processing:** Automatic noise removal for clearer scans.
- **Table Structure Recovery:** Intelligent clustering of text into logical rows (`row_id`).
- **Layout Awareness:** Refined VLM prompts (`<OCR_WITH_REGION>`) for better structural understanding.
- **Handwriting Recognition:** Excellent HTR (Handwritten Text Recognition) capabilities.
- **Local Execution:** Runs entirely offline using GPUs.
- **Scalable Output:** Generates individual JSON files for easy integration.

## Setup

### Prerequisites
- **OS:** Windows / Linux
- **Hardware:** NVIDIA GPU (Recommended 8GB+ VRAM) with CUDA support.
- **Python:** 3.8+

### Installation
1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd doc_processing_pipeline
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Pipeline
Place your document images in the `input/` directory and run:

```bash
python main.py --input_dir input --output_dir output
```

**Arguments:**
- `--input_dir`: Directory containing source images (images/scans). Default: `input`
- `--output_dir`: Directory where JSON results will be saved. Default: `output`

### 2. Output Format
For each input file `input/document.png`, an output file `output/document.json` is created.

**JSON Structure:**
```json
{
  "filename": "document.png",
  "pages": [
    {
      "page_number": 1,
      "elements": [
        {
          "type": "text",
          "content": "Line Item 1",
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.98,
          "row_id": 5
        },
        {
          "type": "table",
          "bbox": [...]
        }
      ]
    }
  ]
}
```

### 3. Visualization Tool
To visually verify the results, use the helper script to draw bounding boxes on the original images:

```bash
python visualize_output.py --input_dir input --json_dir output --output_dir output/visualized
```

Check the `output/visualized/` folder for images with color-coded boxes:
- **Red:** Text
- **Blue:** Tables
- **Green:** Figures/Images

## Dependencies
- `surya-ocr` for layout analysis and text detection.
- `transformers` (Florence-2) for vision-language tasks.
- `torch` for model inference.
