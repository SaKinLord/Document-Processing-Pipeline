import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor, FoundationPredictor
from langdetect import detect, LangDetectException

from src.utils import (
    load_image, convert_pdf_to_images, crop_image, pad_bbox,
    denoise_image, deskew_image, cluster_text_rows, is_bbox_too_large,
    classify_document_type, detect_signature_region,
)
from src.utils.bbox import bbox_overlap_ratio_of, bboxes_intersect
from src.models.handwriting import HandwritingRecognizer
from src.models.table import TableRecognizer
from src.config import CONFIG
from src.postprocessing import (
    normalize_punctuation_spacing,
    add_phone_validation_to_element,
    add_date_validation_to_element,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info("Loading models on %s...", self.device)

        # Routing thresholds (from centralized config)
        self.TEXT_CONFIDENCE_THRESHOLD = CONFIG.text_confidence_threshold
        self.HW_REGION_MIN_CONFIDENCE = CONFIG.hw_region_min_confidence
        self.DOC_HANDWRITTEN_GATE = CONFIG.doc_handwritten_gate
        self.TYPED_LOW_CONF_THRESHOLD = CONFIG.typed_low_conf_threshold
        self.SIGNATURE_TROCR_GATE = CONFIG.signature_trocr_gate
        self.TROCR_BBOX_PADDING = CONFIG.trocr_bbox_padding

        # Load Florence-2 for VLM tasks (Layout, Captioning)
        self.florence_model_id = "microsoft/Florence-2-large-ft"
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            self.florence_model_id,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device).eval()
        self.florence_processor = AutoProcessor.from_pretrained(
            self.florence_model_id,
            trust_remote_code=True
        )

        # Load Surya for High-Quality OCR (v0.17 API)
        logger.info("Loading Surya predictors...")
        self.det_predictor = DetectionPredictor()

        self.rec_foundation = FoundationPredictor()
        self.rec_predictor = RecognitionPredictor(self.rec_foundation)

        # Load New Specialized Models
        logger.info("Loading specialized models (TrOCR, TableTransformer)...")
        self.handwriting_recognizer = HandwritingRecognizer(device=self.device)
        self.table_recognizer = TableRecognizer(device=self.device)

    def run_florence(self, image: Image.Image, task_prompt: str,
                     text_input: Optional[str] = None) -> Dict[str, Any]:
        """Run a Florence-2 inference task on an image."""
        if image is None:
            logger.warning("run_florence received None image")
            return {}

        prompt = task_prompt if text_input is None else task_prompt + text_input

        imgs = [image] if isinstance(image, Image.Image) else image
        inputs = self.florence_processor(text=[prompt], images=imgs, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                use_cache=False,
            )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer

    def process_document(self, file_path: str) -> Tuple[Dict[str, Any], List[Image.Image]]:
        """Process a document file (image or PDF) and return structured OCR results."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            images = convert_pdf_to_images(file_path)
        else:
            img = load_image(file_path)
            images = [img] if img else []

        results = {
            "filename": os.path.basename(file_path),
            "pages": []
        }

        for i, image in enumerate(images):
            logger.info("Processing page %d...", i + 1)
            page_data = self.process_page(image, page_num=i + 1)
            results["pages"].append(page_data)

        return results, images

    def detect_language(self, text: str) -> str:
        """Detect language of text, defaulting to English."""
        try:
            if text and len(text.strip()) > 20:
                return detect(text)
        except LangDetectException:
            pass
        return "en"

    # ------------------------------------------------------------------
    # process_page — decomposed into focused helper methods
    # ------------------------------------------------------------------

    def process_page(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """Process a single page image through the full OCR pipeline."""
        # 0. Pre-processing
        image = self._preprocess_image(image)
        width, height = image.width, image.height

        page_content: Dict[str, Any] = {
            "page_number": page_num,
            "elements": [],
            "language": "en"
        }

        # Document classification
        doc_type, doc_confidence = self._classify_document(image, page_content)

        # Signature region detection
        signature_info = self._detect_signatures(image, page_content)

        # Florence-2 handwritten region detection
        handwritten_regions = self._detect_handwritten_regions(image)

        # Surya OCR + per-line routing
        text_elements, all_text_content = self._run_ocr_with_routing(
            image, width, height, doc_type, doc_confidence,
            handwritten_regions, signature_info
        )

        # Language detection
        full_text = " ".join(all_text_content)
        page_content["language"] = self.detect_language(full_text)

        # Cluster text into reading order
        clustered_text = cluster_text_rows(text_elements)
        page_content["elements"].extend(clustered_text)

        # Table detection (Table Transformer)
        torch.cuda.empty_cache()
        self._detect_tables(image, page_content)

        # Visual element detection (Florence-2 OD + phrase grounding)
        self._detect_visual_elements(image, width, height, page_content)

        return page_content

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply denoising and deskew to the input image."""
        image = denoise_image(image)
        image = deskew_image(image)
        return image

    def _classify_document(self, image: Image.Image,
                           page_content: Dict[str, Any]) -> Tuple[str, float]:
        """Classify document type and store results in page_content."""
        doc_type, doc_confidence = classify_document_type(image)
        page_content["document_type"] = doc_type
        page_content["classification_confidence"] = doc_confidence
        logger.info("  Document classification: %s (confidence: %.2f)", doc_type, doc_confidence)
        return doc_type, doc_confidence

    def _detect_signatures(self, image: Image.Image,
                           page_content: Dict[str, Any]) -> dict:
        """Detect signature region in the bottom of the page."""
        signature_info = detect_signature_region(image)
        if signature_info['has_signature']:
            page_content["signature_detected"] = True
            logger.info("  Signature region detected in bottom %d%% of page",
                        round((1 - CONFIG.signature_region_ratio) * 100))
        return signature_info

    def _detect_handwritten_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect signature and handwritten text regions using Florence-2.

        Uses a single phrase-grounding call with both phrases (period-separated)
        to save ~1 forward pass per page.
        """
        signature_bboxes = []
        handwriting_bboxes = []
        try:
            results = self.run_florence(
                image, "<CAPTION_TO_PHRASE_GROUNDING>",
                text_input="signature. handwritten text"
            )
            if '<CAPTION_TO_PHRASE_GROUNDING>' in results:
                for bbox, label in zip(results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'],
                                       results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']):
                    if 'signature' in label.lower():
                        signature_bboxes.append(bbox)
                    else:
                        handwriting_bboxes.append(bbox)
        except (RuntimeError, ValueError) as e:
            logger.warning("Florence-2 handwriting/signature detection failed: %s", e)
        finally:
            torch.cuda.empty_cache()

        # Build unified list: signatures first (higher routing priority), then
        # handwriting regions independently.  Previous logic deduped handwriting
        # bboxes *against* signature bboxes, but Florence-2's "handwritten text"
        # query returns bboxes that nearly always overlap its "signature" bboxes
        # (signatures are handwritten), so every handwriting region was eliminated
        # and per-line routing never activated.  Now we keep both types and only
        # self-dedup handwriting bboxes against each other to remove redundant
        # detections.  The routing in _route_line iterates signatures first, so
        # lines overlapping both still get the strict 0.90 TrOCR gate.
        handwritten_regions: List[Dict[str, Any]] = []
        for sb in signature_bboxes:
            handwritten_regions.append({'bbox': sb, 'type': 'signature'})

        # Self-dedup handwriting bboxes only (remove redundant detections)
        deduped_hw: List[List[float]] = []
        for hb in handwriting_bboxes:
            redundant = False
            for kept in deduped_hw:
                if bbox_overlap_ratio_of(hb, kept, reference_bbox=hb) > 0.50:
                    redundant = True
                    break
            if not redundant:
                deduped_hw.append(hb)

        for hb in deduped_hw:
            handwritten_regions.append({'bbox': hb, 'type': 'handwriting'})

        if handwriting_bboxes:
            logger.info("  Florence-2 handwriting regions: %d detected, %d after self-dedup",
                        len(handwriting_bboxes), len(deduped_hw))
        if signature_bboxes:
            logger.info("  Florence-2 signature regions: %d detected", len(signature_bboxes))

        return handwritten_regions

    def _run_ocr_with_routing(
        self, image: Image.Image, width: int, height: int,
        doc_type: str, doc_confidence: float,
        handwritten_regions: List[Dict[str, Any]],
        signature_info: dict
    ) -> Tuple[List[Dict], List[str]]:
        """Run Surya OCR with per-line routing to TrOCR for handwritten regions."""
        try:
            predictions = self.rec_predictor([image], ['ocr_with_boxes'], self.det_predictor)
            ocr_result = predictions[0]
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Surya OCR failed: %s", e)
            return [], []

        text_elements: List[Dict] = []
        all_text_content: List[str] = []

        for line in ocr_result.text_lines:
            # Lower confidence threshold for lines in handwritten regions
            in_hw_region = any(
                bboxes_intersect(line.bbox, r['bbox'])
                for r in handwritten_regions
            )
            min_confidence = self.HW_REGION_MIN_CONFIDENCE if in_hw_region else self.TEXT_CONFIDENCE_THRESHOLD
            if line.confidence < min_confidence:
                continue

            final_text, source_model = self._route_line(
                line, image, width, height, doc_type, doc_confidence, handwritten_regions
            )

            text_element = self._build_text_element(
                final_text, line, source_model, signature_info, height
            )

            text_elements.append(text_element)
            all_text_content.append(final_text)

        return text_elements, all_text_content

    def _route_line(self, line, image: Image.Image, width: int, height: int,
                    doc_type: str, doc_confidence: float,
                    handwritten_regions: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Decide which OCR model to use for a text line and return (text, source_model)."""
        final_text = line.text
        source_model = "surya"

        # Check if line overlaps a Florence-2 detected handwritten region
        line_hw_region_type = None
        for region in handwritten_regions:
            if bboxes_intersect(line.bbox, region['bbox']):
                line_hw_region_type = region['type']
                break

        run_handwriting_model = False
        use_ensemble = False

        if line_hw_region_type is not None:
            use_ensemble = True
        elif doc_type == "handwritten" and doc_confidence >= self.DOC_HANDWRITTEN_GATE:
            run_handwriting_model = True
        elif doc_type == "mixed" or doc_confidence < self.DOC_HANDWRITTEN_GATE:
            use_ensemble = True
        elif doc_type == "typed" and line.confidence < self.TYPED_LOW_CONF_THRESHOLD:
            if len(line.text.strip()) > 0:
                run_handwriting_model = True

        if use_ensemble:
            final_text, source_model = self._run_ensemble(
                line, image, width, height, line_hw_region_type
            )
        elif run_handwriting_model:
            padded_bbox = pad_bbox(line.bbox, self.TROCR_BBOX_PADDING, width, height)
            line_crop = crop_image(image, padded_bbox)
            hw_text, hw_conf = self.handwriting_recognizer.recognize(line_crop)
            if hw_text and len(hw_text.strip()) > 0 and hw_conf >= self.HW_REGION_MIN_CONFIDENCE:
                final_text = normalize_punctuation_spacing(hw_text)
                source_model = "trocr"

        return final_text, source_model

    def _run_ensemble(self, line, image: Image.Image, width: int, height: int,
                      line_hw_region_type: Optional[str]) -> Tuple[str, str]:
        """Run both Surya and TrOCR, pick the better result."""
        padded_bbox = pad_bbox(line.bbox, self.TROCR_BBOX_PADDING, width, height)
        line_crop = crop_image(image, padded_bbox)
        hw_text, hw_conf = self.handwriting_recognizer.recognize(line_crop)

        surya_conf = line.confidence
        trocr_valid = hw_text and len(hw_text.strip()) > 0

        if line_hw_region_type == 'signature':
            # Signature regions: require high TrOCR confidence
            if trocr_valid and hw_conf > surya_conf and hw_conf >= self.SIGNATURE_TROCR_GATE:
                return normalize_punctuation_spacing(hw_text), "trocr"
            return line.text, "surya"
        else:
            # Handwriting regions or page-level ensemble: normal gate
            if trocr_valid and hw_conf > surya_conf:
                return normalize_punctuation_spacing(hw_text), "trocr"
            return line.text, "surya"

    def _build_text_element(self, corrected_text: str, line, source_model: str,
                            signature_info: dict, page_height: int) -> Dict[str, Any]:
        """Construct a text element dictionary with all metadata."""
        text_element: Dict[str, Any] = {
            "type": "text",
            "content": corrected_text,
            "bbox": line.bbox,
            "confidence": line.confidence,
            "source_model": source_model
        }

        # Flag elements in signature region
        if signature_info['has_signature']:
            bbox_y_normalized = line.bbox[1] / page_height if page_height > 0 else 0
            if bbox_y_normalized >= CONFIG.signature_region_ratio:
                text_element["in_signature_region"] = True
                text_element["transcription_uncertain"] = True

        # Add phone and date validation flags
        text_element = add_phone_validation_to_element(text_element)
        text_element = add_date_validation_to_element(text_element)

        return text_element

    def _detect_tables(self, image: Image.Image,
                       page_content: Dict[str, Any]) -> None:
        """Run Table Transformer detection and add tables to page_content."""
        try:
            logger.info("  Running Table Transformer...")
            tables = self.table_recognizer.detect_tables(image)
            for table in tables:
                element = {
                    "type": "table",
                    "bbox": table['bbox'],
                    "confidence": table['confidence']
                }
                # Extract internal structure (rows, columns, cells, headers)
                try:
                    table_crop = crop_image(image, table['bbox'])
                    structure = self.table_recognizer.extract_structure(table_crop)
                    tx1, ty1 = table['bbox'][0], table['bbox'][1]
                    for key in ('rows', 'columns', 'cells', 'headers'):
                        for item in structure[key]:
                            b = item['bbox']
                            item['bbox'] = [b[0] + tx1, b[1] + ty1,
                                            b[2] + tx1, b[3] + ty1]
                    element['structure'] = structure
                except (RuntimeError, ValueError) as e:
                    logger.warning("  Table structure extraction failed: %s", e)
                page_content["elements"].append(element)
        except (RuntimeError, ValueError) as e:
            logger.error("  Table detection failed: %s", e)

    def _detect_visual_elements(self, image: Image.Image, width: int, height: int,
                                page_content: Dict[str, Any]) -> None:
        """Run Florence-2 object detection and phrase grounding for visual elements."""
        try:
            # Generic Object Detection
            od_results = self.run_florence(image, "<OD>")

            if '<OD>' in od_results:
                bboxes = od_results['<OD>']['bboxes']
                labels = od_results['<OD>']['labels']

                for bbox, label in zip(bboxes, labels):
                    label_clean = label.lower()

                    if is_bbox_too_large(bbox, width, height, label=label_clean):
                        continue

                    if any(x in label_clean for x in ['image', 'figure', 'chart', 'plot',
                                                        'diagram', 'map', 'table', 'picture',
                                                        'human face']):
                        crop = crop_image(image, bbox)
                        description = self.run_florence(crop, "<MORE_DETAILED_CAPTION>")
                        caption = description.get('<MORE_DETAILED_CAPTION>', "")

                        page_content["elements"].append({
                            "type": label_clean,
                            "bbox": bbox,
                            "description": caption
                        })

            # Targeted Phrase Grounding for semantic elements
            target_phrases = "logo. seal. signature. graphic."
            pg_results = self.run_florence(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=target_phrases)

            if '<CAPTION_TO_PHRASE_GROUNDING>' in pg_results:
                bboxes = pg_results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
                labels = pg_results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']

                for bbox, label in zip(bboxes, labels):
                    label_clean = label.lower()

                    if is_bbox_too_large(bbox, width, height, label=label_clean):
                        continue

                    page_content["elements"].append({
                        "type": label_clean,
                        "bbox": bbox,
                        "description": f"Detected {label}"
                    })
        except (RuntimeError, ValueError) as e:
            logger.error("  Visual element detection failed: %s", e)
