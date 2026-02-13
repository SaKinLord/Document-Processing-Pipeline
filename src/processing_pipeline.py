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
    SpellCorrector, classify_document_type, detect_signature_region,
    split_line_bbox_to_words,
)
from src.models.handwriting import HandwritingRecognizer
from src.models.table import TableRecognizer
from src.models.layout import LayoutAnalyzer
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

        # Load Spell Checker
        domain_dict_path = os.path.join(os.path.dirname(__file__), "resources", "domain_dictionary.txt")
        self.spell_corrector = SpellCorrector(domain_dict_path=domain_dict_path)

        # Thresholds
        self.TEXT_CONFIDENCE_THRESHOLD = 0.4

        # Load Florence-2 for VLM tasks (Layout, Captioning)
        self.florence_model_id = "microsoft/Florence-2-large-ft"
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            self.florence_model_id,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device)
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
        logger.info("Loading specialized models (TrOCR, TableTransformer, LayoutLMv3)...")
        self.handwriting_recognizer = HandwritingRecognizer(device=self.device)
        self.table_recognizer = TableRecognizer(device=self.device)
        self.layout_analyzer = LayoutAnalyzer(device=self.device)

    def run_florence(self, image: Image.Image, task_prompt: str,
                     text_input: Optional[str] = None) -> Dict[str, Any]:
        """Run a Florence-2 inference task on an image."""
        if image is None:
            logger.warning("run_florence received None image")
            return {}

        prompt = task_prompt if text_input is None else task_prompt + text_input

        imgs = [image] if isinstance(image, Image.Image) else image
        inputs = self.florence_processor(text=[prompt], images=imgs, return_tensors="pt").to(self.device)

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
        text_elements, ocr_words, ocr_boxes, all_text_content = self._run_ocr_with_routing(
            image, width, height, doc_type, doc_confidence,
            handwritten_regions, signature_info
        )

        # Language detection
        full_text = " ".join(all_text_content)
        page_content["language"] = self.detect_language(full_text)

        # Cluster text into reading order
        clustered_text = cluster_text_rows(text_elements)
        page_content["elements"].extend(clustered_text)

        # Layout analysis (LayoutLMv3)
        self._run_layout_analysis(image, ocr_words, ocr_boxes, page_content)

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
            logger.info("  Signature region detected in bottom %d%% of page", 20)
        return signature_info

    def _detect_handwritten_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect signature and handwritten text regions using Florence-2."""
        # Detect signature bboxes
        signature_bboxes = []
        try:
            sig_results = self.run_florence(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input="signature")
            if '<CAPTION_TO_PHRASE_GROUNDING>' in sig_results:
                for bbox, label in zip(sig_results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'],
                                       sig_results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']):
                    if 'signature' in label.lower():
                        signature_bboxes.append(bbox)
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Detect handwriting bboxes
        handwriting_bboxes = []
        try:
            hw_results = self.run_florence(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input="handwritten text")
            if '<CAPTION_TO_PHRASE_GROUNDING>' in hw_results:
                for bbox, label in zip(hw_results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'],
                                       hw_results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']):
                    handwriting_bboxes.append(bbox)
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Build unified list with type metadata
        handwritten_regions: List[Dict[str, Any]] = []
        for sb in signature_bboxes:
            handwritten_regions.append({'bbox': sb, 'type': 'signature'})

        for hb in handwriting_bboxes:
            overlaps_signature = False
            for sb in signature_bboxes:
                ix1 = max(hb[0], sb[0])
                iy1 = max(hb[1], sb[1])
                ix2 = min(hb[2], sb[2])
                iy2 = min(hb[3], sb[3])
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    hb_area = (hb[2] - hb[0]) * (hb[3] - hb[1])
                    if hb_area > 0 and intersection / hb_area > 0.50:
                        overlaps_signature = True
                        break
            if not overlaps_signature:
                handwritten_regions.append({'bbox': hb, 'type': 'handwriting'})

        if handwriting_bboxes:
            hw_count = len([r for r in handwritten_regions if r['type'] == 'handwriting'])
            logger.info("  Florence-2 handwriting regions: %d detected, %d after dedup",
                        len(handwriting_bboxes), hw_count)
        if signature_bboxes:
            logger.info("  Florence-2 signature regions: %d detected", len(signature_bboxes))

        return handwritten_regions

    def _run_ocr_with_routing(
        self, image: Image.Image, width: int, height: int,
        doc_type: str, doc_confidence: float,
        handwritten_regions: List[Dict[str, Any]],
        signature_info: dict
    ) -> Tuple[List[Dict], List[str], List[List[float]], List[str]]:
        """Run Surya OCR with per-line routing to TrOCR for handwritten regions."""
        predictions = self.rec_predictor([image], ['ocr_with_boxes'], self.det_predictor)
        ocr_result = predictions[0]

        ocr_words: List[str] = []
        ocr_boxes: List[List[float]] = []
        text_elements: List[Dict] = []
        all_text_content: List[str] = []

        for line in ocr_result.text_lines:
            # Lower confidence threshold for lines in handwritten regions
            in_hw_region = any(
                line.bbox[0] < r['bbox'][2] and line.bbox[2] > r['bbox'][0] and
                line.bbox[1] < r['bbox'][3] and line.bbox[3] > r['bbox'][1]
                for r in handwritten_regions
            )
            min_confidence = 0.15 if in_hw_region else self.TEXT_CONFIDENCE_THRESHOLD
            if line.confidence < min_confidence:
                continue

            final_text, source_model = self._route_line(
                line, image, width, height, doc_type, doc_confidence, handwritten_regions
            )

            # Apply spell correction
            corrected_text = self.spell_corrector.correct_text(final_text)

            text_element = self._build_text_element(
                corrected_text, line, source_model, signature_info, height
            )

            text_elements.append(text_element)
            all_text_content.append(corrected_text)

            # Collect word-level data for LayoutLMv3
            line_words = corrected_text.split()
            if line_words:
                word_bboxes = split_line_bbox_to_words(line.bbox, line_words)
                ocr_words.extend(line_words)
                ocr_boxes.extend(word_bboxes)

        return text_elements, ocr_words, ocr_boxes, all_text_content

    def _route_line(self, line, image: Image.Image, width: int, height: int,
                    doc_type: str, doc_confidence: float,
                    handwritten_regions: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Decide which OCR model to use for a text line and return (text, source_model)."""
        final_text = line.text
        source_model = "surya"

        # Check if line overlaps a Florence-2 detected handwritten region
        line_hw_region_type = None
        for region in handwritten_regions:
            rb = region['bbox']
            if (line.bbox[0] < rb[2] and line.bbox[2] > rb[0] and
                line.bbox[1] < rb[3] and line.bbox[3] > rb[1]):
                line_hw_region_type = region['type']
                break

        run_handwriting_model = False
        use_ensemble = False

        if line_hw_region_type is not None:
            use_ensemble = True
        elif doc_type == "handwritten" and doc_confidence >= 0.45:
            run_handwriting_model = True
        elif doc_type == "mixed" or doc_confidence < 0.45:
            use_ensemble = True
        elif doc_type == "typed" and line.confidence < 0.6:
            if len(line.text.strip()) > 0:
                run_handwriting_model = True

        if use_ensemble:
            final_text, source_model = self._run_ensemble(
                line, image, width, height, line_hw_region_type
            )
        elif run_handwriting_model:
            padded_bbox = pad_bbox(line.bbox, 12, width, height)
            line_crop = crop_image(image, padded_bbox)
            hw_text, hw_conf = self.handwriting_recognizer.recognize(line_crop)
            if hw_text and len(hw_text.strip()) > 0 and hw_conf >= 0.15:
                final_text = normalize_punctuation_spacing(hw_text)
                source_model = "trocr"

        return final_text, source_model

    def _run_ensemble(self, line, image: Image.Image, width: int, height: int,
                      line_hw_region_type: Optional[str]) -> Tuple[str, str]:
        """Run both Surya and TrOCR, pick the better result."""
        padded_bbox = pad_bbox(line.bbox, 12, width, height)
        line_crop = crop_image(image, padded_bbox)
        hw_text, hw_conf = self.handwriting_recognizer.recognize(line_crop)

        surya_conf = line.confidence
        trocr_valid = hw_text and len(hw_text.strip()) > 0

        if line_hw_region_type == 'signature':
            # Signature regions: require high TrOCR confidence (0.90 gate)
            if trocr_valid and hw_conf > surya_conf and hw_conf >= 0.90:
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
            if bbox_y_normalized >= 0.75:
                text_element["in_signature_region"] = True
                text_element["transcription_uncertain"] = True

        # Add phone and date validation flags
        text_element = add_phone_validation_to_element(text_element)
        text_element = add_date_validation_to_element(text_element)

        return text_element

    def _run_layout_analysis(self, image: Image.Image, ocr_words: List[str],
                             ocr_boxes: List[List[float]],
                             page_content: Dict[str, Any]) -> None:
        """Run LayoutLMv3 analysis and add layout regions to page_content."""
        try:
            logger.info("  Running LayoutLMv3...")
            layout_regions = self.layout_analyzer.analyze_layout(image, ocr_words, ocr_boxes)

            current_region = None
            for item in layout_regions:
                label = item['label']
                if label == 'O':
                    continue

                if current_region and current_region['type'] == label:
                    b = item['bbox']
                    current_region['bbox'] = [
                        min(current_region['bbox'][0], b[0]),
                        min(current_region['bbox'][1], b[1]),
                        max(current_region['bbox'][2], b[2]),
                        max(current_region['bbox'][3], b[3])
                    ]
                else:
                    if current_region:
                        page_content["elements"].append(current_region)
                    current_region = {
                        "type": "layout_region",
                        "region_type": label,
                        "bbox": item['bbox']
                    }
            if current_region:
                page_content["elements"].append(current_region)

        except Exception as e:
            logger.error("  LayoutLMv3 failed: %s", e)

    def _detect_tables(self, image: Image.Image,
                       page_content: Dict[str, Any]) -> None:
        """Run Table Transformer detection and add tables to page_content."""
        try:
            logger.info("  Running Table Transformer...")
            tables = self.table_recognizer.detect_tables(image)
            for table in tables:
                page_content["elements"].append({
                    "type": "table",
                    "bbox": table['bbox'],
                    "confidence": table['confidence']
                })
        except Exception as e:
            logger.error("  Table detection failed: %s", e)

    def _detect_visual_elements(self, image: Image.Image, width: int, height: int,
                                page_content: Dict[str, Any]) -> None:
        """Run Florence-2 object detection and phrase grounding for visual elements."""
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
