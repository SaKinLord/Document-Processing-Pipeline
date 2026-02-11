import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from surya.detection import DetectionPredictor

from surya.recognition import RecognitionPredictor, FoundationPredictor
from langdetect import detect, LangDetectException
from src.utils import load_image, convert_pdf_to_images, crop_image, pad_bbox, denoise_image, deskew_image, cluster_text_rows, is_bbox_too_large, SpellCorrector, classify_document_type, detect_signature_region, split_line_bbox_to_words
from src.models.handwriting import HandwritingRecognizer
from src.models.table import TableRecognizer
from src.models.layout import LayoutAnalyzer
from src.postprocessing import normalize_punctuation_spacing, add_phone_validation_to_element, add_date_validation_to_element

class DocumentProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading models on {self.device}...")
        
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
        print("Loading Surya predictors...")
        self.det_predictor = DetectionPredictor()
        
        # RecognitionPredictor requires a FoundationPredictor
        self.rec_foundation = FoundationPredictor()
        self.rec_predictor = RecognitionPredictor(self.rec_foundation)
        
        # Load New Specialized Models
        print("Loading specialized models (TrOCR, TableTransformer, LayoutLMv3)...")
        self.handwriting_recognizer = HandwritingRecognizer(device=self.device)
        self.table_recognizer = TableRecognizer(device=self.device)
        self.layout_analyzer = LayoutAnalyzer(device=self.device)

    def run_florence(self, image, task_prompt, text_input=None):
        if image is None:
            print("Warning: run_florence received None image")
            return {}
            
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
            
        # Wrap image in list if it's a single PIL image
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

    def process_document(self, file_path):
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
            print(f"Processing page {i+1}...")
            page_data = self.process_page(image, page_num=i+1)
            results["pages"].append(page_data)
            
        return results, images


    def detect_language(self, text):
        try:
            if text and len(text.strip()) > 20: 
                return detect(text)
        except LangDetectException:
            pass
        return "en" # Default to English as per user request

    def process_page(self, image, page_num):
        # 0. Pre-processing: Denoise and Deskew
        image = denoise_image(image)
        image = deskew_image(image)
        width, height = image.width, image.height
        
        page_content = {
            "page_number": page_num,
            "elements": [],
            "language": "en" # Default to English
        }

        
        # Detect Document Type (Typed vs Handwritten) - Now returns (type, confidence)
        doc_type, doc_confidence = classify_document_type(image)
        page_content["document_type"] = doc_type
        page_content["classification_confidence"] = doc_confidence
        print(f"  Document classification: {doc_type} (confidence: {doc_confidence:.2f})")

        # Detect signature region for flagging
        signature_info = detect_signature_region(image)
        if signature_info['has_signature']:
            page_content["signature_detected"] = True
            print(f"  Signature region detected in bottom 20% of page")

        # Detect signature bboxes with Florence-2 for ensemble gating
        # This gives us actual pixel-level signature locations (not just bottom 20%)
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

        # Detect handwritten regions with Florence-2 for line-level routing
        # Catches fill-in values, annotations, and names on typed forms
        HANDWRITING_GROUNDING_PROMPT = "handwritten text"
        handwriting_bboxes = []
        try:
            hw_results = self.run_florence(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=HANDWRITING_GROUNDING_PROMPT)
            if '<CAPTION_TO_PHRASE_GROUNDING>' in hw_results:
                for bbox, label in zip(hw_results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'],
                                       hw_results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']):
                    handwriting_bboxes.append(bbox)
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Build unified handwritten_regions list with type metadata
        # Signature regions get stricter TrOCR gating (0.90), handwriting gets normal ensemble
        handwritten_regions = []
        for sb in signature_bboxes:
            handwritten_regions.append({'bbox': sb, 'type': 'signature'})

        for hb in handwriting_bboxes:
            # De-duplicate: skip handwriting bboxes that overlap >50% with a signature bbox
            overlaps_signature = False
            for sb in signature_bboxes:
                # Calculate overlap ratio (intersection / handwriting bbox area)
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
            print(f"  Florence-2 handwriting regions: {len(handwriting_bboxes)} detected, {len([r for r in handwritten_regions if r['type'] == 'handwriting'])} after dedup")
        if signature_bboxes:
            print(f"  Florence-2 signature regions: {len(signature_bboxes)} detected")

        # 1. Run Surya OCR for text
        predictions = self.rec_predictor([image], ['ocr_with_boxes'], self.det_predictor)
        ocr_result = predictions[0]
        
        # Prepare data for LayoutLMv3
        ocr_words = []
        ocr_boxes = []
        
        # Add text lines
        text_elements = []
        all_text_content = [] 
        
        for line in ocr_result.text_lines:
            # Lower confidence threshold for lines overlapping handwriting/signature regions
            # Surya may detect handwritten text but with low confidence — let it through
            # so the per-line routing can send it to TrOCR
            in_hw_region = any(
                line.bbox[0] < r['bbox'][2] and line.bbox[2] > r['bbox'][0] and
                line.bbox[1] < r['bbox'][3] and line.bbox[3] > r['bbox'][1]
                for r in handwritten_regions
            )
            min_confidence = 0.15 if in_hw_region else self.TEXT_CONFIDENCE_THRESHOLD
            if line.confidence < min_confidence:
                continue
            
            final_text = line.text
            source_model = "surya"
            
            # --- Per-line routing: check Florence-2 handwritten regions first ---
            # If a Surya line overlaps a detected handwritten region, force ensemble
            # regardless of page-level classification. This lets TrOCR handle
            # handwritten fill-ins, names, and annotations on typed forms.
            line_hw_region_type = None  # None, 'signature', or 'handwriting'
            for region in handwritten_regions:
                rb = region['bbox']
                if (line.bbox[0] < rb[2] and line.bbox[2] > rb[0] and
                    line.bbox[1] < rb[3] and line.bbox[3] > rb[1]):
                    line_hw_region_type = region['type']
                    break  # Use first overlapping region (signature takes priority due to list order)

            run_handwriting_model = False
            use_ensemble = False

            if line_hw_region_type is not None:
                # Line overlaps a Florence-2 detected handwritten region → ensemble
                use_ensemble = True
            elif doc_type == "handwritten" and doc_confidence >= 0.45:
                run_handwriting_model = True
            elif doc_type == "mixed" or doc_confidence < 0.45:
                use_ensemble = True
            elif doc_type == "typed" and line.confidence < 0.6:
                if len(line.text.strip()) > 0:
                    run_handwriting_model = True

            if use_ensemble:
                # Run both OCR engines, pick the one with higher confidence
                padded_bbox = pad_bbox(line.bbox, 12, width, height)
                line_crop = crop_image(image, padded_bbox)
                hw_text, hw_conf = self.handwriting_recognizer.recognize(line_crop)

                surya_conf = line.confidence
                trocr_valid = hw_text and len(hw_text.strip()) > 0

                if line_hw_region_type == 'signature':
                    # Signature regions: require high TrOCR confidence (0.90 gate)
                    # to prevent hallucinations on cursive strokes
                    if trocr_valid and hw_conf > surya_conf and hw_conf >= 0.90:
                        final_text = normalize_punctuation_spacing(hw_text)
                        source_model = "trocr"
                    else:
                        final_text = line.text
                        source_model = "surya"
                else:
                    # Handwriting regions or page-level ensemble: normal gate
                    if trocr_valid and hw_conf > surya_conf:
                        final_text = normalize_punctuation_spacing(hw_text)
                        source_model = "trocr"
                    else:
                        final_text = line.text
                        source_model = "surya"

            elif run_handwriting_model:
               # Crop the line area with padding to prevent cutting off descenders
               padded_bbox = pad_bbox(line.bbox, 12, width, height)
               line_crop = crop_image(image, padded_bbox)
               # Use TrOCR
               hw_text, hw_conf = self.handwriting_recognizer.recognize(line_crop)
               if hw_text and len(hw_text.strip()) > 0 and hw_conf >= 0.15:
                   # Normalize punctuation spacing in TrOCR output
                   final_text = normalize_punctuation_spacing(hw_text)
                   source_model = "trocr"

            # Apply Spell Correction (Re-enabled with safer whitelist approach)
            corrected_text = self.spell_corrector.correct_text(final_text)
            # corrected_text = final_text
            
            text_element = {
                "type": "text",
                "content": corrected_text,
                "bbox": line.bbox,
                "confidence": line.confidence,
                "source_model": source_model
            }

            # Flag elements in signature region as having uncertain transcription
            # Signature region is bottom 20% of page (y >= 0.80 normalized)
            if signature_info['has_signature']:
                # Normalize bbox y-coordinates to 0-1 range
                bbox_y_normalized = line.bbox[1] / height if height > 0 else 0
                if bbox_y_normalized >= 0.75:  # Allow some margin (75% instead of 80%)
                    text_element["in_signature_region"] = True
                    text_element["transcription_uncertain"] = True

            # Add phone and date validation (flags only, no auto-correct)
            text_element = add_phone_validation_to_element(text_element)
            text_element = add_date_validation_to_element(text_element)
            
            text_elements.append(text_element)
            all_text_content.append(corrected_text)
            
            # Collect for Layout Analysis
            # Split text into words for LayoutLMv3 with word-level bbox estimation
            # This improves LayoutLMv3's spatial reasoning by providing distinct bboxes per word
            line_words = corrected_text.split()
            if line_words:
                word_bboxes = split_line_bbox_to_words(line.bbox, line_words)
                ocr_words.extend(line_words)
                ocr_boxes.extend(word_bboxes)


        # Detect Language
        full_text = " ".join(all_text_content)
        page_content["language"] = self.detect_language(full_text)
            
        # Apply Table Recovery (Clustering) to text elements
        clustered_text = cluster_text_rows(text_elements)
        page_content["elements"].extend(clustered_text)

        # 2. Layout Analysis: LayoutLMv3 (Primary) instead of Florence
        try:
            print("  Running LayoutLMv3...")
            layout_regions = self.layout_analyzer.analyze_layout(image, ocr_words, ocr_boxes)
            # Consolidate nearby regions or just add them
            # LayoutLMv3 returns word-level labels. We need to aggregate them into regions.
            # Simple heuristic: Group consecutive words with same label.
            
            current_region = None
            for item in layout_regions:
                label = item['label']
                if label == 'O': continue 
                
                if current_region and current_region['type'] == label:
                    # Extend bbox
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
            print(f"  LayoutLMv3 failed: {e}")
            # Fallback? No, Florence often crashes OOM.
            
        # 3. Targeted Object Detection (Florence-2) - CAREFUL MEMORY USE
        torch.cuda.empty_cache() # Crucial for T4 16GB
        
        # Only run OD if really needed (Tables are handled by TableTransformer - coming soon)
        # Actually, let's run TableTransformer here instead of Florence for tables
        try:
            print("  Running Table Transformer...")
            tables = self.table_recognizer.detect_tables(image)
            for table in tables:
                page_content["elements"].append({
                    "type": "table",
                    "bbox": table['bbox'],
                    "confidence": table['confidence']
                })
        except Exception as e:
             print(f"  Table detection failed: {e}")

        # Disable broad Florence OD to save memory for now, or ensure we clear cache
        # od_results = self.run_florence(image, "<OD>") ... (Skipping to prevent OOM)
        
        # Note: If user specifically needs Florence, we need to offload others first. 
        # For now, prioritizing Robustness (No crash) over extra detection.

        # 2b. Generic Object Detection (Task <OD>)
        od_results = self.run_florence(image, "<OD>")
        
        # Florence-2 OD output format: {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', ...]}}
        if '<OD>' in od_results:
            bboxes = od_results['<OD>']['bboxes']
            labels = od_results['<OD>']['labels']
            
            for bbox, label in zip(bboxes, labels):
                label_clean = label.lower()
                
                # False Positive Filter: Check Size
                if is_bbox_too_large(bbox, width, height, label=label_clean):
                    continue

                # Heuristic: Identify non-text visual elements
                # Added 'human face' based on user data
                if any(x in label_clean for x in ['image', 'figure', 'chart', 'plot', 'diagram', 'map', 'table', 'picture', 'human face']):
                    
                    # Generate description for this crop
                    crop = crop_image(image, bbox)
                    description = self.run_florence(crop, "<MORE_DETAILED_CAPTION>")
                    caption = description.get('<MORE_DETAILED_CAPTION>', "")
                    
                    page_content["elements"].append({
                        "type": label_clean,
                        "bbox": bbox,
                        "description": caption
                    })

        # 2c. Targeted Phrase Grounding for Semantic Elements (Logos, Seals, Signatures)
        # Task <CAPTION_TO_PHRASE_GROUNDING> requires a text input.
        target_phrases = "logo. seal. signature. graphic."
        pg_results = self.run_florence(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=target_phrases)
        
        if '<CAPTION_TO_PHRASE_GROUNDING>' in pg_results:
            bboxes = pg_results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
            labels = pg_results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
            
            for bbox, label in zip(bboxes, labels):
                # Florence returns the matched phrase as the label
                label_clean = label.lower()
                
                # False Positive Filter: Check Size
                if is_bbox_too_large(bbox, width, height, label=label_clean):
                    continue
                
                # Deduplicate: Check IOA (Intersection over Area) with existing elements to avoid double counting
                # (Simple check: if center point is inside an existing bbox)
                
                page_content["elements"].append({
                    "type": label_clean,
                    "bbox": bbox,
                    "description": f"Detected {label}" 
                })

        return page_content
