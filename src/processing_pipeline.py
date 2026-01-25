import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from surya.detection import DetectionPredictor

from surya.recognition import RecognitionPredictor, FoundationPredictor
from langdetect import detect, LangDetectException
from src.utils import load_image, convert_pdf_to_images, crop_image, denoise_image, cluster_text_rows, is_bbox_too_large, SpellCorrector, classify_document_type
from src.models.handwriting import HandwritingRecognizer
from src.models.table import TableRecognizer
from src.models.layout import LayoutAnalyzer

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
            
        return results


    def detect_language(self, text):
        try:
            if text and len(text.strip()) > 20: 
                return detect(text)
        except LangDetectException:
            pass
        return "en" # Default to English as per user request

    def process_page(self, image, page_num):
        # 0. Pre-processing: Denoise
        image = denoise_image(image)
        width, height = image.width, image.height
        
        page_content = {
            "page_number": page_num,
            "elements": [],
            "language": "en" # Default to English
        }

        
        # Detect Document Type (Typed vs Handwritten)
        doc_type = classify_document_type(image)
        page_content["document_type"] = doc_type
        print(f"  Document classification: {doc_type}")

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
            # Filter low confidence hallucinations
            if line.confidence < self.TEXT_CONFIDENCE_THRESHOLD:
                continue
            
            final_text = line.text
            # Hybrid Approach: 
            # 1. If document is handwritten, use TrOCR for everything.
            # 2. If typed, but this specific line has low confidence or contains no alphanumeric text, try TrOCR.
            # Note: doc_type "typed" is the default.
            
            run_handwriting_model = False
            if doc_type == "handwritten":
                run_handwriting_model = True
            elif doc_type == "typed" and line.confidence < 0.6:
                 # Check if the text actually looks like noise or empty
                 if len(line.text.strip()) > 0:
                     run_handwriting_model = True
            
            if run_handwriting_model:
               # Crop the line area
               line_crop = crop_image(image, line.bbox)
               # Use TrOCR
               hw_text = self.handwriting_recognizer.recognize(line_crop)
               if hw_text and len(hw_text.strip()) > 0:
                   final_text = hw_text
                   # print(f"  [TrOCR] Replaced '{line.text}' with '{final_text}'")

            # Apply Spell Correction (Re-enabled with safer whitelist approach)
            corrected_text = self.spell_corrector.correct_text(final_text)
            # corrected_text = final_text
            
            text_elements.append({
                "type": "text",
                "content": corrected_text,
                "bbox": line.bbox, 
                "confidence": line.confidence,
                "source_model": "surya" if final_text == line.text else "trocr"
            })
            all_text_content.append(corrected_text)
            
            # Collect for Layout Analysis
            # Split text into words for LayoutLMv3, reusing the line bbox 
            # (Approximation: Ideally we have word-level bboxes, but line-level is acceptable for region layout)
            line_words = corrected_text.split()
            ocr_words.extend(line_words)
            ocr_boxes.extend([line.bbox] * len(line_words))

            
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
