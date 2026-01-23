
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor, FoundationPredictor
from src.utils import load_image, convert_pdf_to_images, crop_image, denoise_image, cluster_text_rows

class DocumentProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading models on {self.device}...")
        
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

    def process_page(self, image, page_num):
        # 0. Pre-processing: Denoise
        image = denoise_image(image)
        
        page_content = {
            "page_number": page_num,
            "elements": []
        }

        # 1. Run Surya OCR for text
        predictions = self.rec_predictor([image], ['ocr_with_boxes'], self.det_predictor)
        # predictions is a list of OCRResult
        ocr_result = predictions[0]
        
        # Add text lines
        text_elements = []
        for line in ocr_result.text_lines:
            text_elements.append({
                "type": "text",
                "content": line.text,
                "bbox": line.bbox, # [x1, y1, x2, y2]
                "confidence": line.confidence
            })
            
        # Apply Table Recovery (Clustering) to text elements
        clustered_text = cluster_text_rows(text_elements)
        page_content["elements"].extend(clustered_text)

        # 2. Run Florence-2 for Object Detection and Layout Analysis
        
        # 2a. Layout Analysis with <OCR_WITH_REGION> (Better for regions)
        # Note: We keep <OD> for general objects, but use this for structure if available
        layout_results = self.run_florence(image, "<OCR_WITH_REGION>")
        if '<OCR_WITH_REGION>' in layout_results:
             # Standard output might be 'quad_boxes' or 'bboxes' depending on post-processing
             # If quad_boxes, we convert to bbox
             # Keys likely: 'quad_boxes', 'labels' (labels are the OCR text usually in this task?)
             # Actually <OCR_WITH_REGION> returns text in 'labels'. We might just want regions.
             # Alternatively, use <REGION_PROPOSAL> or similar? 
             # User requested <OCR_WITH_REGION> specifically.
             # We will store it as 'region' elements.
             
             data = layout_results['<OCR_WITH_REGION>']
             boxes = data.get('quad_boxes', data.get('bboxes', []))
             labels = data.get('labels', [])
             
             for box, label in zip(boxes, labels):
                 # Convert quad to bbox if length is 8
                 if len(box) == 8:
                     xs = box[0::2]
                     ys = box[1::2]
                     bbox = [min(xs), min(ys), max(xs), max(ys)]
                 else:
                     bbox = box
                 
                 # Avoid duplicating dense text if we rely on Surya
                 # But we add it as 'layout_region' for structure
                 page_content["elements"].append({
                     "type": "layout_region",
                     "bbox": bbox,
                     "content_hint": label # The text found by Florence
                 })

        # 2b. Generic Object Detection (Task <OD>)
        od_results = self.run_florence(image, "<OD>")
        
        # Florence-2 OD output format: {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', ...]}}
        if '<OD>' in od_results:
            bboxes = od_results['<OD>']['bboxes']
            labels = od_results['<OD>']['labels']
            
            for bbox, label in zip(bboxes, labels):
                label_clean = label.lower()
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
                
                # Deduplicate: Check IOA (Intersection over Area) with existing elements to avoid double counting
                # (Simple check: if center point is inside an existing bbox)
                
                page_content["elements"].append({
                    "type": label_clean,
                    "bbox": bbox,
                    "description": f"Detected {label}" 
                })

        return page_content
