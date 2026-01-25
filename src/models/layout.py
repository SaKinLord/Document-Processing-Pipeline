from transformers import AutoProcessor, AutoModelForTokenClassification
import torch
import numpy as np

class LayoutAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading LayoutLMv3 model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-large").to(self.device)
        
    def analyze_layout(self, image, ocr_words, ocr_boxes):
        """
        Analyze layout using LayoutLMv3.
        Args:
            image: PIL Image
            ocr_words: List of strings
            ocr_boxes: List of [x1, y1, x2, y2]
        """
        if not ocr_words:
            return []
            
        width, height = image.size
        
        # Normalize boxes to 0-1000
        normalized_boxes = []
        for box in ocr_boxes:
            normalized_boxes.append([
                max(0, min(1000, int(box[0] * 1000 / width))),
                max(0, min(1000, int(box[1] * 1000 / height))),
                max(0, min(1000, int(box[2] * 1000 / width))),
                max(0, min(1000, int(box[3] * 1000 / height)))
            ])
            
        encoding = self.processor(
            image,
            ocr_words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        bbox = encoding["bbox"].to(self.device)
        pixel_values = encoding["pixel_values"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                bbox=bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask
            )
            
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        # Map back to words (simplified mapping, assumes 1-to-1 token mapping which isn't always true due to subword tokenization)
        # Real implementation needs alignment. For now, we take the label of the first token of the word.
        
        word_labels = []
        token_boxes = encoding.word_ids() # Identify which word each token belongs to
        
        current_word_idx = -1
        for i, pred_label in enumerate(predictions):
            word_idx = token_boxes[i]
            if word_idx is not None and word_idx != current_word_idx:
                # Valid token for a new word
                predicted_label_name = self.model.config.id2label[pred_label]
                word_labels.append(predicted_label_name)
                current_word_idx = word_idx
                
        # Trim or pad to match original length if needed (subword misalignment fix)
        if len(word_labels) < len(ocr_words):
            word_labels.extend(["O"] * (len(ocr_words) - len(word_labels)))
        elif len(word_labels) > len(ocr_words):
            word_labels = word_labels[:len(ocr_words)]
            
        results = []
        for word, box, label in zip(ocr_words, ocr_boxes, word_labels):
            results.append({
                "text": word,
                "bbox": box,
                "label": label
            })
            
        return results
