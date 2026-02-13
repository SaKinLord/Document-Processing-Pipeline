import logging
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info("Loading LayoutLMv3 model on %s...", self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-large").to(self.device)

    def analyze_layout(self, image: Image.Image, ocr_words: List[str],
                       ocr_boxes: List[List[float]]) -> List[Dict]:
        """
        Analyze layout using LayoutLMv3.

        Args:
            image: PIL Image
            ocr_words: List of word strings
            ocr_boxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of dicts with 'text', 'bbox', and 'label' keys
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

        # Map predictions back to words
        word_labels = []
        token_boxes = encoding.word_ids()

        current_word_idx = -1
        for i, pred_label in enumerate(predictions):
            word_idx = token_boxes[i]
            if word_idx is not None and word_idx != current_word_idx:
                predicted_label_name = self.model.config.id2label[pred_label]
                word_labels.append(predicted_label_name)
                current_word_idx = word_idx

        # Align to original word count
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
