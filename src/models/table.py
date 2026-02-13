import logging
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

logger = logging.getLogger(__name__)


class TableRecognizer:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info("Loading Table Transformer models on %s...", self.device)

        # Detection Model
        self.det_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.det_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(self.device)

        # Structure Model
        self.struct_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.struct_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(self.device)

    def detect_tables(self, image: Image.Image, threshold: float = 0.7) -> List[Dict]:
        """
        Detect tables in a full page image.

        Args:
            image: PIL Image of the full page
            threshold: Minimum confidence to report a detection

        Returns:
            List of dicts with 'bbox' and 'confidence' keys
        """
        inputs = self.det_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.det_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.det_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 0:  # label 0 is table
                tables.append({
                    "bbox": box.tolist(),
                    "confidence": score.item()
                })
        return tables

    def extract_structure(self, table_crop: Image.Image, threshold: float = 0.5) -> Dict:
        """
        Extract rows, columns, and cells from a cropped table image.

        Args:
            table_crop: PIL Image of the cropped table region
            threshold: Minimum confidence for structure elements

        Returns:
            Dict with 'rows', 'columns', 'cells', 'headers' keys
        """
        inputs = self.struct_processor(images=table_crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.struct_model(**inputs)

        target_sizes = torch.tensor([table_crop.size[::-1]])
        results = self.struct_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

        structure: Dict = {"rows": [], "columns": [], "cells": [], "headers": []}

        labels_map = self.struct_model.config.id2label

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = labels_map.get(label.item(), str(label.item()))

            bbox = box.tolist()
            item = {"bbox": bbox, "confidence": score.item()}

            if "row" in label_name.lower() and "header" not in label_name.lower():
                structure["rows"].append(item)
            elif "column" in label_name.lower() and "header" not in label_name.lower():
                structure["columns"].append(item)
            elif "header" in label_name.lower():
                structure["headers"].append(item)
            else:
                structure["cells"].append(item)

        structure["rows"].sort(key=lambda x: x["bbox"][1])
        structure["columns"].sort(key=lambda x: x["bbox"][0])

        return structure
