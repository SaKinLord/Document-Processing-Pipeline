from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image

class TableRecognizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading Table Transformer models on {self.device}...")
        
        # Detection Model
        self.det_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.det_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(self.device)
        
        # Structure Model
        self.struct_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.struct_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(self.device)

    def detect_tables(self, image, threshold=0.7):
        """
        Detect tables in a full page image.
        Returns list of dicts: {'bbox': [x1, y1, x2, y2], 'confidence': score}
        """
        inputs = self.det_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.det_model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.det_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # label 0 is table
            if label.item() == 0:
                tables.append({
                    "bbox": box.tolist(),
                    "confidence": score.item()
                })
        return tables

    def extract_structure(self, table_crop, threshold=0.5):
        """
        Extract rows, columns, and cells from a cropped table image.
        """
        inputs = self.struct_processor(images=table_crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.struct_model(**inputs)
            
        target_sizes = torch.tensor([table_crop.size[::-1]])
        results = self.struct_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
        
        structure = {"rows": [], "columns": [], "cells": [], "headers": []}
        
        # id2label mapping usually: {0: 'table', 1: 'column', 2: 'row', 3: 'column header', 4: 'projected row header', 5: 'spanning cell'}
        # But let's check the model config or assume standard pubtables map
        # Standard PubTables: 0: table, 1: column, 2: row, 3: table col header, 4: table projected row header, 5: table spanning cell
        
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
                structure["cells"].append(item) # Spanning cells mostly
                
        # Sort rows by Y
        structure["rows"].sort(key=lambda x: x["bbox"][1])
        # Sort columns by X
        structure["columns"].sort(key=lambda x: x["bbox"][0])
        
        return structure
