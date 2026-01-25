from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

class HandwritingRecognizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading TrOCR model on {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(self.device)

    def recognize(self, image_crop):
        """
        Recognize handwriting in a cropped image.
        Args:
            image_crop (PIL.Image): Cropped image containing handwritten text.
        Returns:
            str: Recognized text.
        """
        if image_crop is None:
            return ""
            
        try:
            # Ensure RGB
            if image_crop.mode != "RGB":
                image_crop = image_crop.convert("RGB")

            pixel_values = self.processor(images=image_crop, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=128)
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            print(f"Error in handwriting recognition: {e}")
            return ""
