import logging
from typing import Tuple

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


class HandwritingRecognizer:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info("Loading TrOCR model on %s...", self.device)
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(self.device)

        # CRITICAL FIX: Disable the pooler to prevent randomly initialized weights from affecting inference
        if hasattr(self.model.encoder, 'pooler'):
            self.model.encoder.pooler = None
            logger.info("  Pooler layer disabled to prevent uninitialized weight interference")

    def recognize(self, image_crop: Image.Image) -> Tuple[str, float]:
        """
        Recognize handwriting in a cropped image.

        Args:
            image_crop: Cropped image containing handwritten text.

        Returns:
            Tuple of (recognized_text, confidence) where confidence is 0.0-1.0.
        """
        if image_crop is None:
            return ("", 0.0)

        try:
            if image_crop.mode != "RGB":
                image_crop = image_crop.convert("RGB")

            pixel_values = self.processor(images=image_crop, return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = outputs.sequences
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract confidence from beam search sequence score
            confidence = 0.5  # fallback
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                log_prob = outputs.sequences_scores[0].item()
                confidence = max(0.0, min(1.0, 1.0 + log_prob / 2.5))

            return (generated_text, confidence)
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Error in handwriting recognition: %s", e)
            return ("", 0.0)
