from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

class HandwritingRecognizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading TrOCR model on {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(self.device)
        
        # CRITICAL FIX: Disable the pooler to prevent randomly initialized weights from affecting inference
        # The pooler is not used by the decoder (which uses last_hidden_state for cross-attention)
        # but it IS computed during forward pass, potentially corrupting the hidden states
        if hasattr(self.model.encoder, 'pooler'):
            self.model.encoder.pooler = None
            print("  Pooler layer disabled to prevent uninitialized weight interference")

    def recognize(self, image_crop):
        """
        Recognize handwriting in a cropped image.
        Args:
            image_crop (PIL.Image): Cropped image containing handwritten text.
        Returns:
            tuple: (recognized_text, confidence) where confidence is 0.0-1.0.
        """
        if image_crop is None:
            return ("", 0.0)

        try:
            # Ensure RGB
            if image_crop.mode != "RGB":
                image_crop = image_crop.convert("RGB")

            pixel_values = self.processor(images=image_crop, return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                # Optimized generation with beam search for better quality
                outputs = self.model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,                 # Beam search for better quality
                    early_stopping=True,         # Stop at first complete sequence
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = outputs.sequences
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract confidence from beam search sequence score
            # sequences_scores is length-normalized sum of log-probs
            confidence = 0.5  # fallback
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                log_prob = outputs.sequences_scores[0].item()
                # Map normalized log-prob to [0, 1]:
                #   log_prob ≈  0.0 → conf = 1.00 (perfect)
                #   log_prob ≈ -0.5 → conf = 0.80
                #   log_prob ≈ -1.25 → conf = 0.50
                #   log_prob ≈ -2.5 → conf = 0.00 (garbage)
                confidence = max(0.0, min(1.0, 1.0 + log_prob / 2.5))

            return (generated_text, confidence)
        except Exception as e:
            print(f"Error in handwriting recognition: {e}")
            return ("", 0.0)
