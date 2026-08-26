from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from rcgraph.ocr.dataset import preprocess, split_bands


class OCRModel:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(self.device).eval()

    @torch.inference_mode()
    def transcribe(self, image_path: str | Path, n_bands: int = 4) -> str:
        img = preprocess(Image.open(image_path))
        bands = [b for b, _ in split_bands(img, " " * n_bands, n_bands)]
        px = self.processor(images=bands, return_tensors="pt").pixel_values.to(self.device)
        ids = self.model.generate(px, max_length=self.model.config.max_length, num_beams=4)
        return " ".join(self.processor.batch_decode(ids, skip_special_tokens=True)).strip()
