"""Page-image / transcription pairs for TrOCR fine-tuning.

TrOCR is a line-level model. Full pension pages are too tall, so we split each page into
horizontal bands and pair each band with the corresponding slice of the transcription by
proportional character count. This is a deliberate v1 simplification; the honest upgrade is
a line-segmentation step (e.g. Kraken or a projection-profile splitter) with aligned lines.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def preprocess(img: Image.Image) -> Image.Image:
    """Grayscale -> autocontrast -> RGB. Cheap and helps faded ink."""
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    return g.convert("RGB")


def split_bands(img: Image.Image, text: str, n_bands: int = 4) -> list[tuple[Image.Image, str]]:
    w, h = img.size
    band_h = h // n_bands
    n = len(text)
    out = []
    for i in range(n_bands):
        box = (0, i * band_h, w, (i + 1) * band_h if i < n_bands - 1 else h)
        chunk = text[i * n // n_bands:(i + 1) * n // n_bands].strip()
        if chunk:
            out.append((img.crop(box), chunk))
    return out


@dataclass
class Sample:
    image_path: str
    text: str


class OCRDataset(Dataset):
    def __init__(self, samples: list[Sample], processor, max_target_len: int = 256,
                 n_bands: int = 4):
        self.processor = processor
        self.max_target_len = max_target_len
        self.items: list[tuple[Image.Image, str]] = []
        for s in samples:
            try:
                img = preprocess(Image.open(s.image_path))
            except (OSError, ValueError):
                continue
            self.items.extend(split_bands(img, s.text, n_bands))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        img, text = self.items[i]
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_len, truncation=True
        ).input_ids
        labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}
