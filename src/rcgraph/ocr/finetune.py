"""Fine-tune TrOCR on NARA pension pages with human transcriptions as targets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from rcgraph.data import download_images, load_nara
from rcgraph.metrics import cer, wer
from rcgraph.ocr.dataset import OCRDataset, Sample
from rcgraph.text import normalize_for_cer


def build_samples(cfg: dict) -> tuple[list[Sample], list[Sample], pd.DataFrame]:
    c = cfg["data"]["nara"]
    df = load_nara(cfg)
    df["image_path"] = download_images(df, c["image_col"], cfg["data"]["images_dir"])
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    # Split by pension FILE, not by page, so pages from one file never straddle train/eval.
    files = df[c["file_col"]].unique()
    rng = torch.Generator().manual_seed(cfg["ocr"]["seed"])
    perm = torch.randperm(len(files), generator=rng).tolist()
    n_eval = max(1, int(len(files) * cfg["ocr"]["eval_split"]))
    eval_files = {files[i] for i in perm[:n_eval]}
    df["split"] = df[c["file_col"]].map(lambda f: "eval" if f in eval_files else "train")

    def to_samples(d):
        return [Sample(r.image_path, r[c["truth_col"]]) for _, r in d.iterrows()]

    return to_samples(df[df.split == "train"]), to_samples(df[df.split == "eval"]), df


def train(cfg: dict) -> Path:
    o = cfg["ocr"]
    processor = TrOCRProcessor.from_pretrained(o["base_model"])
    model = VisionEncoderDecoderModel.from_pretrained(o["base_model"])
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = o["max_target_len"]
    model.config.num_beams = 4

    train_s, eval_s, df = build_samples(cfg)
    train_ds = OCRDataset(train_s, processor, o["max_target_len"])
    eval_ds = OCRDataset(eval_s, processor, o["max_target_len"])

    def compute_metrics(pred):
        labels = pred.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        hyp = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        ref = processor.batch_decode(labels, skip_special_tokens=True)
        hyp = [normalize_for_cer(h) for h in hyp]
        ref = [normalize_for_cer(r) for r in ref]
        return {"cer": cer(ref, hyp), "wer": wer(ref, hyp)}

    args = Seq2SeqTrainingArguments(
        output_dir=o["output_dir"],
        per_device_train_batch_size=o["batch_size"],
        per_device_eval_batch_size=o["batch_size"],
        num_train_epochs=o["epochs"],
        learning_rate=o["lr"],
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        report_to=[],
        seed=o["seed"],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=default_data_collator, compute_metrics=compute_metrics,
        tokenizer=processor.image_processor,
    )
    trainer.train()
    out = Path(o["output_dir"]) / "best"
    trainer.save_model(out)
    processor.save_pretrained(out)
    df.to_parquet(Path(o["output_dir"]) / "split.parquet", index=False)
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    return out
