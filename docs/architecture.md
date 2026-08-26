# Architecture

```
Hugging Face (CC0)                      local cache                 outputs
──────────────────                      ───────────                 ───────
NARA pension pages  ──┐                 data/cache/*.parquet        models/trocr-revwar/best
  image + human txt   │                 data/images/*.jpg           reports/ocr_eval.json
Smithsonian objects ──┼─► data.py ──►   ┌──────────────┐            reports/*.predictions.parquet
  metadata + OCR      │                 │ ocr/finetune │─► TrOCR ─► evaluate.py (CER/WER, sliced)
Chronicling America ──┘                 └──────────────┘
  newspaper OCR                                │
                                               ▼
                                        entities.py (rules / spaCy / LLM)
                                               │
                                               ▼
                                        link.py (surname block → fuzzy → year check)
                                               │
                                               ▼
                                        graph.py ──► Neo4j (Person / Place / Year / Record)
```

## Decisions and trade-offs

**Why TrOCR.** Transformer encoder-decoder, strong handwriting checkpoints, fine-tunes on a single
GPU in hours. The dataset already includes a VLM OCR baseline (Chandra OCR 2), so the interesting
question is whether a small, cheap, domain-fine-tuned model can close the gap on this specific
script. That is the same question you face with any edge deployment.

**Band splitting instead of line segmentation.** v1 splits each page into four horizontal bands
and aligns text proportionally. It is crude and puts a ceiling on CER. The upgrade is a proper
line segmenter (Kraken, or a projection-profile splitter) with per-line alignment. Kept simple so
the end-to-end loop works first.

**Split by pension file, not page.** Pages from the same file share hand, ink, and layout. A
page-level split leaks that into eval and overstates performance.

**Sliced evaluation.** CER on "typed form" pages and CER on "handwritten letter" pages are
different problems. The report breaks scores out by `pageImageType` and `levelOfDescription` so a
regression on one slice isn't hidden by an aggregate.

**Linking carries a score and a reason.** Every `SAME_AS` edge stores the fuzzy score and whether
it was exact or fuzzy, so downstream queries can filter and a human can audit low-confidence
merges. Blocking on surname keeps it O(n·k) instead of O(n²).

**Rules before models for entities.** Both NARA and Smithsonian ship curated name fields. Using
them first gives a high-precision skeleton; spaCy/LLM extraction on newspaper text adds recall.
