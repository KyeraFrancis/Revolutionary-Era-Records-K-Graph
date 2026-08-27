# Revolutionary-Era Records Knowledge Graph

OCR fine-tuning, entity extraction, and cross-collection linking over three public,
CC0-licensed archives from the [Revolution Crossroads](https://huggingface.co/RevolutionCrossroads)
project:

| Collection | What it is | Used for |
|---|---|---|
| NARA Revolutionary War pension files | ~2.2M page images with human transcriptions and two machine OCR baselines | Fine-tune and evaluate handwriting OCR; names |
| Smithsonian Revolutionary-era collections | 12.7k museum object records, indexed names/places/dates, VLM OCR on selected images | Entities and artifacts |
| Chronicling America, 1770–1810 | Newspaper issues with OCR text | Contemporary mentions of people and events |

The output is a Neo4j graph where a pension applicant, an artifact that names him, and a
newspaper column that mentions him become one connected `Person` with auditable `SAME_AS` edges.

> This is a public rebuild of the pattern behind professional work I can't share. Same problem
> shape (image-only 18th-century archives, no clean labels, cross-source identity), all-open data.

## Results

_Fill in after running `rcgraph evaluate`. Suggested table:_

| System | CER (all) | WER (all) | CER (handwritten) | CER (typed forms) | n pages |
|---|---|---|---|---|---|
| NARA full-text extraction (`fs_extractedText`) | | | | | |
| Revolution Crossroads VLM OCR (`RevX_OCR_text`) | | | | | |
| **TrOCR fine-tuned (this repo)** | | | | | |

Graph: _N_ records, _N_ people, _N_ cross-collection links (≥ 0.92 similarity).

## Quickstart

```bash
git clone https://github.com/KyeraFrancis/revolution-crossroads-graph && cd revolution-crossroads-graph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env            # set NEO4J_PASSWORD

rcgraph fetch                   # cache the three datasets locally (~minutes; sample of NARA)
rcgraph train                   # fine-tune TrOCR (GPU recommended; ~1–2 h on a single T4)
rcgraph evaluate models/trocr-revwar/best
rcgraph build-graph --use-spacy # extract, link, load Neo4j
```

Or with Docker (starts Neo4j and runs the graph build):

```bash
docker compose up --build
open http://localhost:7474      # Neo4j browser
```

The datasets are public and CC0, so downloads work unauthenticated — the Hub prints a
`You are sending unauthenticated requests to the HF Hub` warning in that case. Set `HF_TOKEN`
in `.env` (a read token from https://huggingface.co/settings/tokens) to silence it and get
higher rate limits and faster downloads.

Example Cypher once loaded:

```cypher
// People who appear in a pension file AND a Smithsonian record
MATCH (r1:Record {collection:'nara'})-[:MENTIONS]->(p:Person)<-[:MENTIONS]-(r2:Record {collection:'smithsonian'})
RETURN p.key, r1.title, r2.title LIMIT 25;

// Fuzzy identity links to review
MATCH (a:Person)-[s:SAME_AS]-(b:Person) WHERE s.reason STARTS WITH 'fuzzy'
RETURN a.key, b.key, s.score ORDER BY s.score LIMIT 50;
```

## Layout

```
src/rcgraph/
  data.py        HF loaders + image download, cached to parquet
  ocr/           dataset.py (band split + preprocessing), finetune.py (TrOCR), predict.py
  metrics.py     CER/WER + per-slice scoring (pure, tested)
  evaluate.py    benchmark vs. the two OCR baselines in the dataset
  entities.py    rules → spaCy → (LLM hook) entity extraction
  link.py        surname blocking + fuzzy matching + year sanity check
  graph.py       Neo4j schema and batched upserts
  cli.py         rcgraph fetch | train | evaluate | build-graph
configs/         default.yaml (all knobs), docker.yaml (override)
tests/           metrics, normalization, entity parsing, linking
docs/            architecture and decisions
```

## What I'd do next

- Replace band splitting with real line segmentation and per-line alignment (biggest CER win).
- Add an LLM relation-extraction pass (served/commanded-by, married-to) as tool-calling agent
  that only escalates pages where rules and spaCy disagree.
- Human-in-the-loop review queue for `SAME_AS` edges under 0.95, feeding corrections back as
  training pairs.
- Drift monitor: CER by page-type over time as new NARA pages are transcribed.

## Data and licensing

All three datasets are CC0 via the Smithsonian's Revolution Crossroads project. Cite:
`doi:10.57967/hf/6529` (NARA), `doi:10.57967/hf/6527` (Smithsonian). Code is MIT.
