"""Command-line entry points. `rcgraph --help`."""
from __future__ import annotations

from pathlib import Path

import typer

from rcgraph.config import load_config

app = typer.Typer(no_args_is_help=True, help="Revolutionary-era OCR + knowledge graph pipeline.")


@app.command()
def fetch(config: Path | None = None):
    """Pull the three collections from Hugging Face and cache locally."""
    from rcgraph.data import load_chronam, load_nara, load_smithsonian
    cfg = load_config(config)
    loaders = [("nara", load_nara), ("smithsonian", load_smithsonian), ("chronam", load_chronam)]
    for name, fn in loaders:
        df = fn(cfg)
        typer.echo(f"{name}: {len(df):,} rows, columns={list(df.columns)[:8]}...")


@app.command()
def train(config: Path | None = None):
    """Fine-tune TrOCR on NARA pension pages (needs `pip install -e .[ocr]`)."""
    from rcgraph.ocr.finetune import train as _train
    out = _train(load_config(config))
    typer.echo(f"saved best model to {out}")


@app.command()
def evaluate(model_dir: Path, config: Path | None = None, limit: int | None = None):
    """Score the fine-tuned model vs. dataset baselines, sliced by page type."""
    from rcgraph.evaluate import evaluate as _eval
    report = _eval(load_config(config), model_dir, limit=limit)
    typer.echo(report[report.slice == "all"].to_string(index=False))


@app.command()
def build_graph(config: Path | None = None, use_spacy: bool = False, limit: int | None = None):
    """Extract entities from all three collections, link people, and load Neo4j."""
    from rcgraph.data import load_chronam, load_nara, load_smithsonian
    from rcgraph.entities import extract_rules, extract_spacy
    from rcgraph.graph import GraphLoader
    from rcgraph.link import Mention, link_mentions

    cfg = load_config(config)
    c = cfg["data"]
    nlp = None
    if use_spacy:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    records: list[tuple[dict, object]] = []
    mentions: list[Mention] = []

    nara = load_nara(cfg, with_truth_only=False)
    if limit:
        nara = nara.head(limit)
    for _, r in nara.iterrows():
        text = r.get(c["nara"]["truth_col"]) or r.get("RevX_OCR_text")
        ents = extract_rules(text, r.get(c["nara"]["names_col"]))
        rid = f"nara:{r[c['nara']['id_col']]}"
        records.append(({"id": rid, "collection": "nara", "title": r.get("title"),
                         "url": f"https://catalog.archives.gov/id/{r[c['nara']['file_col']]}",
                         "source": "extractedNames"}, ents))
        mentions += [Mention("nara", rid, p, frozenset(ents.years)) for p in ents.people]

    si = load_smithsonian(cfg)
    if limit:
        si = si.head(limit)
    s = c["smithsonian"]
    for _, r in si.iterrows():
        text = r.get(s["text_col"])
        if isinstance(text, (list, tuple)) or hasattr(text, "tolist"):
            text = " ".join(t for t in text if t)
        ents = extract_rules(text, r.get(s["names_col"]), r.get(s["places_col"]))
        for d in (r.get(s["dates_col"]) if r.get(s["dates_col"]) is not None else []):
            if str(d)[:4].isdigit():
                ents.years.add(int(str(d)[:4]))
        rid = f"si:{r[s['id_col']]}"
        records.append(({"id": rid, "collection": "smithsonian", "title": r.get("title"),
                         "url": r.get("collectionsURL"), "source": "indexed_names"}, ents))
        mentions += [Mention("smithsonian", rid, p, frozenset(ents.years)) for p in ents.people]

    ca = load_chronam(cfg)
    if limit:
        ca = ca.head(limit)
    for _, r in ca.iterrows():
        ents = extract_spacy(r["text"], nlp) if use_spacy else extract_rules(r["text"])
        rid = f"chronam:{r['id']}"
        records.append(({"id": rid, "collection": "chronam", "title": r.get("title"),
                         "url": None, "source": "spacy" if use_spacy else "rules"}, ents))
        mentions += [Mention("chronam", rid, p, frozenset(ents.years)) for p in ents.people]

    links = link_mentions(mentions, threshold=cfg["link"]["name_threshold"])
    typer.echo(f"records={len(records):,} mentions={len(mentions):,} "
               f"cross-collection links={len(links):,}")

    g = cfg["graph"]
    loader = GraphLoader(g["uri"], g["user"], g["password"], g["batch_size"])
    try:
        loader.init_schema()
        loader.load_records(records)
        loader.load_links(links)
    finally:
        loader.close()
    typer.echo("graph loaded")


if __name__ == "__main__":
    app()
