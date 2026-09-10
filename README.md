# Semiotic Relations and Proof Methods

Data and code for the paper *Semiotic Relations and Proof Methods: A Cross-Genre
Study of Argument Structure with Large Language Models*.

The study operationalizes the four semiotic relations (syntagmatic, paradigmatic,
antithetic, and meronymic, abbreviated SYN, PAR, ANT, MER) as an explicit set of
definitions and applies them, with a panel of large language models, to a corpus
of arguments from mathematics, law, and everyday reasoning.

## Layout

```
definitions/
  scheme_primary.json     the operational definitions applied in the study
  scheme.py               loads the definitions and assembles the annotation prompt
corpus/
  math.jsonl              627 mathematical items
  legal.jsonl             440 legal items (400 analysed + 40 rare-type, flagged)
  everyday.jsonl          99 everyday items
  manifest.json           SHA-256 and item count per file
annotations/
  labels.jsonl            panel labels: 3 models x 3 runs per item
scripts/
  build/                  corpus construction (build_math, build_legal, build_everyday,
                          corpus_stats, wikitext_clean)
  annotate/               annotate_panel.py
  analyse/                distribution.py
                          combination.py
                          confidence.py
                          external_validity.py
results/                  generated tables, regenerable from scripts/analyse/
```

## Data Formats

**Corpus items** (`corpus/*.jsonl`, one JSON object per line) share `item_id`,
`genre`, `source`, `source_id`, `license`, `citation`, `text`, and `n_words`.
Legal items additionally carry `expert_type` (the expert argument type, used only
by the external check and never shown to a model) and `in_random`: `true` for the
400 analysed items, `false` for the 40 rare-type items reserved for the external
check and excluded from every distribution calculations.

**Labels** (`annotations/labels.jsonl`, one JSON object per line) record, for each
`item_id` x `model` x `run`: the identified `claim` and `support`, the `dominant`
relation, any `subordinate` relations, a `confidence` rating on a three-point
scale (1 = guessing, 2 = plausible, 3 = clear), and a free-text `reasoning`.
Labels join to corpus items on `item_id`.

## Reproducing the Analysis

The repository already ships `corpus/` and `annotations/`, so reproducing the
tables in Section 5 of the paper needs only the analysis step. 

```bash
pip install -r requirements.txt

python scripts/analyse/distribution.py --labels annotations/labels.jsonl --data-dir corpus --csv results/
python scripts/analyse/combination.py --labels annotations/labels.jsonl --data-dir corpus
python scripts/analyse/confidence.py --labels annotations/labels.jsonl
python scripts/analyse/external_validity.py --labels annotations/labels.jsonl --legal corpus/legal.jsonl
```

## Rebuilding the Corpus and Re-running Annotation

Full reproduction from the original sources is a three-stage pipeline. Stages 1
and 2 are needed only to regenerate `corpus/` and `annotations/` from scratch;
they require the original datasets that are not redistributed here. 
Run the stages in order.

**Stage 1: build and freeze the corpus.**

```bash
python scripts/build/build_math.py     --path <naturalproofs_proofwiki.json> --out corpus/math.jsonl
python scripts/build/build_legal.py    --repo <mining-legal-arguments>       --out corpus/legal.jsonl
python scripts/build/build_everyday.py --path <arg_microtexts_en>            --out corpus/everyday.jsonl
python scripts/build/corpus_stats.py check  --data-dir corpus
python scripts/build/corpus_stats.py freeze --data-dir corpus --version 2.0
```

**Stage 2: annotate with the model panel.**

```bash
python scripts/annotate/annotate_panel.py \
    --corpora corpus/math.jsonl corpus/everyday.jsonl corpus/legal.jsonl \
    --scheme definitions/scheme_primary.json \
    --base-url http://localhost:8000/v1 \
    --models Qwen3.8-27B GLM-5.2-753B gemma-4-31b --runs 3 --rationale \
    --out annotations/labels.jsonl
```

**Stage 3: analysis.** Run the four analysis commands from the previous section.

```bash
python scripts/analyse/distribution.py --labels annotations/labels.jsonl --data-dir corpus --csv results/
python scripts/analyse/combination.py --labels annotations/labels.jsonl --data-dir corpus
python scripts/analyse/confidence.py --labels annotations/labels.jsonl
python scripts/analyse/external_validity.py --labels annotations/labels.jsonl --legal corpus/legal.jsonl
```
