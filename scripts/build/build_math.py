#!/usr/bin/env python3
"""
Build the math genre corpus from the NaturalProofs ProofWiki dump.

The cleaning logic lives in wikitext_clean.py and is imported verbatim rather
than copied, so the diagnostic and the builder cannot drift apart. Keep both
files in the same directory.

    python scripts/build/build_math.py --path <naturalproofs_proofwiki.json> --out corpus/math.jsonl

Every item in the pool is cue-agnostic: the cue fields are recorded per item
but nothing is filtered on them, so random draws from this file give an
unbiased picture of the genre. That is what Sample A needs.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cleaner():
    p = os.path.join(HERE, "wikitext_clean.py")
    if not os.path.exists(p):
        sys.exit("wikitext_clean.py must be in the same directory as this script.")
    spec = importlib.util.spec_from_file_location("wikitext_clean", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="naturalproofs_proofwiki.json")
    ap.add_argument("--out", default="corpus/math.jsonl")
    ap.add_argument("--all-subjects", action="store_true")
    args = ap.parse_args()

    c = load_cleaner()
    ds = json.load(open(args.path, encoding="utf-8")).get("dataset", {})
    theorems = ds.get("theorems", [])

    items, n_subject, n_excl = [], 0, 0
    n_raw_band, n_dup = 0, 0
    seen_text = {}
    for t in theorems:
        proofs = t.get("proofs") or []
        if not proofs:
            continue
        cats = t.get("categories") or []
        if not args.all_subjects and not any(x in c.ACCESSIBLE for x in cats):
            continue
        n_subject += 1
        raw = c.flatten(t.get("contents")) + " " + c.flatten(proofs[0])
        text, reasons = c.inspect(raw)
        if reasons:
            n_excl += 1
            continue

        # Two length measures, because they disagree and both matter.
        # prose tokens collapse each $...$ span to one token, so they track
        # argument content; raw words are what an annotator actually reads.
        # A proof can sit inside the prose band and still be 1200 tokens of
        # LaTeX on the page, which is unlabellable and out of band against the
        # other genres. Require both.
        n_prose, ratio = c.prose_stats(text)
        n_words = len(text.split())
        if not (c.WORD_MIN <= n_prose <= c.WORD_MAX):
            n_excl += 1
            continue
        if not (c.WORD_MIN <= n_words <= c.WORD_MAX):
            n_raw_band += 1
            continue

        # NaturalProofs reaches some theorems under more than one title.
        key = re.sub(r"\s+", " ", text.strip().lower())
        if key in seen_text:
            n_dup += 1
            continue
        seen_text[key] = t.get("title", "")

        items.append({
            "genre": "math",
            "source": "naturalproofs-proofwiki",
            "source_id": t.get("title", ""),
            "license": "CC BY-SA 4.0",
            "citation": "Welleck et al. (2021), NaturalProofs; ProofWiki",
            "text": text,
            "n_words": n_words,
            "n_prose_tokens": n_prose,
            "prose_ratio": round(ratio, 3),
            "categories": cats,
            "cues": [k for k, rx in c.CUES.items() if rx.search(text)],
        })

    for i, it in enumerate(items):
        it["item_id"] = f"math-{i:03d}"

    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    cue = Counter()
    for it in items:
        for k in (it["cues"] or ["none"]):
            cue[k] += 1

    print(f"accessible subjects  {n_subject}")
    print(f"excluded (clean/prose band) {n_excl}")
    print(f"excluded (raw word band)    {n_raw_band}")
    print(f"excluded (duplicate text)   {n_dup}")
    print(f"written              {len(items)} -> {args.out}")
    if items:
        raws = sorted(i["n_words"] for i in items)
        pros = sorted(i["n_prose_tokens"] for i in items)
        med = lambda x: x[len(x)//2]
        print(f"  raw words:    min {raws[0]} med {med(raws)} max {raws[-1]}")
        print(f"  prose tokens: min {pros[0]} med {med(pros)} max {pros[-1]}")
    print("\ncue distribution (recorded, NOT filtered on):")
    for k, v in cue.most_common():
        print(f"  {k:6s} {v:5d}")
    print("\nCues are heuristics for the boosted sample only. Draws for Sample A "
          "should ignore them entirely.")


if __name__ == "__main__":
    main()
