#!/usr/bin/env python3
"""
Confidence in the inference label (paper Section 5.2.4).

Because inference (SYN) dominates the distribution, one concern is that the panel
assigns SYN as a fallback under uncertainty, which would make its prevalence an
artifact of the instrument rather than a property of the arguments. That account
predicts that SYN labels should carry low confidence. This script computes the
mean self-reported confidence (three-point scale: 1 = guessing, 2 = plausible,
3 = clear) for each relation, overall and per genre, so the prediction can be
checked directly against the labels.

The finding reported in the paper is that the prediction is rejected: SYN is
among the highest-confidence labels, comparable to ANT and MER and above PAR and
NONE, so the panel places its uncertainty on NONE and PAR, not on SYN.

Usage:
    python scripts/analyse/confidence.py --labels annotations/labels.jsonl

This measures how confidently the scheme can be applied, not whether the labels
are correct; the confidence ratings are the models' own and are not an
independent measure (see the paper's concluding remarks).
"""
import argparse
import json
import statistics as st
from collections import defaultdict

ORDER = ["SYN", "PAR", "ANT", "MER", "NONE"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarise(rows):
    by = defaultdict(list)
    for r in rows:
        c, d = r.get("confidence"), r.get("dominant")
        if c is not None and d:
            by[d].append(c)
    return by


def report(title, by):
    print(title)
    print(f"  {'relation':8s} {'mean':>6s} {'sd':>6s} {'n':>7s}")
    for k in ORDER:
        v = by.get(k, [])
        if not v:
            print(f"  {k:8s} {'-':>6s} {'-':>6s} {0:7d}")
            continue
        sd = st.pstdev(v) if len(v) > 1 else 0.0
        print(f"  {k:8s} {st.mean(v):6.3f} {sd:6.3f} {len(v):7d}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="annotations/labels.jsonl")
    a = ap.parse_args()
    rows = read_jsonl(a.labels)

    conds = {(r.get("scheme"), r.get("rationale")) for r in rows}
    if len(conds) > 1:
        print(f"! labels hold >1 condition {sorted(conds)}; pooling all rows\n")

    report(f"Mean confidence by dominant relation (all genres, {len(rows)} labels)",
           summarise(rows))

    for genre in ("math", "legal", "everyday"):
        g = [r for r in rows if r.get("genre") == genre]
        if g:
            report(f"Genre: {genre} ({len(g)} labels)", summarise(g))


if __name__ == "__main__":
    main()
