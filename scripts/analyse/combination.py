#!/usr/bin/env python3
"""
Internal-structure figures for the evaluation section (5.5): combination rate
broken down by the dominant relation, and the leading dominant->subordinate
compounding pairs. distribution.py reports combination only by genre; this adds
the by-dominant view that shows SYN as the unmarked default and ANT/MER as the
marked departures that embed inference.

Reads the frozen labels; standard library only. Blind items only, matching the
distribution figures (the 40 boosted legal spans are excluded).

    python combination_structure.py
    python combination_structure.py --labels output/labels.jsonl --data-dir data
"""
import argparse
import json
import os
from collections import Counter, defaultdict


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="annotations/labels.jsonl")
    ap.add_argument("--data-dir", default="corpus")
    a = ap.parse_args()

    rows = load_jsonl(a.labels)
    corpus = {}
    for name in ("math.jsonl", "everyday.jsonl",
                 "legal.jsonl"):
        p = os.path.join(a.data_dir, name)
        if os.path.exists(p):
            for r in load_jsonl(p):
                corpus[r["item_id"]] = r

    def blind(iid):
        it = corpus.get(iid, {})
        return it.get("genre") != "legal" or bool(it.get("in_random"))

    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)

    dom_total = Counter()
    dom_with_sub = Counter()
    pairs = Counter()
    for iid, rs in by_item.items():
        if not blind(iid):
            continue
        n = len(rs)
        dom = Counter(r["dominant"] for r in rs).most_common(1)[0][0]
        dom_total[dom] += 1
        # subordinate present in a MAJORITY of votes (same rule as 06_analyse)
        if sum(1 for r in rs if r.get("subordinate")) > n / 2:
            dom_with_sub[dom] += 1
            subc = Counter()
            for r in rs:
                for s in (r.get("subordinate") or []):
                    subc[s] += 1
            for s, c in subc.items():
                if c > n / 2:
                    pairs[(dom, s)] += 1

    print("Combination rate by dominant relation (majority subordinate / "
          "total), blind items:")
    for d in ["SYN", "MER", "ANT", "PAR", "NONE"]:
        if dom_total[d]:
            print(f"  {d:4s} {dom_with_sub[d]:4d}/{dom_total[d]:<4d} = "
                  f"{dom_with_sub[d]/dom_total[d]:5.1%}")

    print("\nLeading dominant -> subordinate pairs (both in a majority of "
          "votes), blind items:")
    for (d, s), c in pairs.most_common(12):
        print(f"  {d} + {s}: {c}")


if __name__ == "__main__":
    main()
