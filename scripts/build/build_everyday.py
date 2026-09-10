#!/usr/bin/env python3
"""
Build the everyday genre corpus from the Argumentative Microtext Corpus.

    python scripts/build/build_everyday.py --out corpus/everyday.jsonl

Downloads Peldszus & Stede's corpus from GitHub (or reads a local copy with
--path) and writes the in-band English texts. No cleaning pipeline is needed:
the corpus ships as plain text, already segmented into complete single
arguments, which is why this file is a fraction of the size of the other two
builders.

Licence: CC BY-NC-SA 4.0. Non-commercial, which is fine for a preprint but
means a merged corpus cannot be redistributed under one permissive licence.
Release annotations keyed by item_id instead.
"""

import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile

URL = "https://codeload.github.com/peldszus/arg-microtexts/tar.gz/refs/heads/master"
WORD_MIN, WORD_MAX = 50, 300


def fetch(dest):
    import requests
    print(f"downloading {URL}")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as t:
        members = [m for m in t.getmembers()
                   if "/corpus/en/" in m.name and m.name.endswith(".txt")]
        if not members:
            sys.exit("no English texts found in the archive; layout may have changed")
        for m in members:
            m.name = os.path.basename(m.name)
        t.extractall(dest, members=members)
    print(f"  extracted {len(members)} texts to {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="local corpus/en directory; downloads if omitted")
    ap.add_argument("--cache", default="arg_microtexts_en")
    ap.add_argument("--out", default="corpus/everyday.jsonl")
    args = ap.parse_args()

    src = args.path
    if not src:
        src = args.cache
        if os.path.isdir(src) and glob.glob(os.path.join(src, "*.txt")):
            print(f"using cached {src}")
        else:
            os.makedirs(src, exist_ok=True)
            fetch(src)

    files = sorted(glob.glob(os.path.join(src, "*.txt")))
    if not files:
        sys.exit(f"no .txt files in {src}")

    items, short, long_ = [], 0, 0
    for f in files:
        text = re.sub(r"\s+", " ", open(f, encoding="utf-8").read()).strip()
        n = len(text.split())
        if n < WORD_MIN:
            short += 1
            continue
        if n > WORD_MAX:
            long_ += 1
            continue
        items.append({
            "item_id": f"everyday-{len(items):03d}",
            "genre": "everyday",
            "source": "arg-microtexts",
            "source_id": os.path.splitext(os.path.basename(f))[0],
            "license": "CC BY-NC-SA 4.0",
            "citation": "Peldszus & Stede (2016), arg-microtexts",
            "text": text,
            "n_words": n,
        })

    with open(args.out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    ns = sorted(i["n_words"] for i in items)
    print(f"\n{len(files)} texts | below {WORD_MIN}: {short} | above {WORD_MAX}: {long_}")
    print(f"written {len(items)} -> {args.out}")
    if ns:
        print(f"  words: min {ns[0]} median {ns[len(ns)//2]} max {ns[-1]}")
    print("\nNote: this is effectively the whole corpus, not a sample. There is "
          "no\nboosted stratum for this genre and none is possible.")


if __name__ == "__main__":
    main()
