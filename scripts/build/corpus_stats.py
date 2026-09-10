#!/usr/bin/env python3
"""
Corpus validation, statistics, and freeze.

    python corpus_stats.py check
    python corpus_stats.py freeze --version 1.0

`check` validates schema and reports the descriptive statistics the paper needs.
`freeze` additionally writes corpus_manifest.json with SHA-256 per file, so the
dataset you annotate is provably the dataset you describe. Re-run build_*.py
after freezing and the hashes will not match, which is the point.
"""

import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

FILES = {"math": "math.jsonl",
         "everyday": "everyday.jsonl",
         "legal": "legal.jsonl"}
DATA_DIR = "."


def path_for(genre):
    return os.path.join(DATA_DIR, FILES[genre])

# Fields the annotation pipeline actually reads. Anything else is provenance.
REQUIRED = ["item_id", "genre", "text"]
EXPECTED = ["source", "source_id", "license"]


def load(path):
    if not os.path.exists(path):
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! {path}:{n} bad JSON: {e}", file=sys.stderr)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def quart(xs):
    s = sorted(xs)
    if not s:
        return (0, 0, 0, 0, 0)
    q = lambda p: s[min(len(s) - 1, int(len(s) * p))]
    return (s[0], q(.25), q(.50), q(.75), s[-1])


def check(strict=False):
    corpora, problems = {}, []
    for genre, path in FILES.items():
        path = path_for(genre)
        rows = load(path)
        if rows is None:
            problems.append(f"{path} missing -- run the builder for {genre}")
            continue
        corpora[genre] = rows

    if not corpora:
        print("no corpus files found")
        return None, ["no corpus files"]

    print("=" * 70)
    print("COUNTS")
    total = 0
    for genre in FILES:
        if genre not in corpora:
            continue
        n = len(corpora[genre])
        total += n
        print(f"  {genre:10s} {n:5d}   {path_for(genre)}")
    print(f"  {'TOTAL':10s} {total:5d}")

    # ---- schema ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCHEMA")
    for genre, rows in corpora.items():
        keys = Counter(k for r in rows for k in r)
        missing_req = [f for f in REQUIRED if keys.get(f, 0) < len(rows)]
        missing_exp = [f for f in EXPECTED if keys.get(f, 0) < len(rows)]
        partial = sorted(k for k, v in keys.items() if 0 < v < len(rows))
        print(f"\n  {genre}")
        print(f"    fields on every row: {sorted(k for k,v in keys.items() if v==len(rows))}")
        if partial:
            print(f"    on SOME rows only:   {partial}")
        if missing_req:
            problems.append(f"{genre}: missing required field(s) {missing_req}")
            print(f"    ** MISSING REQUIRED: {missing_req}")
        if missing_exp:
            print(f"    (no {missing_exp} -- provenance only, not fatal)")

    # ---- integrity -------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTEGRITY")
    all_ids, all_texts = Counter(), defaultdict(list)
    for genre, rows in corpora.items():
        for r in rows:
            all_ids[r.get("item_id")] += 1
            all_texts[norm(r.get("text"))].append(r.get("item_id"))

    dup_ids = {i: c for i, c in all_ids.items() if c > 1}
    if dup_ids:
        problems.append(f"duplicate item_ids: {list(dup_ids)[:5]}")
        print(f"  ** duplicate item_ids: {len(dup_ids)}")
    else:
        print("  item_ids unique across all genres")

    dup_text = {t: ids for t, ids in all_texts.items() if len(ids) > 1}
    if dup_text:
        print(f"  ** duplicate texts: {len(dup_text)} groups")
        for t, ids in list(dup_text.items())[:3]:
            print(f"       {ids} :: {t[:60]}...")
        problems.append(f"{len(dup_text)} duplicate text groups")
    else:
        print("  no duplicate texts")

    empty = [r.get("item_id") for rows in corpora.values() for r in rows
             if not norm(r.get("text"))]
    if empty:
        problems.append(f"{len(empty)} empty texts")
        print(f"  ** empty texts: {len(empty)}")

    # ---- length ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("LENGTH (words)")
    print(f"  {'genre':10s} {'n':>5s} {'min':>5s} {'p25':>5s} {'med':>5s} "
          f"{'p75':>5s} {'max':>5s} {'mean':>6s}")
    lens = {}
    for genre, rows in corpora.items():
        w = [len((r.get("text") or "").split()) for r in rows]
        lens[genre] = w
        mn, q1, md, q3, mx = quart(w)
        print(f"  {genre:10s} {len(w):5d} {mn:5d} {q1:5d} {md:5d} {q3:5d} "
              f"{mx:5d} {st.mean(w):6.1f}")
        if mn < 50 or mx > 300:
            problems.append(f"{genre}: {sum(1 for x in w if x<50 or x>300)} "
                            f"items outside the 50-300 band")

    if len(lens) > 1:
        meds = {g: quart(w)[2] for g, w in lens.items()}
        hi, lo = max(meds.values()), min(meds.values())
        if hi > 1.6 * lo:
            print(f"\n  NOTE: median length varies {lo}-{hi} across genres. "
                  f"Length is a\n  plausible confound for label distribution; "
                  f"report it, and consider\n  checking whether label varies "
                  f"with length within each genre.")

    # ---- licences --------------------------------------------------------
    print("\n" + "=" * 70)
    print("LICENCE")
    for genre, rows in corpora.items():
        for lic, k in Counter(r.get("license", "?") for r in rows).most_common():
            print(f"  {genre:10s} {k:5d}  {lic}")
    lics = {r.get("license") for rows in corpora.values() for r in rows}
    if any(l and "NC" in str(l) for l in lics):
        print("\n  NOTE: at least one source is non-commercial. Fine for an "
              "arXiv\n  preprint; it does constrain redistribution of a merged "
              "corpus.")

    # ---- per-genre provenance -------------------------------------------
    print("\n" + "=" * 70)
    print("PROVENANCE")
    if "math" in corpora:
        rows = corpora["math"]
        cues = Counter()
        for r in rows:
            for c in (r.get("cues") or ["none"]):
                cues[c] += 1
        cats = Counter(c for r in rows for c in (r.get("categories") or []))
        print(f"\n  math: {len(cats)} distinct ProofWiki categories")
        print(f"    cue tags: {dict(cues)}")
        print(f"    top categories: "
              f"{', '.join(c for c,_ in cats.most_common(6))}")
    if "legal" in corpora:
        rows = corpora["legal"]
        print(f"\n  legal: {len({r.get('source_id') for r in rows})} "
              f"distinct source documents")
        if any("in_random" in r for r in rows):
            n_blind = sum(1 for r in rows if r.get("in_random"))
            print(f"    blind (in_random):  {n_blind}")
            print(f"    boosted:            {len(rows) - n_blind}"
                  f"   (excluded from distribution estimates)")

        # Expert argument-type annotation, present in the ECHR corpus. It is
        # an independent human signal, never a label in our scheme.
        if any(r.get("expert_type") for r in rows):
            blind = [r for r in rows if r.get("in_random", True)]
            print("    expert argument type (blind sample):")
            for t, k in Counter(r.get("expert_type") for r in blind).most_common(8):
                print(f"      {k:5d}  {t}")
            boosted = [r for r in rows if not r.get("in_random", True)]
            if boosted:
                print("    expert argument type (boost):")
                for t, k in Counter(r.get("expert_type")
                                    for r in boosted).most_common(8):
                    print(f"      {k:5d}  {t}")

        # Fields from the earlier US case-law corpus. Absent in the ECHR one.
        strata = Counter(r.get("cue_stratum") for r in rows
                         if r.get("cue_stratum"))
        if strata:
            print(f"    cue_stratum: {dict(strata)}")
        yrs = sorted(str(y) for y in (r.get("year") for r in rows) if y)
        if yrs:
            print(f"    years {yrs[0]}-{yrs[-1]}")
        courts = Counter(c for c in (r.get("court") for r in rows) if c)
        if courts:
            print(f"    {len(courts)} courts; top: "
                  f"{', '.join(c for c, _ in courts.most_common(4))}")
            top_share = courts.most_common(1)[0][1] / len(rows)
            if top_share > 0.25:
                print(f"    NOTE: one court is {top_share:.0%} of legal items")

    if "everyday" in corpora:
        rows = corpora["everyday"]
        print(f"\n  everyday: {len({r.get('source_id') for r in rows})} "
              f"distinct source texts")

    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("no problems found")
    return corpora, problems


def freeze(version):
    corpora, problems = check()
    if corpora is None:
        sys.exit(1)
    if problems:
        print("\nrefusing to freeze with unresolved problems above.")
        print("fix them, or re-run with --force if they are understood and "
              "intentional.")
        if not FORCE:
            sys.exit(1)
        print("(--force given, freezing anyway)")

    man = {"version": version, "files": {}, "counts": {}, "total": 0}
    for genre in FILES:
        path = path_for(genre)
        if not os.path.exists(path):
            continue
        rows = corpora.get(genre, [])
        man["files"][path] = {"sha256": sha256(path), "n_items": len(rows),
                              "bytes": os.path.getsize(path)}
        man["counts"][genre] = len(rows)
        man["total"] += len(rows)

    man_path = os.path.join(DATA_DIR, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print(f"\nwrote {man_path} (version {version}, "
          f"{man['total']} items)")
    for p, d in man["files"].items():
        print(f"  {p:26s} {d['n_items']:5d}  {d['sha256'][:16]}...")
    print("\nCite this version in the paper. Re-running any build_*.py will "
          "change\nthe hashes, which is how you find out the corpus moved "
          "under you.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "freeze"])
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-dir", default="corpus",
                    help="directory holding the corpus jsonl files")
    a = ap.parse_args()
    FORCE = a.force
    globals()['DATA_DIR'] = a.data_dir
    if a.cmd == "check":
        check()
    else:
        freeze(a.version)
