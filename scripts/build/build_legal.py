#!/usr/bin/env python3
"""
Build the legal genre corpus from the ECHR argument mining corpus.

    python scripts/build/build_legal.py --repo <mining-legal-arguments> --out corpus/legal.jsonl

Replaces the earlier builder, which sampled paragraphs from US published
opinions. That approach sampled a document type rather than an argument type: a
judicial opinion is mostly procedural recitation, evidence summary, citation
apparatus and disposition, so a third of the items were not arguments at all.
Two attempts to filter the non-arguments out failed, and the second failed
informatively: the filters dropped six genuine arguments and none of the
non-arguments, because whether a paragraph does argumentative work is not
recoverable from surface features.

This builder takes the unit from expert annotation instead. Habernal et al.
(2023) annotated 373 European Court of Human Rights decisions with argument
spans and argument types. Each item here is one such span, so every item is an
argument by construction.

Source: https://github.com/trusthlt/mining-legal-arguments  (Apache 2.0)
  Habernal, Faber, Recchia, Bretthauer, Gurevych, Spiecker genannt Doehmann,
  Burchard. "Mining legal arguments in court decisions."
  Artificial Intelligence and Law, 2023.

The expert argument type is carried through as `expert_type`. It is NOT a label
in our scheme and must never be used as one. It exists so the model labels can
be checked against an independent human annotation: if PAR concentrates on
Distinguishing and prior-case-law spans while SYN concentrates on Subsumtion,
that is external evidence the labelling tracks something real.
"""

import argparse
import ast
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

WORD_MIN, WORD_MAX = 50, 300

# Expert types that bear most directly on the analogy relation, and are rare
# enough that a proportional draw leaves too few to check anything.
RARE_TYPES = {
    "Distinguishing", "Overruling", "Rechtsvergleichung",
    "Sinn & Zweck Auslegung", "Systematische Auslegung",
    "Wortlaut Auslegung", "Historische Auslegung",
}

# Habernal's scheme is in German legal terminology. Translations are for the
# reader of the corpus file; nothing downstream branches on them.
TYPE_EN = {
    "Subsumtion": "subsumption (applying a rule to the facts)",
    "Vorherige Rechtsprechung des EGMR": "prior ECtHR case law",
    "Entscheidung des EGMR": "decision of the ECtHR",
    "Distinguishing": "distinguishing a prior case",
    "Overruling": "overruling",
    "Rechtsvergleichung": "comparative law",
    "Sinn & Zweck Auslegung": "teleological interpretation",
    "Wortlaut Auslegung": "literal interpretation",
    "Systematische Auslegung": "systematic interpretation",
    "Historische Auslegung": "historical interpretation",
    "Einschaetzungsspielraum": "margin of appreciation",
    "Einschätzungsspielraum": "margin of appreciation",
    "Konsens der prozessualen Parteien": "consensus of the parties",
    "Verhältnismäßigkeitsprüfung – Rechtsgrundlage": "proportionality: legal basis",
    "Verhältnismäßigkeitsprüfung – Legitimer Zweck": "proportionality: legitimate aim",
    "Verhältnismäßigkeitsprüfung – Geeignetheit": "proportionality: suitability",
    "Verhältnismäßigkeitsprüfung – Erforderlichkeit": "proportionality: necessity",
    "Verhältnismäßigkeitsprüfung – Angemessenheit": "proportionality: stricto sensu",
}


def detokenise(toks):
    """Reassemble annotator tokens into readable prose."""
    s = " ".join(toks)
    s = re.sub(r"\s+([.,;:!?%\u00bb\u201d\u2019)\]])", r"\1", s)
    s = re.sub(r"([\u00ab\u201c\u2018(\[])\s+", r"\1", s)
    s = re.sub(r"\s+('s|n't|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    s = re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", s)
    s = s.replace("\u00a7 \u00a7", "\u00a7\u00a7")
    return re.sub(r"\s+", " ", s).strip()


def read_doc(path):
    """Yield (tokens, labels) per annotated sentence."""
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            try:
                toks = ast.literal_eval(row[0])
                labs = ast.literal_eval(row[1])
            except (ValueError, SyntaxError):
                continue
            if isinstance(toks, list) and isinstance(labs, list):
                yield toks, labs


def sentence_type(labs):
    """Dominant argument type of a sentence, or None if not argumentative."""
    c = Counter(l[2:] for l in labs if l != "O")
    return c.most_common(1)[0][0] if c else None


def units_from_doc(path):
    """Merge consecutive sentences sharing an argument type into one unit."""
    out, cur, toks = [], None, []
    for t, l in read_doc(path):
        typ = sentence_type(l)
        if typ == cur and typ is not None:
            toks += t
        else:
            if cur and toks:
                out.append((cur, toks))
            cur, toks = typ, list(t) if typ else []
    if cur and toks:
        out.append((cur, toks))
    return out


# A span can still be a bare quotation of the Convention text or a stub that
# only introduces one. Both are structural exclusions, not judgements about
# which relation an argument uses.
QUOTE_ONLY = re.compile(r"^\W*[\u201c\"]")
READS_AS_FOLLOWS = re.compile(
    r"reads? as follows\s*:?\s*$|provides? as follows\s*:?\s*$", re.I)


def looks_unusable(text):
    if READS_AS_FOLLOWS.search(text):
        return "statute quotation stub"
    if QUOTE_ONLY.match(text) and text.count("\u201d") + text.count('"') >= 2:
        return "quotation only"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mining-legal-arguments-main",
                    help="unpacked trusthlt/mining-legal-arguments checkout")
    ap.add_argument("--out", default="corpus/legal.jsonl")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--boost-rare", type=int, default=40,
                    help="extra items from rare expert types, flagged "
                         "in_random=false. They enable the validation check "
                         "and are excluded from every distribution estimate.")
    ap.add_argument("--per-doc", type=int, default=3,
                    help="max units from any one decision")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()

    pattern = os.path.join(a.repo, "data", "argType", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no annotated documents at {pattern}\n"
                 f"Download and unpack:\n"
                 f"  https://github.com/trusthlt/mining-legal-arguments")

    rng = random.Random(a.seed)
    all_units, every_unit, n_units, n_band = [], [], 0, 0
    n_unusable = Counter()

    for path in files:
        doc = os.path.splitext(os.path.basename(path))[0]
        keep = []
        for typ, toks in units_from_doc(path):
            n_units += 1
            if not (WORD_MIN <= len(toks) <= WORD_MAX):
                continue
            n_band += 1
            text = detokenise(toks)
            if not (WORD_MIN <= len(text.split()) <= WORD_MAX):
                continue
            why = looks_unusable(text)
            if why:
                n_unusable[why] += 1
                continue
            keep.append({"doc": doc, "expert_type": typ, "text": text})
        every_unit += keep                       # pre-cap, for the rare boost
        if len(keep) > a.per_doc:
            keep = rng.sample(keep, a.per_doc)
        all_units += keep

    rng.shuffle(all_units)
    blind = all_units[:a.target]
    for u in blind:
        u["in_random"] = True

    # Rare expert types carry the validation check. A proportional draw gives
    # only a handful of Distinguishing spans, too few to test whether PAR
    # concentrates on them. Add a flagged boost, kept OUT of in_random so it
    # never touches a distribution estimate -- the same discipline the previous
    # corpus used for its cue strata.
    chosen = {u["text"] for u in blind}
    boost = []
    if a.boost_rare:
        # Draw from every in-band unit, not from the per-doc-capped pool: the
        # cap is a diversity measure for the blind sample and would otherwise
        # discard most of the rare types before they can be boosted.
        rare = [u for u in every_unit
                if u["expert_type"] in RARE_TYPES and u["text"] not in chosen]
        rng.shuffle(rare)
        seen_doc = Counter()
        for u in rare:
            if len(boost) >= a.boost_rare:
                break
            if seen_doc[u["doc"]] >= 2:
                continue
            seen_doc[u["doc"]] += 1
            u = dict(u, in_random=False)
            boost.append(u)

    items = blind + boost

    outdir = os.path.dirname(a.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for n, it in enumerate(items):
            text = it["text"]
            f.write(json.dumps({
                "item_id": f"legal-{n:03d}",
                "genre": "legal",
                "source": "trusthlt/mining-legal-arguments",
                "source_id": it["doc"],
                "license": "Apache-2.0",
                "citation": "Habernal et al. (2023), Mining legal arguments in "
                            "court decisions; ECtHR judgments",
                "text": text,
                "n_words": len(text.split()),
                "in_random": it.get("in_random", True),
                "expert_type": it["expert_type"],
                "expert_type_en": TYPE_EN.get(it["expert_type"], ""),
                "sampling_seed": a.seed,
            }, ensure_ascii=False) + "\n")

    print("=" * 70)
    print(f"  documents                {len(files):6d}")
    print(f"  argument units           {n_units:6d}")
    print(f"  in {WORD_MIN}-{WORD_MAX} words        {n_band:6d}")
    for why, k in n_unusable.most_common():
        print(f"    dropped, {why:24s} {k:6d}")
    print(f"  eligible after per-doc   {len(all_units):6d}")
    print(f"  blind sample             {len(blind):6d}")
    if boost:
        print(f"  rare-type boost          {len(boost):6d}"
              f"   (in_random=false, validation only)")
    print(f"  written                  {len(items):6d} -> {a.out}")

    ns = sorted(len(i["text"].split()) for i in items)
    if ns:
        print(f"  words: min {ns[0]} p25 {ns[len(ns)//4]} med {ns[len(ns)//2]} "
              f"p75 {ns[3*len(ns)//4]} max {ns[-1]}")
    print(f"  distinct decisions       {len({i['doc'] for i in items}):6d}")

    print("\n  expert argument type (independent annotation, not our labels):")
    for t, k in Counter(i["expert_type"] for i in items).most_common():
        en = TYPE_EN.get(t, "")
        print(f"    {k:5d}  {t}" + (f"\n           {en}" if en else ""))

    for typ in ("Distinguishing", "Vorherige Rechtsprechung des EGMR",
                "Subsumtion"):
        pool = [i for i in items if i["expert_type"] == typ]
        if not pool:
            continue
        print("\n" + "-" * 70)
        print(f"SAMPLE -- {typ} ({len(pool)} items)")
        for i in rng.sample(pool, min(a.show, len(pool))):
            print(f"\n  [{i['doc']}]  {len(i['text'].split())} words")
            print("  " + i["text"][:400] + "...")

    print("\n" + "=" * 70)
    print("""expert_type is carried through for VALIDATION ONLY. Never treat it
as a label in the four-relation scheme, and never filter on it: either would
make the comparison circular. Its use is to check, after labelling, whether PAR
concentrates on Distinguishing and prior-case-law spans while SYN concentrates
on Subsumtion. That is the only independent human signal in this study.""")


if __name__ == "__main__":
    main()
