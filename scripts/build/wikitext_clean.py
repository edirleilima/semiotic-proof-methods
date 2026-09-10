#!/usr/bin/env python3
"""
Step 1d: cleaning corrected from the template census, plus re-run of the funnel.

Fixes, all census-driven:

  * Scanner validates the template NAME. Previously LaTeX like \\left\\{{x}\\right\\}
    produced a false '{{' opening and the scanner swallowed real proof text
    until it found a closing pair. That is where the garbage census rows came
    from ('x}\\right\\' x6, 'end-eqn}} hence we can express...' x1).
  * {{AimForCont}} (x70) now renders as 'Aiming for a contradiction, suppose'.
    This is the phrase that marks proof by contradiction and it was being
    deleted, which is why ANT looked rare.
  * {{TFAE}}, {{LHS}}, {{RHS}}, {{WRT}}, {{AoC}} rendered.
  * Maintenance names matched with whitespace normalised, so {{proof wanted}}
    (x9) is caught alongside {{ProofWanted}}. {{finish}} (x24) and {{wtd}} added.
  * Items containing {{:Page Name}} transclusions are excluded: the proof body
    lives on another page and is not present in this text.

    python clean_v2.py --path naturalproofs_proofwiki.json

Writes nothing.
"""

import argparse
import json
import re
from collections import Counter

WORD_MIN, WORD_MAX = 50, 300

ACCESSIBLE = [
    "Number Theory", "Prime Numbers", "Divisibility", "Divisors",
    "Real Analysis", "Elementary Number Theory", "Integers", "Rational Numbers",
    "Irrational Numbers", "Natural Numbers", "Combinatorics", "Euclidean Algorithm",
    "Sums of Sequences", "Square Numbers", "Triangular Numbers", "Factorials",
    "Modulo Arithmetic", "Greatest Common Divisor", "Fibonacci Numbers",
    "Geometry", "Triangles", "Set Theory", "Relations", "Functions",
    "Graph Theory", "Logic", "Propositional Logic", "Coprime Integers",
    "Irrationality Proofs", "Cube Numbers", "Square Roots",
]

# --- template handling, from the census -------------------------------------

RENDER = {
    "aimforcont": "Aiming for a contradiction, suppose",
    "iff": "if and only if",
    "wlog": "Without loss of generality",
    "tfae": "the following are equivalent",
    "lhs": "the left hand side",
    "rhs": "the right hand side",
    "wrt": "with respect to",
    "aoc": "the Axiom of Choice",
    "somechoice": "an axiom of choice",
    "modusponens": "Modus Ponens",
    "qed": "",
    "cf": "compare",
}

# Proof is incomplete, disputed, or under repair -> exclude the item.
MAINT_NAMES = {
    "proofwanted", "theoremwanted", "stub", "handwaving", "explain",
    "missinglinks", "improve", "refactor", "questionable", "disputed",
    "wip", "tidy", "proofread", "expand", "rewrite", "delete", "finish",
    "wtd", "help", "dubious", "clarify",
}


def norm_name(name):
    return re.sub(r"[\s_]+", "", name).strip().lower()


# A real template name: starts with a letter or colon, short, no math or links.
NAME_OK = re.compile(r"^[A-Za-z:][A-Za-z0-9 '\-/_.,:()^+]*$")
MAX_NAME = 60


def _name_at(text, i):
    """If a genuine template opens at i (text[i:i+2] == '{{'), return its name."""
    if text.startswith("\\", max(0, i - 1)):        # \{{ from LaTeX \{
        return None
    j, n = i + 2, len(text)
    end = min(n, j + MAX_NAME)
    stop = n
    for k in range(j, end):
        if text[k] == "|" or text.startswith("}}", k):
            stop = k
            break
    else:
        return None
    name = text[j:stop]
    if "$" in name or "[[" in name or "{{" in name:
        return None
    return name if NAME_OK.match(name) else None


def scan_templates(text):
    """Yield (name, inner, start, end) for validated top-level templates."""
    i, n = 0, len(text)
    while i < n:
        if text.startswith("{{", i):
            name = _name_at(text, i)
            if name is None:
                i += 1
                continue
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("{{", j) and _name_at(text, j):
                    depth += 1
                    j += 2
                elif text.startswith("}}", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            yield name, text[i + 2:j - 2], i, j
            i = j
        else:
            i += 1


def render(name, inner):
    key = norm_name(name)
    if key in RENDER:
        return f" {RENDER[key]} "
    if key == "defof":
        parts = inner.split("|")
        return f" definition of {parts[1].strip()} " if len(parts) > 1 else " "
    return " "


FILE_LINK = re.compile(r"\[\[\s*(?:File|Image)\s*:[^\]]*\]\]", re.I)
LINK_PIPED = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
LINK_PLAIN = re.compile(r"\[\[([^\]]*)\]\]")
MATHY = re.compile(r"\$[^$]*\$")


def inspect(raw):
    """Returns (cleaned, reasons_to_exclude)."""
    reasons = []
    spans = list(scan_templates(raw))
    for name, inner, s, e in spans:
        key = norm_name(name)
        if key in MAINT_NAMES:
            reasons.append(f"maint:{key}")
        if name.startswith(":"):
            reasons.append("transclusion")
    if FILE_LINK.search(raw):
        reasons.append("figure")

    out, last = [], 0
    for name, inner, s, e in spans:
        out.append(raw[last:s])
        out.append(render(name, inner))
        last = e
    out.append(raw[last:])
    text = "".join(out)

    text = FILE_LINK.sub(" ", text)
    text = LINK_PIPED.sub(r"\1", text)
    text = LINK_PLAIN.sub(r"\1", text)
    text = re.sub(r"Definition:", "", text)
    parts = re.split(r"(\$[^$]*\$)", text)     # protect LaTeX primes
    text = "".join(p if p.startswith("$") else re.sub(r"'{2,}", "", p)
                   for p in parts)
    text = re.sub(r"={2,}\s*(.+?)\s*={2,}", r"\1.", text)
    text = re.sub(r"^\s*[:*#]+", " ", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip(), sorted(set(reasons))


WORD = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


def prose_stats(text):
    toks = MATHY.sub(" MATH ", text).split()
    if not toks:
        return 0, 0.0
    return len(toks), sum(1 for t in toks if WORD.match(t.strip(".,;:()"))) / len(toks)


# --- cues, corrected --------------------------------------------------------
# 'A similar argument' was removed from PAR: that is case-symmetry, not
# cross-domain analogy. These enrich the boosted sample. They are not labels.

CUES = {
    "ANT": re.compile(
        r"aiming for a contradiction|suppose (?:this is )?not\b"
        r"|assume,? (?:on )?the contrary|for (?:the sake of )?(?:a )?contradiction"
        r"|leads? to a contradiction|contradicts|\ba contradiction\b|reductio"
        r"|which is absurd", re.I),
    "MER": re.compile(
        r"\bcase [1-9i]\b|\b(?:two|three|four|several|both)\s+(?:cases|possibilities)\b"
        r"|(?:consider|split into|divide into|distinguish)\b.{0,30}\bcases\b"
        r"|basis for the induction|induction hypothesis|induction step"
        r"|by (?:finite |strong |complete )?induction|proof by induction"
        r"|the following are equivalent|necessary condition|sufficient condition",
        re.I),
    "PAR": re.compile(
        r"\banalogous|by analogy|analogously|mutatis mutandis"
        r"|is isomorphic to|reduces? to the (?:problem|case) of"
        r"|corresponds? exactly to", re.I),
}


def flatten(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(flatten(i) for i in x)
    if isinstance(x, dict):
        for k in ("contents", "content", "text"):
            if k in x:
                return flatten(x[k])
    return ""


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p / 100))] if s else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="naturalproofs_proofwiki.json")
    ap.add_argument("--all-subjects", action="store_true")
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    ds = json.load(open(args.path, encoding="utf-8")).get("dataset", {})
    theorems = ds.get("theorems", [])

    excl = Counter()
    kept, n_subject = [], 0
    for t in theorems:
        proofs = t.get("proofs") or []
        if not proofs:
            continue
        cats = t.get("categories") or []
        if not args.all_subjects and not any(c in ACCESSIBLE for c in cats):
            continue
        n_subject += 1
        raw = flatten(t.get("contents")) + " " + flatten(proofs[0])
        text, reasons = inspect(raw)
        if reasons:
            for r in reasons:
                excl[r] += 1
            continue
        n, ratio = prose_stats(text)
        if not (WORD_MIN <= n <= WORD_MAX):
            excl["out of band"] += 1
            continue
        kept.append({"title": t.get("title", ""), "categories": cats,
                     "text": text, "n": n, "prose": ratio,
                     "cues": [k for k, rx in CUES.items() if rx.search(text)]})

    print("=" * 68)
    print("FUNNEL")
    print(f"  in accessible subjects       {n_subject:6d}")
    for r, k in excl.most_common():
        print(f"    - {r:26s} {k:6d}")
    print(f"  = kept                       {len(kept):6d}")
    if not kept:
        return
    ns = [i["n"] for i in kept]
    print(f"  words: p25 {pct(ns,25)}  median {pct(ns,50)}  p75 {pct(ns,75)}")

    print("\n" + "=" * 68)
    print(f"CUES  (n={len(kept)})   heuristics for the boosted sample, not labels")
    for k in ("ANT", "MER", "PAR"):
        h = [i for i in kept if k in i["cues"]]
        print(f"    {k}  {len(h):5d}  ({len(h)/len(kept):5.1%})")
    none = [i for i in kept if not i["cues"]]
    multi = [i for i in kept if len(i["cues"]) > 1]
    print(f"    none{len(none):5d}  ({len(none)/len(kept):5.1%})  <- SYN pool")
    print(f"    2+  {len(multi):5d}  ({len(multi)/len(kept):5.1%})  <- combination candidates")

    for k in ("ANT", "PAR", None):
        pool = [i for i in kept if (k in i["cues"] if k else not i["cues"])]
        print("\n" + "=" * 68)
        print(f"SAMPLES -- {k or 'no cue'}   ({len(pool)} available)")
        for i in pool[:args.show]:
            print("\n" + "-" * 68)
            print(f"{i['title']}  [{i['n']} tokens]")
            print(i["text"][:700] + ("..." if len(i["text"]) > 700 else ""))

    print("\n" + "=" * 68)
    print("""Check:

1. ANT should now be materially higher -- {{AimForCont}} appears 70 times in
   this subset and was previously deleted. If it is still ~5%, the render is
   not reaching the cue.
2. Read one ANT sample end to end and confirm no sentence has been swallowed.
   That was the scanner bug and it failed silently.
3. If the kept count is comfortably above ~150, we have enough for both samples
   and this is the last cleaning pass.""")


if __name__ == "__main__":
    main()
