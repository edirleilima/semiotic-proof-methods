#!/usr/bin/env python3
"""
Analyse labels.jsonl.

    python scripts/analyse/distribution.py --labels annotations/labels.jsonl --data-dir corpus
    python scripts/analyse/distribution.py --labels annotations/labels.jsonl --data-dir corpus --csv results/

Rows are keyed by (item_id, model, run, scheme, tropes, rationale). Conditions
are never pooled: each section states which slice it describes.

Headline measures, all computed per genre on the primary condition:
  residue           fraction labelled NONE          -- exhaustiveness
  combination rate  fraction with subordinates      -- purity
  distribution      how the categories split        -- scope
  agreement         cross-model, and self-consistency across runs

Reported with bootstrap intervals over items. The interval covers sampling of
items only; it says nothing about model error, which needs human validation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

# Join key, not a filename. Every row in labels.jsonl carries a frozen
# provenance tag scheme="codebook.md" (written as SCHEME_ID in
# scripts/annotate/annotate_panel.py); this value selects those rows and must
# match the tag verbatim. It therefore stays "codebook.md" even though the paper
# and README refer to these as the operational definitions. See the "Provenance
# tag" note in the README.
PRIMARY_SCHEME = "codebook.md"


# ---------------------------------------------------------------- loading

def load_jsonl(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found")
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ! {path}:{n} bad JSON, skipped", file=sys.stderr)
    return out


def load_corpus(data_dir):
    items = {}
    for name in ("math.jsonl", "everyday.jsonl",
                 "legal.jsonl"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            for r in load_jsonl(p):
                items[r["item_id"]] = r
    return items


def cond(r):
    return (r.get("scheme", "?"), bool(r.get("tropes", True)),
            bool(r.get("rationale", False)))


def cond_name(c):
    scheme, tropes, rat = c
    bits = [scheme]
    if rat:
        bits.append("rationale")
    return " / ".join(bits)


# ---------------------------------------------------------------- stats

def boot_ci(values, stat, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over a list of per-item values."""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    point = stat(values)
    k = len(values)
    reps = []
    for _ in range(n):
        reps.append(stat([values[rng.randrange(k)] for _ in range(k)]))
    reps.sort()
    lo = reps[int(alpha / 2 * n)]
    hi = reps[int((1 - alpha / 2) * n)]
    return point, lo, hi


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def krippendorff_nominal(units):
    """Krippendorff's alpha for nominal data.

    units: list of lists of labels, one inner list per item (raters may differ
    in number across items, which Cohen's kappa cannot handle). Used here for
    the model panel, where each item has several models x runs."""
    units = [u for u in units if len(u) > 1]
    if not units:
        return float("nan")
    cats = sorted({c for u in units for c in u})
    idx = {c: i for i, c in enumerate(cats)}
    n_total = sum(len(u) for u in units)

    # observed disagreement
    Do = 0.0
    for u in units:
        m = len(u)
        cnt = Counter(u)
        pairs = m * (m - 1)
        same = sum(v * (v - 1) for v in cnt.values())
        Do += (pairs - same) / (m - 1)
    Do /= n_total

    # expected disagreement
    marg = Counter(c for u in units for c in u)
    De = 0.0
    for a in cats:
        for b in cats:
            if a != b:
                De += marg[a] * marg[b]
    De /= (n_total * (n_total - 1))
    return 1 - Do / De if De else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ---------------------------------------------------------------- reporting

def modal_labels(rows):
    """item_id -> (modal dominant, n votes, any_subordinate, stability).
    Ties broken deterministically by label name, and flagged."""
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    out, ties = {}, 0
    for iid, rs in by_item.items():
        cnt = Counter(r["dominant"] for r in rs)
        top = cnt.most_common()
        best = top[0][1]
        winners = sorted(l for l, c in top if c == best)
        if len(winners) > 1:
            ties += 1
        modal = winners[0]
        # subordinate: present in a majority of votes, not merely once
        sub_votes = sum(1 for r in rs if r.get("subordinate"))
        out[iid] = {
            "dominant": modal,
            "n_votes": len(rs),
            "agreement": best / len(rs),
            "subordinate_majority": sub_votes > len(rs) / 2,
            "subordinate_any": sub_votes > 0,
            "tie": len(winners) > 1,
        }
    return out, ties


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def is_blind(iid, corpus):
    """True if the item is eligible for an unbiased distribution estimate.

    The legal corpus contains cue-selected items alongside a blind reservoir.
    The cue pools exist to enrich thin cells for reliability work; counting
    them in a distribution inflates whatever the keywords selected for. Math
    and everyday have no cue selection, so every item qualifies."""
    it = corpus.get(iid, {})
    if it.get("genre") != "legal":
        return True
    return bool(it.get("in_random"))


def report_distribution(modal, corpus, labels, tag, rows_out=None,
                        blind_only=True):
    by_genre = defaultdict(list)
    for iid, m in modal.items():
        if blind_only and not is_blind(iid, corpus):
            continue
        g = corpus.get(iid, {}).get("genre", "?")
        by_genre[g].append(m["dominant"])

    hdr = "  " + f"{'genre':10s} {'n':>5s} " + " ".join(f"{l:>7s}" for l in labels)
    print(hdr)
    for g in sorted(by_genre):
        labs = by_genre[g]
        c = Counter(labs)
        cells = " ".join(f"{c[l]/len(labs):6.1%} " for l in labels)
        print(f"  {g:10s} {len(labs):5d} {cells}")
        if rows_out is not None:
            for l in labels:
                lo, hi = wilson(c[l], len(labs))
                rows_out.append({"genre": g, "label": l,
                                 "n": len(labs), "count": c[l],
                                 "prop": c[l] / len(labs),
                                 "ci_lo": lo, "ci_hi": hi})
    print("\n  counts")
    for g in sorted(by_genre):
        c = Counter(by_genre[g])
        print(f"  {g:10s} " + " ".join(f"{l}={c[l]:<5d}" for l in labels))


def report_headline(modal, corpus, tag, rows_out=None, blind_only=True):
    by_genre = defaultdict(list)
    for iid, m in modal.items():
        if blind_only and not is_blind(iid, corpus):
            continue
        by_genre[corpus.get(iid, {}).get("genre", "?")].append(m)
    print(f"  {'genre':10s} {'n':>5s} {'residue':>22s} {'combination':>22s}")
    for g in sorted(by_genre) + (["ALL"] if len(by_genre) > 1 else []):
        ms = [m for mm in by_genre.values() for m in mm] if g == "ALL" else by_genre[g]
        res = [1.0 if m["dominant"] == "NONE" else 0.0 for m in ms]
        com = [1.0 if m["subordinate_majority"] else 0.0 for m in ms]
        r, rlo, rhi = boot_ci(res, mean)
        c, clo, chi = boot_ci(com, mean)
        print(f"  {g:10s} {len(ms):5d} "
              f"{r:7.1%} [{rlo:5.1%},{rhi:5.1%}] "
              f"{c:9.1%} [{clo:5.1%},{chi:5.1%}]")
        if rows_out is not None:
            rows_out.append({"genre": g, "n": len(ms),
                             "residue": r, "residue_lo": rlo, "residue_hi": rhi,
                             "combination": c, "combination_lo": clo,
                             "combination_hi": chi})


def report_reliability(rows, corpus):
    models = sorted({r["model"] for r in rows})
    runs = sorted({r["run"] for r in rows})

    if len(runs) > 1:
        print("  self-consistency (same model, same item, across runs)")
        for m in models:
            byi = defaultdict(list)
            for r in rows:
                if r["model"] == m:
                    byi[r["item_id"]].append(r["dominant"])
            multi = [v for v in byi.values() if len(v) > 1]
            if multi:
                stable = sum(1 for v in multi if len(set(v)) == 1)
                lo, hi = wilson(stable, len(multi))
                print(f"    {m:26s} {stable}/{len(multi)} = "
                      f"{stable/len(multi):5.1%} [{lo:.1%},{hi:.1%}]")
    else:
        print("  self-consistency: n/a (single run)")

    if len(models) > 1:
        print("\n  cross-model agreement on the per-model modal label")
        byi = defaultdict(lambda: defaultdict(list))
        for r in rows:
            byi[r["item_id"]][r["model"]].append(r["dominant"])
        full = [v for v in byi.values() if len(v) == len(models)]
        if full:
            unan = 0
            for v in full:
                mods = {Counter(x).most_common(1)[0][0] for x in v.values()}
                unan += len(mods) == 1
            lo, hi = wilson(unan, len(full))
            print(f"    unanimous on {unan}/{len(full)} = {unan/len(full):5.1%} "
                  f"[{lo:.1%},{hi:.1%}]")

        print("\n  Krippendorff alpha (all model x run judgements per item)")
        units = defaultdict(list)
        for r in rows:
            units[r["item_id"]].append(r["dominant"])
        print(f"    overall  alpha = {krippendorff_nominal(list(units.values())):.3f}")
        per = defaultdict(list)
        for iid, labs in units.items():
            per[corpus.get(iid, {}).get("genre", "?")].append(labs)
        for g in sorted(per):
            print(f"    {g:9s} alpha = {krippendorff_nominal(per[g]):.3f}")
        print("\n    Agreement among models measures shared bias as readily as")
        print("    correctness. It bounds how consistently the categories can be")
        print("    applied; it is not evidence that they fit.")

        print("\n  pairwise disagreement, most frequent confusions")
        pair = Counter()
        for iid, mm in byi.items():
            mods = sorted((m, Counter(v).most_common(1)[0][0]) for m, v in mm.items())
            for i in range(len(mods)):
                for j in range(i + 1, len(mods)):
                    a, b = mods[i][1], mods[j][1]
                    if a != b:
                        pair[tuple(sorted((a, b)))] += 1
        for (a, b), k in pair.most_common(8):
            print(f"    {a:5s} vs {b:5s}  {k}")


def report_rationale_quality(rows, corpus, n_show):
    withr = [r for r in rows if r.get("claim")]
    if not withr:
        return
    print(f"  {len(withr)} rows carry claim/support/reasoning")
    lens = [len(r["claim"].split()) for r in withr]
    print(f"  claim length: median {sorted(lens)[len(lens)//2]} words")
    keys = ("concession", "contrast", "however", "precedent", "analog",
            "induction", "contradiction", "cases", "warrant", "implicit")
    hits = Counter()
    for r in withr:
        blob = (r.get("reasoning", "") + " " + r.get("support", "")).lower()
        for k in keys:
            if k in blob:
                hits[k] += 1
    print("  reasoning mentions: " +
          ", ".join(f"{k}={v}" for k, v in hits.most_common()))
    print("\n  Grep the reasoning field when a distribution looks wrong -- that "
          "is how\n  the concession/ANT conflation was found. Self-reports are "
          "hypotheses about\n  failure modes, not accounts of what produced the "
          "label.")
    if n_show:
        print(f"\n  {n_show} examples:")
        for r in withr[:n_show]:
            print(f"\n    [{r['item_id']} -> {r['dominant']}]")
            print(f"      claim   : {r.get('claim','')[:150]}")
            print(f"      support : {r.get('support','')[:150]}")
            print(f"      reason  : {r.get('reasoning','')[:150]}")


def write_csv(out_dir, name, rows):
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="annotations/labels.jsonl")
    ap.add_argument("--data-dir", default="corpus")
    ap.add_argument("--csv", help="directory for machine-readable output")
    ap.add_argument("--show-rationales", type=int, default=0)
    a = ap.parse_args()

    rows = load_jsonl(a.labels)
    corpus = load_corpus(a.data_dir)
    if not rows:
        sys.exit("no labels")
    if not corpus:
        print("! no corpus files found; genre breakdowns will be empty",
              file=sys.stderr)

    conds = Counter(cond(r) for r in rows)
    section("CONDITIONS PRESENT")
    print(f"  {len(rows)} label rows over {len({r['item_id'] for r in rows})} items")
    for c, k in conds.most_common():
        rs = [r for r in rows if cond(r) == c]
        print(f"  {k:7d} rows  {len({r['item_id'] for r in rs}):5d} items  "
              f"{len({r['model'] for r in rs})} models x "
              f"{len({r['run'] for r in rs})} runs   {cond_name(c)}")

    missing = {r["item_id"] for r in rows} - set(corpus)
    if missing:
        print(f"\n  ! {len(missing)} labelled items not in the corpus "
              f"(stale labels?): {sorted(missing)[:3]}")

    # Item ids are positional, so a rebuilt corpus can reuse an id for
    # different text. Labels carrying text_sha1 let that be caught rather than
    # silently producing labels attached to the wrong passage.
    stale = set()
    for r in rows:
        h = r.get("text_sha1")
        if not h:
            continue
        it = corpus.get(r["item_id"])
        if it and hashlib.sha1(
                it["text"].encode("utf-8")).hexdigest()[:12] != h:
            stale.add(r["item_id"])
    if stale:
        print(f"\n  !! {len(stale)} items have labels whose recorded text does")
        print(f"     NOT match the current corpus. The corpus was rebuilt and")
        print(f"     these ids now point at different passages. Delete those")
        print(f"     label rows before trusting anything below.")
        print(f"     e.g. {sorted(stale)[:5]}")
    n_hashed = sum(1 for r in rows if r.get("text_sha1"))
    if rows and not n_hashed:
        print("\n  (labels predate text_sha1; rebuild mismatches cannot be "
              "checked)")

    # primary condition: the scheme under test, tropes on, most rows
    prim = [c for c in conds if c[0] == PRIMARY_SCHEME and c[1]]
    if not prim:
        sys.exit(f"no rows for scheme {PRIMARY_SCHEME} with tropes on")
    primary = max(prim, key=lambda c: conds[c])
    prows = [r for r in rows if cond(r) == primary]
    labels = sorted({r["dominant"] for r in prows},
                    key=lambda l: (l == "NONE", l))

    modal, ties = modal_labels(prows)

    section(f"PRIMARY CONDITION -- {cond_name(primary)}")
    print(f"  {len(prows)} rows, {len(modal)} items, "
          f"{len(prows)/max(1,len(modal)):.1f} votes/item")
    if ties:
        print(f"  ! {ties} items had a tied modal label "
              f"(broken alphabetically -- treat as unresolved)")

    n_cue = sum(1 for iid in modal if not is_blind(iid, corpus))

    dist_rows, head_rows = [], []
    section("DISTRIBUTION BY GENRE (unbiased: blind items only)")
    if n_cue:
        print(f"  excluding {n_cue} cue-selected legal items. They were chosen "
              f"by keyword to\n  enrich thin cells; counting them would inflate "
              f"whatever the keywords sought.\n")
    report_distribution(modal, corpus, labels, cond_name(primary), dist_rows)

    section("HEADLINE MEASURES (blind items only, bootstrap 95% CI)")
    report_headline(modal, corpus, cond_name(primary), head_rows)

    if n_cue:
        section("FOR REFERENCE ONLY: full pool including cue-selected items")
        print("  Not an estimate of anything. Shown so the effect of the cue\n"
              "  strata is visible rather than hidden.\n")
        report_distribution(modal, corpus, labels,
                            cond_name(primary) + " [full pool]",
                            None, blind_only=False)
    combo_any = sum(1 for m in modal.values() if m["subordinate_any"])
    print(f"\n  subordinate present in ANY vote: {combo_any}/{len(modal)} = "
          f"{combo_any/max(1,len(modal)):.1%}")
    print("  (combination rate above uses a majority of votes; if the two differ")
    print("   greatly the field is unstable and should be reported as such)")

    section("RELIABILITY")
    report_reliability(prows, corpus)

    if any(r.get("claim") for r in prows):
        section("RATIONALE FIELDS")
        report_rationale_quality(prows, corpus, a.show_rationales)

    if a.csv:
        section("CSV OUTPUT")
        write_csv(a.csv, "distribution.csv", dist_rows)
        write_csv(a.csv, "headline.csv", head_rows)
        per_item = [{"item_id": i,
                     "genre": corpus.get(i, {}).get("genre", "?"),
                     "blind": int(is_blind(i, corpus)),
                     "cue_stratum": corpus.get(i, {}).get("cue_stratum") or "",
                     "dominant": m["dominant"], "n_votes": m["n_votes"],
                     "agreement": round(m["agreement"], 3),
                     "subordinate_majority": int(m["subordinate_majority"]),
                     "tie": int(m["tie"])}
                    for i, m in sorted(modal.items())]
        write_csv(a.csv, "per_item.csv", per_item)

    section("REMINDERS")
    print("""  - Intervals cover item sampling only. They do not cover model
    error, which needs human validation to estimate.
  - No human labels are involved anywhere above.
  - Report the operational-definitions version (v1.3) and that it was revised after piloting.""")


if __name__ == "__main__":
    main()
