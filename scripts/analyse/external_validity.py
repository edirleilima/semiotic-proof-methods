#!/usr/bin/env python3
"""
Habernal external check: SYN and PAR, exactly as pre-registered (see the paper's
external-validity subsection).

Joins model labels to the legal corpus on item_id, takes the per-item modal
dominant (same tie rule as distribution.py), groups by expert_type, and computes:

  PRIMARY   P(PAR | Distinguishing + comparative law) vs P(PAR | prior case law)
            -- one-sided Fisher exact + Newcombe 95% CI on the rate difference.
            Both classes are citation-dense, so a separation cannot come from
            matching on citations.
  SECONDARY P(SYN | Subsumtion), Wilson CI, vs P(SYN | Distinguishing). Weak:
            SYN is the corpus default.

Reports cell sizes, the blind/boosted composition of the PAR cell (boosting is
legitimate here -- within-type conditional rate, not a prevalence estimate), and
the vote-agreement of PAR-cell items (a specificity result resting on 5-4
majorities is worth flagging).

    python scripts/analyse/external_validity.py --labels annotations/labels.jsonl \
        --legal corpus/legal.jsonl
"""
import argparse, json, math, sys
from collections import Counter, defaultdict

# expert_type groupings (exact strings from the legal corpus)
PAR_TYPES  = {"Distinguishing", "Rechtsvergleichung"}
BASELINE   = {"Vorherige Rechtsprechung des EGMR"}
SYN_TYPE   = {"Subsumtion"}


def load_jsonl(p):
    if not p or not __import__("os").path.exists(p):
        sys.exit(f"{p} not found")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def newcombe_diff(x1, n1, x2, n2, z=1.96):
    """95% CI for p1 - p2 (Newcombe method 10, score-interval based)."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return p1 - p2, lo, hi


def fisher_greater(a, b, c, d):
    """One-sided Fisher exact: P(PAR-cell PAR count >= observed) given margins.
    Table [[a,b],[c,d]] = [[PAR-cell PAR, non-PAR],[baseline PAR, non-PAR]]."""
    r1, r2, c1 = a + b, c + d, a + c
    N = r1 + r2
    comb = math.comb
    def pmf(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(N, c1)
    hi = min(r1, c1)
    return sum(pmf(x) for x in range(a, hi + 1))


def as_types(v):
    if isinstance(v, list):
        return {str(x) for x in v}
    return {str(v)} if v is not None else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--legal", required=True)
    a = ap.parse_args()

    legal = {r["item_id"]: r for r in load_jsonl(a.legal)}
    rows = load_jsonl(a.labels)

    # guard: this analysis is for a single condition. Warn if the file mixes them.
    conds = {(r.get("scheme", "codebook.md"), r.get("tropes", True),
              r.get("rationale", False)) for r in rows}
    if len(conds) > 1:
        print(f"! labels.jsonl holds >1 condition {sorted(conds)}; keeping the "
              f"primary (tropes on). Ablation/other rows are excluded.")
    rows = [r for r in rows if r.get("tropes", True)]

    # per-item modal dominant over all votes (alphabetical tie-break, flagged)
    votes = defaultdict(list)
    for r in rows:
        if r["item_id"] in legal:
            votes[r["item_id"]].append(r["dominant"])
    modal, ties = {}, 0
    agree = {}
    for iid, ds in votes.items():
        cnt = Counter(ds)
        best = max(cnt.values())
        winners = sorted(l for l, c in cnt.items() if c == best)
        ties += len(winners) > 1
        modal[iid] = winners[0]
        agree[iid] = best / len(ds)          # modal share of votes
    print(f"legal items with labels: {len(modal)}  "
          f"(votes/item ~{sum(len(v) for v in votes.values())/max(1,len(votes)):.1f})"
          f"{'  ! '+str(ties)+' modal ties' if ties else ''}")

    # attach expert_type
    def group(iid):
        return as_types(legal[iid].get("expert_type"))

    # ---- descriptive crosstab expert_type x dominant --------------------
    cross = defaultdict(Counter)
    for iid, dom in modal.items():
        for t in group(iid):
            cross[t][dom] += 1
    labels_order = ["SYN", "PAR", "ANT", "MER", "NONE"]
    print("\nexpert_type x modal relation (descriptive, all types)")
    print(f"  {'expert_type':38s} {'n':>4s} " +
          " ".join(f"{l:>5s}" for l in labels_order))
    for t in sorted(cross, key=lambda k: -sum(cross[k].values())):
        c = cross[t]
        n = sum(c.values())
        print(f"  {t:38s} {n:4d} " +
              " ".join(f"{c[l]:5d}" for l in labels_order))

    # ---- PRIMARY TEST ---------------------------------------------------
    par_items = [i for i in modal if group(i) & PAR_TYPES]
    base_items = [i for i in modal if group(i) & BASELINE]
    a_ = sum(modal[i] == "PAR" for i in par_items); b_ = len(par_items) - a_
    c_ = sum(modal[i] == "PAR" for i in base_items); d_ = len(base_items) - c_

    print("\n" + "=" * 68)
    print("PRIMARY: PAR specificity (correspondence-drawing vs authority-citation)")
    print("=" * 68)
    if not par_items or not base_items:
        print("  ! a cell is empty; cannot run the contrast")
    else:
        pl, pu = wilson(a_, len(par_items))
        bl, bu = wilson(c_, len(base_items))
        diff, dlo, dhi = newcombe_diff(a_, len(par_items), c_, len(base_items))
        p = fisher_greater(a_, b_, c_, d_)
        print(f"  PAR cell (Distinguishing + comparative law): "
              f"{a_}/{len(par_items)} PAR = {a_/len(par_items):.1%} "
              f"[{pl:.1%},{pu:.1%}]")
        print(f"  baseline (prior case law):                   "
              f"{c_}/{len(base_items)} PAR = {c_/len(base_items):.1%} "
              f"[{bl:.1%},{bu:.1%}]")
        print(f"  difference: {diff:+.1%}  Newcombe 95% CI [{dlo:+.1%},{dhi:+.1%}]")
        print(f"  one-sided Fisher exact p = {p:.3g}")
        # boost composition of the PAR cell
        comp = Counter()
        for i in par_items:
            comp[bool(legal[i].get("in_random"))] += 1
        print(f"  PAR-cell composition: blind={comp[True]}, boosted={comp[False]} "
              f"(boosting legitimate: within-type conditional rate)")
        # per-type breakdown inside the PAR cell
        for t in sorted(PAR_TYPES):
            its = [i for i in par_items if t in group(i)]
            if its:
                k = sum(modal[i] == "PAR" for i in its)
                print(f"    {t:20s} {k}/{len(its)} PAR")
        # vote robustness of PAR-cell items
        unan = sum(1 for i in par_items if agree[i] == 1.0)
        print(f"  PAR-cell vote agreement: {unan}/{len(par_items)} unanimous; "
              f"modal-share min = {min(agree[i] for i in par_items):.0%}")

        # --- Distinguishing-only (the unambiguous correspondence type) -----
        dis = [i for i in modal if "Distinguishing" in group(i)]
        ad = sum(modal[i] == "PAR" for i in dis); bd = len(dis) - ad
        dl, du = wilson(ad, len(dis))
        pd = fisher_greater(ad, bd, c_, d_)
        print(f"\n  [Distinguishing only] {ad}/{len(dis)} PAR = "
              f"{ad/len(dis):.1%} [{dl:.1%},{du:.1%}] vs baseline "
              f"{c_/len(base_items):.1%}; Fisher p = {pd:.3g}")

        # --- vote-level robustness (descriptive: 9 votes/item, not independent)
        def vote_rate(items):
            pv = sum(1 for i in items for d in votes[i] if d == "PAR")
            tv = sum(len(votes[i]) for i in items)
            return pv, tv
        pv1, tv1 = vote_rate(par_items)
        pv0, tv0 = vote_rate(base_items)
        print(f"  [vote-level, descriptive] PAR cell {pv1}/{tv1} votes = "
              f"{pv1/tv1:.1%}; baseline {pv0}/{tv0} = {pv0/tv0:.1%} "
              f"(votes within an item are correlated; no p-value)")

        # --- items to inspect: fragile PAR-cell labels + comparative-law null
        fragile = sorted(i for i in par_items if agree[i] < 0.6)
        print(f"  inspect (modal-share <60%): {fragile}")
        cl = sorted(i for i in modal if "Rechtsvergleichung" in group(i))
        print(f"  comparative-law items (0/5 PAR): {cl}")

    # ---- SECONDARY TEST -------------------------------------------------
    sub_items = [i for i in modal if group(i) & SYN_TYPE]
    dis_items = [i for i in modal if "Distinguishing" in group(i)]
    print("\n" + "=" * 68)
    print("SECONDARY: SYN sensitivity (weak -- SYN is the default)")
    print("=" * 68)
    if sub_items:
        k = sum(modal[i] == "SYN" for i in sub_items)
        l, u = wilson(k, len(sub_items))
        print(f"  P(SYN | Subsumtion)     = {k}/{len(sub_items)} = "
              f"{k/len(sub_items):.1%} [{l:.1%},{u:.1%}]")
    if dis_items:
        k = sum(modal[i] == "SYN" for i in dis_items)
        print(f"  P(SYN | Distinguishing) = {k}/{len(dis_items)} = "
              f"{k/len(dis_items):.1%}   (should be lower)")


if __name__ == "__main__":
    main()
