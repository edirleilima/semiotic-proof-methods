#!/usr/bin/env python3
"""
Scheme loading and prompt assembly for the LLM annotation run.

Single condition. Each relation line is

    - CODE (name; axis; trope: X; method: Y): body

with the four relations shuffled per call and NONE pinned last (position bias in
option lists is real and would otherwise be baked into every label).
"""

import json
import os
import sys


def load_scheme(path):
    """Return (labels, defs).

    labels : ordered canonical label list, NONE last
    defs   : {code: {code,name,axis,trope,method,body}}  (NONE: body only)
    """
    if not os.path.exists(path):
        sys.exit(f"{path} not found")
    s = json.load(open(path, encoding="utf-8"))
    order = s["order"]
    defs = {}
    for r in s["relations"]:
        defs[r["code"]] = {k: r.get(k) for k in
                           ("code", "name", "axis", "trope", "method", "body")}
    if "none_body" not in s:
        sys.exit(f"{path} has no none_body; the residual category is required")
    defs["NONE"] = {"code": "NONE", "name": None, "axis": None,
                    "trope": None, "method": None, "body": s["none_body"]}
    if len(order) < 2:
        sys.exit(f"{path} declares only {order}; check the scheme")
    return order + ["NONE"], defs


def relation_line(code, d):
    """One relation line. Full parenthetical for the four relations; NONE (which
    has no name/axis/trope/method) renders as code + body only."""
    if d.get("name"):
        paren = (f"{d['name']}; {d['axis']}; "
                 f"trope: {d['trope']}; method: {d['method']}")
        return f"- {code} ({paren}): {d['body']}"
    return f"- {code}: {d['body']}"


def build_defs_block(rng, labels, defs):
    """The category block: four relations shuffled, NONE pinned last."""
    order = [l for l in labels if l != "NONE"]
    rng.shuffle(order)
    if "NONE" in labels:
        order.append("NONE")
    return "\n".join(relation_line(k, defs[k]) for k in order)
