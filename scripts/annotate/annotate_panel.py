#!/usr/bin/env python3
"""
E2: label the full corpus with LLMs.

    export LLM_API_KEY=...
    python llm_annotate.py \
        --base-url https://your-llm-server/v1 \
        --models qwen3.5-122b llama-3.3-70b gpt-oss-120b \
        --runs 3 --out labels.jsonl

The prompt is built from scheme_primary.json at runtime, so the applied text
cannot drift from the definitions of record. Only the per-relation glosses come
from that file; the wrapper instructions and the rationale tail are defined below.

Blind by construction: the model sees item_id and text only. No genre, no
stratum, no cue -- the same blindness the human sheets have.

--runs > 1 samples each item repeatedly at non-zero temperature. The spread is
a result, not overhead: an item a model flips on across runs is telling you
something the modal label hides.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dotenv import load_dotenv

import requests

# Relation glosses are READ FROM a structured scheme file (scheme_primary.json),
# not duplicated here. Duplicating them is how the prompt drifts silently -- an
# earlier draft quietly added "reasoning from precedent" to one category, which
# would have told the models that legal precedent is analogy. That single phrase
# could have manufactured a result.
# definitions/ holds scheme.py and scheme_primary.json; add it to the path so
# this script can be run from the repository root.
_DEFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "definitions")
sys.path.insert(0, _DEFS)
from scheme import load_scheme, build_defs_block


# No mention of the source paper, semiotics, or the master tropes. Burke,
# White and Chandler are all in training data; naming them invites the model to
# retrieve a remembered association instead of reading the passage, which would
# measure recall rather than application of the operational definitions.
INSTRUCTIONS = """Classify how a passage of reasoning establishes its claim.

Each passage argues for some claim. Call the claim S, and call whatever the \
passage rests that claim on S*. Your job is to identify the relationship \
between S and S*. That is, what KIND of move the passage makes to get from \
its support to its conclusion.

The categories:
{defs}

How to decide:

- Judge only what is in the passage. Do not use outside knowledge of the topic, \
and do not reconstruct an argument the text does not actually make.
- Passages may be excerpts, especially from longer documents. Some will begin or \
end mid-argument. Classify the move the excerpt itself makes; do not guess at \
what surrounded it.
- The dominant category is the one the argument depends on: remove that move and \
the passage no longer establishes its claim. There is exactly one.
- Subordinate categories are for moves doing real supporting work. Leave the list \
empty when there are none. Do not add a category as subordinate merely because \
the passage contains some faint trace of it; if that field fires on everything it \
carries no information.
- Choose NONE when the passage states a conclusion without arguing for it, \
recites facts or rules without drawing an inference, or makes a move none of the \
categories fits. NONE is a legitimate answer and you should expect to use it. \
Do not stretch a category to avoid it.
- These categories are not equally common. Do not try to balance your answers \
across a set of passages, and do not assume any category must appear.

Respond with JSON only, no other text:
{{"dominant": "...", "subordinate": [...], "confidence": N}}

confidence: 1 if you are guessing, 2 if plausible, 3 if clear."""

# Extended condition. The model states S and S* before classifying, so the move
# the framework is about becomes inspectable and the label can be checked
# against whether the claim was even identified correctly.
#
# This is a DIFFERENT CONDITION, not an improvement bolted on: reasoning before
# answering shifts the label distribution. Rows record which condition produced
# them so the two never get pooled by accident.
RATIONALE_TAIL = """Work in this order, then respond with JSON only, no other text:

1. claim: the claim the passage is establishing (S), in your own words, one sentence.
2. support: what the passage rests that claim on (S*), one sentence.
3. reasoning: what kind of move connects support to claim, one sentence.
4. dominant, subordinate, confidence, as defined above.

{"claim": "...", "support": "...", "reasoning": "...", "dominant": "...", \
"subordinate": [...], "confidence": N}

confidence: 1 if you are guessing, 2 if plausible, 3 if clear."""


def build_prompt(rng, labels, defs, rationale=False):
    """Assemble the system prompt for one call. The four relations are shuffled
    and NONE pinned last (position bias in option lists is real and would
    otherwise be baked into every label)."""
    defs_block = build_defs_block(rng, labels, defs)
    prompt = INSTRUCTIONS.format(defs=defs_block)
    if rationale:
        head = prompt.split("Respond with JSON only")[0]
        prompt = head + RATIONALE_TAIL
    return prompt


JSON_RE = re.compile(r"\{.*\}", re.S)   # greedy: rationales contain LaTeX braces


def _loads(txt):
    """json.loads, with one recovery pass for LaTeX.

    A rationale quoting a proof writes things like "$\\sqrt p$" into a JSON
    string. A lone backslash is an invalid escape and the whole object fails to
    parse -- which cost 2 of 30 calls in the first maths rationale run, i.e.
    ~7% of a genre.

    On recovery, escape EVERY backslash, not merely the illegal ones. If the
    first parse failed the model was emitting raw LaTeX, so its backslashes are
    all literal -- and several LaTeX commands collide with legal JSON escapes
    (\\frac, \\beta, \\neq, \\times, \\rightarrow). Escaping selectively leaves
    those to be decoded, which silently turns \\frac into a formfeed and 'rac'.
    The label survives that, but the rationale text is quietly corrupted."""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(txt.replace("\\", "\\\\"))
    except json.JSONDecodeError:
        return None


THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
OPEN_THINK_RE = re.compile(r"<think>.*", re.S | re.I)


def parse(content, labels):
    if not content:
        return None
    txt = content.strip()
    # Reasoning models emit <think>...</think> before the answer. Braces inside
    # it would poison the greedy JSON match, so remove it first. An unclosed
    # block means the response was truncated mid-thought: nothing usable.
    txt = THINK_RE.sub(" ", txt)
    if "<think>" in txt.lower():
        return None
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    m = JSON_RE.search(txt)
    if not m:
        return None
    d = _loads(m.group(0))
    if not isinstance(d, dict):
        return None
    dom = str(d.get("dominant", "")).strip().upper()
    if dom not in labels:
        return None
    sub = d.get("subordinate") or []
    if isinstance(sub, str):
        sub = [s.strip() for s in sub.split(";") if s.strip()]
    sub = [s.strip().upper() for s in sub if str(s).strip().upper() in labels]
    try:
        conf = int(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0
    out = {"dominant": dom, "subordinate": sorted(set(sub) - {dom}),
           "confidence": conf}
    for k in ("claim", "support", "reasoning"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = re.sub(r"\s+", " ", v.strip())[:400]
    return out


def call(base_url, key, model, system, user, temperature,
         max_tokens=300, timeout=600, extra=None):
    body = {"model": model, "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if extra:
        body.update(extra)
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json=body, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # vLLM/SGLang with a reasoning parser return the trace in reasoning_content
    # and leave content as the answer. Servers without the parser inline it as
    # <think>...</think>, which parse() strips.
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    if think and not content.strip():
        # The model spent the whole budget thinking and never emitted an
        # answer. Counting that as unparseable would look like a prompt
        # problem; it is a max_tokens problem, so say so.
        raise RuntimeError(
            f"reasoning used the whole budget ({len(think)} chars) with no "
            f"answer; raise --max-tokens above {max_tokens}")
    return content, think


def hms(sec):
    sec = int(max(0, sec))
    h, m = divmod(sec, 3600)
    m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+",
                    default=["corpus/math.jsonl", "corpus/everyday.jsonl",
                             "corpus/legal.jsonl"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--out", default="annotations/labels.jsonl")
    ap.add_argument("--scheme", default="definitions/scheme_primary.json",
                    help="structured scheme file (relation glosses)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default 300, or 800 with --rationale. Raise well "
                         "above that if the model thinks before answering.")
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable a reasoning model's built-in thinking mode "
                         "(Qwen chat_template_kwargs.enable_thinking=false). "
                         "Recommended: --rationale is the reasoning "
                         "manipulation under study, and hidden thinking in one "
                         "panel model but not others confounds cross-model "
                         "agreement.")
    ap.add_argument("--reasoning-effort", default=None,
                    help="pass through for models that support it "
                         "(e.g. low/medium/xhigh). Ignored with --no-thinking.")
    ap.add_argument("--think-cap", type=int, default=0,
                    help="truncate the stored reasoning trace to N characters. "
                         "0 (default) stores it whole: traces run ~5KB, so a "
                         "full pass is tens of MB, and they cannot be "
                         "recovered without re-running.")
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged into the request body, for anything not "
                         "covered above.")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the assembled system and user messages for "
                         "each call. Use with --limit 1 to inspect; it floods "
                         "stdout over a full run.")
    ap.add_argument("--rationale", action="store_true",
                    help="model states S and S* and its reasoning before "
                         "labelling. A different condition, not an upgrade: "
                         "reasoning first shifts the distribution.")
    args = ap.parse_args()
    load_dotenv()

    labels, defs = load_scheme(args.scheme)
    # Scheme-version tag recorded in each row's 'scheme' field. Kept as
    # "codebook.md" (a frozen provenance identifier, NOT a file in this repo) so
    # rows written by a re-run pool with the frozen labels, which carry the same
    # tag. It also feeds the prompt RNG seed below, so changing it would alter
    # category order and break reproduction of the frozen labels. Do not change.
    SCHEME_ID = "codebook.md"

    extra = {}
    if args.top_p is not None:
        extra["top_p"] = args.top_p
    if args.presence_penalty is not None:
        extra["presence_penalty"] = args.presence_penalty
    if args.no_thinking:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    elif args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    if args.extra_body:
        extra.update(json.loads(args.extra_body))
    # A reasoning model spends its budget on the trace before the answer
    # begins, so the default must be far larger when thinking is on.
    if args.max_tokens:
        max_tokens = args.max_tokens
    elif not args.no_thinking:
        max_tokens = 8000
    else:
        max_tokens = 800 if args.rationale else 300

    print(f"scheme {'/'.join(labels)} loaded from {args.scheme} "
          f"(id={SCHEME_ID!r})"
          + ("  [rationale condition]" if args.rationale else "")
          + ("  [thinking disabled]" if args.no_thinking else ""))
    if extra:
        print(f"  extra body: {json.dumps(extra)}")
    print(f"  max_tokens: {max_tokens}")

    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        print("warning: LLM_API_KEY unset", file=sys.stderr)

    items = []
    for p in args.corpora:
        got = load(p)
        if not got:
            print(f"  ! {p} missing or empty", file=sys.stderr)
        items += got
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} items x {len(args.models)} models x {args.runs} runs "
          f"= {len(items)*len(args.models)*args.runs} calls")

    # resume support: long runs should survive a dropped connection
    done = {(r["item_id"], r["model"], r["run"], r.get("scheme"),
             r.get("tropes", True), r.get("rationale", False))
            for r in load(args.out)}
    if done:
        print(f"  resuming, {len(done)} calls already recorded")

    todo = sum(1
               for model in args.models
               for run in range(args.runs)
               for it in items
               if (it["item_id"], model, run, SCHEME_ID,
                   True, args.rationale) not in done)
    print(f"{todo} calls to make")
    if not todo:
        print("nothing to do")

    n_fail = n_done = 0
    think_total = 0
    t0 = time.time()
    last_draw = [0.0]

    def progress(label=""):
        now = time.time()
        if now - last_draw[0] < 0.4 and n_done < todo:
            return
        last_draw[0] = now
        el = now - t0
        rate = n_done / el if el > 0 else 0
        eta = (todo - n_done) / rate if rate > 0 else 0
        pct = n_done / todo if todo else 1
        bar_w = 24
        filled = int(bar_w * pct)
        bar = "#" * filled + "." * (bar_w - filled)
        avg_think = f" think~{think_total // max(1, n_done)}c" if think_total else ""
        sys.stderr.write(
            f"\r[{bar}] {n_done}/{todo} {pct:5.1%} | {rate*60:5.1f}/min"
            f" | {hms(el)} elapsed | eta {hms(eta)}"
            f" | {n_fail} failed{avg_think} {label}   ")
        sys.stderr.flush()

    with open(args.out, "a", encoding="utf-8") as out:
        for model in args.models:
            for run in range(args.runs):
                for i, it in enumerate(items):
                    tag = (it["item_id"], model, run, SCHEME_ID,
                           True, args.rationale)
                    if tag in done:
                        continue
                    # Seed per (item, model, run, condition) rather than
                    # drawing from one shared stream. Two reasons: a shared
                    # stream makes category order depend on how many items were
                    # skipped, so an interrupted-and-resumed run reorders
                    # everything after the gap; and it hands every model the
                    # same order for a given item, so any order bias is shared
                    # across models and inflates apparent agreement.
                    #
                    # The seed string is kept in the pre-refactor form so the
                    # frozen labels reproduce exactly from scratch: the scheme
                    # provenance ("codebook.md") and the original CLI's
                    # no-tropes default (False) occupied these two slots.
                    prompt_rng = random.Random(
                        f"{args.seed}|{it['item_id']}|{model}|{run}"
                        f"|{SCHEME_ID}"
                        f"|{False}|{args.rationale}")
                    system = build_prompt(prompt_rng, labels, defs,
                                          rationale=args.rationale)
                    user = f"Passage:\n\n{it['text']}"
                    if args.show_prompt:
                        print("SYSTEM: ", system)
                        print("USER: ", user)
                    rec = None
                    for attempt in range(3):
                        try:
                            content, think = call(
                                args.base_url, key, model, system, user,
                                args.temperature, max_tokens, extra=extra)
                            rec = parse(content, labels)
                            if rec:
                                if think:
                                    rec["think_chars"] = len(think)
                                    t = re.sub(r"\s+", " ", think)
                                    rec["think"] = (t[:args.think_cap]
                                                    if args.think_cap else t)
                                break
                        except Exception as e:
                            if attempt == 2:
                                sys.stderr.write(
                                    f"\r  ! {it['item_id']} {model}: {e}\n")
                            time.sleep(2 ** attempt)
                    if not rec:
                        n_fail += 1
                        n_done += 1
                        progress()
                        continue
                    out.write(json.dumps({
                        "item_id": it["item_id"], "genre": it.get("genre"),
                        "model": model, "run": run,
                        "tropes": True,
                        "scheme": SCHEME_ID,
                        "rationale": args.rationale,
                        "thinking": not args.no_thinking,
                        # Hash of the text actually sent. Item ids are
                        # positional, so rebuilding a corpus can point the same
                        # id at different text; this makes that detectable
                        # instead of silently joining labels to the wrong
                        # passage.
                        "text_sha1": hashlib.sha1(
                            it["text"].encode("utf-8")).hexdigest()[:12],
                        **rec}, ensure_ascii=False) + "\n")
                    out.flush()
                    n_done += 1
                    think_total += rec.get("think_chars", 0)
                    progress(f"{model.split('/')[-1]} r{run}")

    if todo:
        sys.stderr.write("\n")
        sys.stderr.flush()
        print(f"{n_done} calls in {hms(time.time() - t0)}, {n_fail} failed")

    # ---- summary -----------------------------------------------------------
    # Restrict to rows from this run's scheme/rationale. The output file may
    # accumulate other schemes; pooling them would mix label sets into one
    # meaningless modal count. distribution.py reports properly.
    this = (SCHEME_ID, True, args.rationale)
    rows = [r for r in load(args.out)
            if (r.get("scheme"), r.get("tropes", True),
                r.get("rationale", False)) == this]
    print(f"\n{len(rows)} labels in this condition, {n_fail} unparseable "
          f"this invocation")
    if not rows:
        return
    # models present in the condition, not just the ones passed this time
    seen_models = sorted({r["model"] for r in rows})
    seen_runs = sorted({r["run"] for r in rows})
    if set(seen_models) != set(args.models):
        print(f"  (file also holds: {', '.join(m for m in seen_models if m not in args.models)})")

    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)

    print("\nmodal label by genre (across all models and runs):")
    per = defaultdict(Counter)
    for iid, rs in by_item.items():
        modal = Counter(r["dominant"] for r in rs).most_common(1)[0][0]
        per[rs[0].get("genre", "?")][modal] += 1
    for g in sorted(per):
        tot = sum(per[g].values())
        row = "  ".join(f"{l} {per[g][l]:4d}" for l in labels)
        print(f"  {g:9s} (n={tot:4d})  {row}")

    # self-consistency: same model, same item, across runs
    if len(seen_runs) > 1:
        print("\nself-consistency (same model, repeated runs):")
        for m in seen_models:
            agree = tot = 0
            byi = defaultdict(list)
            for r in rows:
                if r["model"] == m:
                    byi[r["item_id"]].append(r["dominant"])
            for iid, labs in byi.items():
                if len(labs) > 1:
                    tot += 1
                    agree += len(set(labs)) == 1
            if tot:
                print(f"  {m:24s} {agree}/{tot} = {agree/tot:.1%} stable")

    # cross-model agreement on the modal label
    if len(seen_models) > 1:
        print("\ncross-model agreement (modal label per model):")
        unan = tot = 0
        for iid, rs in by_item.items():
            per_m = {}
            for r in rs:
                per_m.setdefault(r["model"], []).append(r["dominant"])
            if len(per_m) < len(seen_models):
                continue
            tot += 1
            modals = {Counter(v).most_common(1)[0][0] for v in per_m.values()}
            unan += len(modals) == 1
        if tot:
            print(f"  all models agree on {unan}/{tot} = {unan/tot:.1%}")
        print("\n  Note: cross-model agreement measures shared bias as readily "
              "as\n  correctness. It is not evidence the taxonomy fits. Only the "
              "human\n  validation set can support that claim.")


if __name__ == "__main__":
    main()
