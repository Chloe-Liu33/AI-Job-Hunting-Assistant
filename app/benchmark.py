"""Prompt-engineering benchmark with an LLM-as-judge.

This is the "prompt engineering benchmark" piece. It treats prompting as an
experiment: several candidate prompt formulations (see PROMPT_VARIANTS in
prompts.py) are run over a small evaluation set, and a *separate* LLM call
scores each output on a fixed rubric (faithfulness / specificity /
actionability / structure, 1-5 each). We then rank the prompts by mean score
and write a reproducible report.

Why this matters: it turns "this prompt feels better" into a measurable,
versioned comparison — exactly what teams do before shipping a prompt change.

Run it:
    python app/benchmark.py            # uses your real CV/JD + a control case
    python app/benchmark.py --quick    # baseline + concise only

Outputs eval/results/benchmark.md and benchmark.csv. Every model call is also
captured by the LLMOps tracer (data/traces/llm_traces.jsonl).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

# Make sibling app modules importable when run as `python app/benchmark.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

import llm  # noqa: E402
import paths  # noqa: E402
import prompts  # noqa: E402
from loaders import load_dir  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"

# A deliberately mismatched control JD: a good analysis MUST score this as a
# poor fit and must NOT fabricate matching experience. It's how we catch
# prompts that flatter the candidate (low faithfulness).
CONTROL_JD = {
    "name": "control_unrelated_sales_role.md",
    "text": (
        "# Regional Sales Director — Industrial Lubricants\n"
        "We need a sales leader to own a $40M territory selling industrial "
        "lubricants to manufacturing plants. Requirements: 10+ years B2B field "
        "sales, proven quota attainment, distributor-channel management, and a "
        "rolodex of plant procurement contacts. No technical/research role."
    ),
}


def _load_cases() -> list[dict]:
    """Build evaluation cases from the user's real CV + JDs, plus the control."""
    cv_docs = load_dir(paths.REPO_DATA / "cv")
    jd_docs = load_dir(paths.REPO_DATA / "jd")
    if not cv_docs:
        raise SystemExit("No CV found in data/cv/. Add one before benchmarking.")
    cv_text = "\n".join(d["text"] for d in cv_docs)
    cases = [{"name": jd["name"], "cv": cv_text, "jd": jd["text"]} for jd in jd_docs]
    cases.append({"name": CONTROL_JD["name"], "cv": cv_text, "jd": CONTROL_JD["text"]})
    return cases


def _parse_judge_json(raw: str) -> dict | None:
    """Robustly pull the rubric JSON out of the judge's reply."""
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


DIMENSIONS = ("faithfulness", "specificity", "actionability", "structure")


def run(variant_names: list[str] | None = None) -> dict:
    cases = _load_cases()
    variants = variant_names or list(prompts.PROMPT_VARIANTS)
    rows = []
    for vname in variants:
        builder = prompts.PROMPT_VARIANTS[vname]
        for case in cases:
            t0 = time.perf_counter()
            output = llm.complete(builder(case["cv"], case["jd"]), system=prompts.SYSTEM)
            gen_ms = (time.perf_counter() - t0) * 1000

            judge_raw = llm.complete(
                prompts.judge_prompt(case["cv"], case["jd"], output),
                system=prompts.JUDGE_SYSTEM,
            )
            scores = _parse_judge_json(judge_raw) or {}
            dim_scores = {d: scores.get(d) for d in DIMENSIONS}
            valid = [v for v in dim_scores.values() if isinstance(v, (int, float))]
            mean = round(sum(valid) / len(valid), 2) if valid else None
            rows.append(
                {
                    "variant": vname,
                    "case": case["name"],
                    "mean": mean,
                    **dim_scores,
                    "gen_ms": round(gen_ms, 0),
                    "comment": scores.get("comment", ""),
                }
            )
            print(f"  {vname:12s} | {case['name'][:32]:32s} | mean={mean}")

    # Aggregate per variant (ignore cases where the judge failed to score).
    agg = {}
    for v in variants:
        means = [r["mean"] for r in rows if r["variant"] == v and r["mean"] is not None]
        agg[v] = round(sum(means) / len(means), 2) if means else None
    ranking = sorted(agg.items(), key=lambda kv: (kv[1] is not None, kv[1] or 0), reverse=True)
    return {"rows": rows, "agg": agg, "ranking": ranking, "variants": variants}


def _write_reports(result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["variant", "case", "mean", *DIMENSIONS, "gen_ms", "comment"]
        )
        w.writeheader()
        w.writerows(result["rows"])

    md = ["# Prompt-engineering benchmark\n", "## Ranking (mean rubric score, 1–5)\n"]
    for i, (v, score) in enumerate(result["ranking"], 1):
        md.append(f"{i}. **{v}** — {score}")
    md.append("\n## Per-case detail\n")
    md.append("| variant | case | mean | " + " | ".join(DIMENSIONS) + " | gen_ms |")
    md.append("|" + "---|" * (4 + len(DIMENSIONS)))
    for r in result["rows"]:
        dims = " | ".join(str(r.get(d, "")) for d in DIMENSIONS)
        md.append(f"| {r['variant']} | {r['case']} | {r['mean']} | {dims} | {r['gen_ms']} |")
    (RESULTS_DIR / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {RESULTS_DIR / 'benchmark.md'}")


if __name__ == "__main__":
    provider, status = llm.provider_status()
    if not provider:
        raise SystemExit(status)
    print(f"Benchmarking with {status}\n")
    names = ["baseline", "concise"] if "--quick" in sys.argv else None
    res = run(names)
    print("\nRanking:")
    for i, (v, s) in enumerate(res["ranking"], 1):
        print(f"  {i}. {v}: {s}")
    _write_reports(res)
