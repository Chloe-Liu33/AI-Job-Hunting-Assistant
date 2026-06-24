"""Prompt templates for the job-hunting agent."""

SYSTEM = (
    "You are a sharp, honest job-application coach. You analyze how well a candidate's "
    "CV fits a specific job description, and you give concrete, actionable advice. "
    "Be specific and grounded only in the provided CV and JD — never invent experience "
    "the candidate doesn't have. If the candidate writes in Chinese, you may answer in "
    "Chinese; otherwise answer in English. Be concise and well-structured."
)


def fit_analysis_prompt(cv_text: str, jd_text: str) -> str:
    return f"""Analyze the fit between this candidate and this job.

=== CANDIDATE CV ===
{cv_text}

=== JOB DESCRIPTION ===
{jd_text}

Produce a structured analysis with these sections:

**Fit score**: X/10 with one sentence of reasoning.

**Strengths**: 3–5 bullet points where the CV genuinely matches what the JD asks for. Quote the matching requirement briefly.

**Gaps & risks**: 2–4 bullet points where the candidate is weak or missing something the JD wants. Be honest.

**Keywords to add**: a short list of exact terms/skills from the JD that the candidate should surface in their CV (only ones they can credibly claim).

**Next move**: one or two sentences on whether/how to apply.
"""


def fit_analysis_rag_prompt(cv_context: str, jd_text: str) -> str:
    """RAG-grounded variant: the CV is supplied as *retrieved* passages with
    [n] source markers instead of the full document. The model must ground its
    claims in those passages and cite them, which is the whole point of RAG —
    less context, more relevance, traceable evidence."""
    return f"""Analyze the fit between this candidate and this job.

You are given the candidate's CV as RETRIEVED PASSAGES (each tagged with a [n]
marker). Treat these passages as your only evidence about the candidate. When
you assert a strength, cite the passage it comes from, e.g. "(see [2])". Do not
claim experience that is not supported by a passage.

=== RETRIEVED CV PASSAGES ===
{cv_context}

=== JOB DESCRIPTION ===
{jd_text}

Produce a structured analysis with these sections:

**Fit score**: X/10 with one sentence of reasoning.

**Strengths**: 3–5 bullets where the passages genuinely match the JD. Cite [n].

**Gaps & risks**: 2–4 bullets where evidence is missing or weak. Be honest.

**Keywords to add**: exact terms/skills from the JD the candidate can credibly claim.

**Next move**: one or two sentences on whether/how to apply.
"""


# ---- Prompt-engineering benchmark variants ----------------------------------
# These are the candidate prompts the benchmark (app/benchmark.py) compares
# head-to-head with an LLM-as-judge. Each takes (cv_text, jd_text).
def _variant_baseline(cv_text: str, jd_text: str) -> str:
    return fit_analysis_prompt(cv_text, jd_text)


def _variant_concise(cv_text: str, jd_text: str) -> str:
    return f"""You are a recruiter. In <=120 words, rate this candidate's fit for the
job from 1-10 and give the single biggest strength and single biggest gap.

CV:
{cv_text}

JOB:
{jd_text}
"""


def _variant_persona_cot(cv_text: str, jd_text: str) -> str:
    return f"""You are a senior hiring manager who has screened 10,000 CVs.
Think step by step BEFORE answering: (1) extract the JD's top-5 must-haves,
(2) check the CV for evidence of each, (3) only then write the verdict.

Output exactly these sections: Fit score (X/10), Strengths (with the JD
requirement each maps to), Gaps & risks, Keywords to add, Next move.
Ground every claim in the CV; never invent experience.

CV:
{cv_text}

JOB:
{jd_text}
"""


# Registry consumed by the benchmark. Add a variant here to enter it in the race.
PROMPT_VARIANTS = {
    "baseline": _variant_baseline,
    "concise": _variant_concise,
    "persona_cot": _variant_persona_cot,
}


JUDGE_SYSTEM = (
    "You are a strict evaluation judge for job-fit analyses. You score outputs "
    "on a rubric and respond with ONLY a JSON object, no prose."
)


def judge_prompt(cv_text: str, jd_text: str, candidate_output: str) -> str:
    """LLM-as-judge: score one candidate output on a 1-5 rubric, return JSON."""
    return f"""Score the CANDIDATE ANALYSIS below on how well it judges this
candidate's fit for this job. Use the CV and JD as ground truth.

=== CV ===
{cv_text}

=== JOB DESCRIPTION ===
{jd_text}

=== CANDIDATE ANALYSIS (to be scored) ===
{candidate_output}

Score each dimension from 1 (poor) to 5 (excellent):
- faithfulness: are all claims supported by the CV (no fabricated experience)?
- specificity: does it cite concrete skills/requirements, not generic filler?
- actionability: are the gaps and next steps genuinely useful?
- structure: is it well-organized and easy to act on?

Respond with ONLY this JSON (no markdown fence, no commentary):
{{"faithfulness": int, "specificity": int, "actionability": int, "structure": int, "comment": "one short sentence"}}
"""


def cover_letter_prompt(cv_text: str, jd_text: str, extra: str = "") -> str:
    notes = f"\n\nExtra instructions from the candidate:\n{extra}" if extra.strip() else ""
    return f"""Write a tailored cover letter (or outreach message) for this candidate applying to this role.

=== CANDIDATE CV ===
{cv_text}

=== JOB DESCRIPTION ===
{jd_text}

Requirements:
- ~250–350 words, confident but not arrogant.
- Open with a specific hook tied to the role/company, not a generic greeting.
- Highlight 2–3 concrete achievements from the CV that map to the JD's top needs.
- Do NOT fabricate experience. Use only what's in the CV.
- End with a clear, low-friction call to action.
- Match the candidate's language (Chinese or English) based on the CV.{notes}
"""
