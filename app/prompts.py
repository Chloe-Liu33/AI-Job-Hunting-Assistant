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
