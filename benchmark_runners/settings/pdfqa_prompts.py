from __future__ import annotations

import hashlib

# Final frozen prompt registry for the PDF-QA benchmark.
# The previous partial Qwen3-VL-4B run is not part of the formal benchmark.
PROMPT_VERSION = "pdfqa-prompt-v2-strict-min"


# =============================================================================
# Ordinary PDF-QA
# =============================================================================

ORDINARY_QA_PROMPT_V2 = """You are answering a scientific PDF question using one supplied complete scientific paper.

This benchmark contains both answerable and deliberately unanswerable questions. Decide answerability only from the supplied PDF pages.

Some unsupported questions may contain a subtle alteration to a year, number, model/version, entity, threshold, experimental condition, or another detail. Do not silently correct the question to a nearby fact in the paper.

Return exactly one valid JSON object:
{"answer_pre":"<complete but concise final answer>","evidence_pages":[<physical PDF page numbers>]}

IMPORTANT OUTPUT FORMAT:
- evidence_pages must be a JSON array of integer page numbers only.
- Correct example:
  {"answer_pre":"...","evidence_pages":[2,11,18]}
- Never output PDF_PAGE_2, "PDF_PAGE_2", Page 2, "Page 2", 2/20, or any other page-label text inside evidence_pages.
- PDF_PAGE_N_START and PDF_PAGE_N_END are input boundary labels only. In the output JSON, represent that page as the integer N.
- Do not output any text before or after the JSON object.

Rules:
1. Use only information contained in the supplied PDF page images.
2. Inspect the question exactly as written.
3. If the PDF supports the question, answer it concisely and accurately.
4. If the PDF does not support the question as written, return exactly:
   {"answer_pre":"Unanswerable","evidence_pages":[]}
5. Do not guess, repair, reinterpret, or substitute a nearby supported fact for an unsupported question.
6. For answerable questions, evidence_pages must contain the external physical PDF pages that materially support the answer.
7. Use the external PDF_PAGE_N labels to identify physical pages; do not use page numbers printed inside the paper body, header, footer, figure, or table.
8. Evidence page numbers are 1-based physical PDF pages.
9. EVIDENCE MINIMALITY:
   - evidence_pages MUST be the SMALLEST SUFFICIENT SET of physical PDF pages needed to verify the final answer.
   - Apply a deletion test: remove every page that is not indispensable to verify the answer.
   - If one page alone fully supports the answer, return exactly one page.
   - Add another page only when it contributes a distinct indispensable premise, value, condition, comparison side, formula, or causal/reasoning step.
   - Never include neighboring, background, introduction, related-work, reference, redundant, or merely consulted pages.
   - There is no fixed numeric cap; return the minimum sufficient count.
10. Preserve requested units, signs, precision, formulas, terminology, conditions, model names, dataset names, and comparison direction.
11. Keep answer_pre complete but concise.
12. answer_pre must contain the final answer, not an outline, plan, or reasoning trace.
13. Do not output chain-of-thought, derivations, Markdown fences, citations, comments, <think> tags, or additional JSON keys.
"""


# =============================================================================
# Cross-document PDF-QA
# Old Cross400 and Hard Cross400 intentionally use exactly the same prompt.
# No document map is supplied to either dataset.
# =============================================================================

CROSS_DOCUMENT_QA_PROMPT_V2 = """You are answering a cross-document scientific reasoning question from one merged PDF.

The merged PDF contains multiple independent scientific papers concatenated in order. References in the question such as "Paper 1"/"Doc 1", "Paper 2"/"Doc 2", and "Paper 3"/"Doc 3" refer to the first, second, and third source papers in that merged PDF.

Identify paper boundaries yourself from the supplied PDF pages using titles, authors, front matter, section restarts, references, and content continuity. No external document-boundary map is provided.

Every question in this cross-document benchmark is answerable from the supplied merged PDF. Do not abstain, refuse, or answer "unanswerable".

A complete answer may require synthesizing definitions, assumptions, methods, mechanisms, formulas, quantitative results, limitations, causal relations, experimental findings, or methodological conditions from two or three source papers.

Inspect every paper requested by the question. Do not stop after locating only one relevant document or one supporting fact.

Return exactly one valid JSON object:
{"answer_pre":"<complete but concise final answer>","evidence_pages":[<merged physical PDF page numbers>]}

IMPORTANT OUTPUT FORMAT:
- evidence_pages must be a JSON array of integer merged-page numbers only.
- Correct example:
  {"answer_pre":"...","evidence_pages":[5,31,52]}
- Never output PDF_PAGE_5, "PDF_PAGE_5", Page 5, Doc 2 Page 3, 5/80, or any other page-label text inside evidence_pages.
- PDF_PAGE_N_START and PDF_PAGE_N_END are input boundary labels only. In the output JSON, represent that merged physical page as the integer N.
- Do not output source-document-local page numbers unless they are also the corresponding merged physical page numbers.
- Do not output any text before or after the JSON object.

Rules:
1. Use only information contained in the supplied merged PDF page images.
2. Every question is answerable from the supplied pages. Never output "unanswerable".
3. Resolve Paper/Doc numbering by source-paper order in the merged PDF.
4. Answer every comparison, contrast, mechanism, condition, calculation, transfer, conflict, or conclusion requested.
5. evidence_pages must include the external merged physical pages that materially support all required parts of the answer.
6. Use the external PDF_PAGE_N labels to identify merged physical pages; do not use page numbers printed inside an individual source paper.
7. Evidence page numbers are 1-based physical pages of the complete merged PDF.
8. EVIDENCE MINIMALITY:
   - evidence_pages MUST be the SMALLEST SUFFICIENT SET of merged physical PDF pages needed to verify the complete final answer.
   - Apply a deletion test: remove every page that is not indispensable to verify the answer.
   - If one page alone fully supports a required part of the answer, do not add redundant pages for that part.
   - Add another page only when it contributes a distinct indispensable premise, value, condition, comparison side, formula, document-specific fact, or causal/reasoning step.
   - Never include neighboring, background, introduction, related-work, reference, redundant, or merely consulted pages.
   - Do not pad evidence_pages to match the number of source documents.
   - There is no fixed numeric cap; return the minimum sufficient count that still supports all required parts of the cross-document answer.
9. Preserve technical terms, formulas, signs, units, numerical values, theorem names, architecture names, method names, dataset names, and document-specific distinctions.
10. Keep answer_pre complete but concise.
11. answer_pre must contain the final answer, not a plan, hidden reasoning trace, or derivation transcript.
12. Do not output chain-of-thought, derivations, Markdown fences, citations, comments, <think> tags, or additional JSON keys.
"""


# =============================================================================
# Single-document compositional reasoning PDF-QA
# Old Reasoning100 and Hard Reasoning100 use exactly the same prompt.
# =============================================================================

REASONING_QA_PROMPT_V2 = """You are answering a compositional reasoning question about one supplied scientific PDF.

Every question in this reasoning benchmark has a verified answer supported by the supplied PDF. Do not abstain, refuse, or answer "unanswerable".

The answer may require combining multiple facts from non-adjacent physical PDF pages and then performing one or more reasoning operations, including conditional inference, causal chaining, comparison, multi-step calculation, or constraint intersection.

The benchmark is deliberately constructed so that one isolated premise may be insufficient. Resolve hidden bridge entities, conditions, quantities, mechanisms, or dependencies from the PDF before producing the final answer.

Inspect the complete supplied PDF. Do not stop after finding only one relevant premise. Internally identify every distinct fact needed for the answer and internally perform the required reasoning.

Return exactly one valid JSON object:
{"answer_pre":"<complete but concise final answer>","evidence_pages":[<physical PDF page numbers>]}

IMPORTANT OUTPUT FORMAT:
- evidence_pages must be a JSON array of integer page numbers only.
- Correct example:
  {"answer_pre":"...","evidence_pages":[4,17,29]}
- Never output PDF_PAGE_4, "PDF_PAGE_4", Page 4, "Page 4", 4/50, or any other page-label text inside evidence_pages.
- PDF_PAGE_N_START and PDF_PAGE_N_END are input boundary labels only. In the output JSON, represent that page as the integer N.
- Do not output any text before or after the JSON object.

Rules:
1. Use only information contained in the supplied PDF page images.
2. Every question is answerable from the supplied PDF. Never output "unanswerable".
3. Do not answer from the question wording alone when the required facts are in the PDF.
4. Combine all required premises before producing the final answer. Do not infer the terminal answer from a single page when the question requires a bridge or dependency.
5. evidence_pages must include the external physical PDF pages that materially support every required premise.
6. EVIDENCE MINIMALITY:
   - evidence_pages MUST be the SMALLEST SUFFICIENT SET of physical PDF pages needed to verify the final answer and all indispensable reasoning premises.
   - Apply a deletion test: remove every page whose removal would still leave enough evidence to verify the complete answer.
   - If one page contains all indispensable premises, return exactly one page.
   - Add another page only when it contributes a distinct indispensable premise, value, condition, comparison side, formula, bridge fact, or causal/reasoning step.
   - Never include neighboring, background, introduction, related-work, reference, redundant, or merely consulted pages.
   - There is no fixed numeric cap; return the minimum sufficient count.
7. Use the external PDF_PAGE_N labels to identify physical pages; do not use page numbers printed inside the PDF page.
8. Evidence page numbers are 1-based physical PDF pages.
9. Preserve requested units, signs, precision, formulas, terminology, conditions, model names, dataset names, and comparison direction.
10. Keep answer_pre complete but concise.
11. answer_pre must contain the final answer, not a plan or reasoning trace.
12. Do not output chain-of-thought, derivations, Markdown fences, citations, comments, <think> tags, or additional JSON keys.
"""


# =============================================================================
# Registry
# =============================================================================

PROMPTS = {
    "ordinary": ("ordinary-v2-strict-min", ORDINARY_QA_PROMPT_V2),
    "cross_document": ("cross-document-v2-strict-min", CROSS_DOCUMENT_QA_PROMPT_V2),
    "reasoning": ("reasoning-v2-strict-min", REASONING_QA_PROMPT_V2),
}


def get_prompt(prompt_key: str) -> tuple[str, str]:
    if prompt_key not in PROMPTS:
        raise KeyError(
            f"Unknown prompt_key={prompt_key!r}. "
            f"Available: {', '.join(PROMPTS)}"
        )
    return PROMPTS[prompt_key]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
