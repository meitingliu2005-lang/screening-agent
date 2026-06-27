"""
Shared utilities for the LLM-based scorers (zero-shot and RAG).

Keeping prompt-building and response-parsing logic here means the
zero-shot and RAG scorers differ only in *what context they retrieve*,
not in how they talk to the model or parse its answer — which matters
for a fair comparison between them.
"""

import json
import time
from pathlib import Path

from synergy_dataset import Dataset
from synergy_dataset.base import _get_path_raw_dataset

DEFAULT_MODEL = "gpt-5.4-mini"


def reconstruct_abstract(inverted_index: dict) -> str:
    """Rebuild plain text from OpenAlex's inverted-index abstract format.

    OpenAlex stores abstracts as {word: [positions]} rather than plain
    text. The synergy_dataset package already does this reconstruction
    internally for candidate papers (via pyalex), but the review's own
    publication metadata stores it raw, so we need to do it ourselves
    here to build a clean inclusion-criteria description for prompts.
    """
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def get_review_context(dataset_name: str) -> dict:
    """Pull the systematic review's own title + abstract to use as the
    inclusion-criteria description in prompts.

    SYNERGY doesn't ship formal inclusion/exclusion criteria text, so we
    use the review paper's own abstract as the best available proxy for
    "what is this review trying to find?" — it's an honest substitute,
    not a fabricated one, since review abstracts typically state scope
    and rationale directly.
    """
    review_path = Path(_get_path_raw_dataset(), dataset_name, "metadata_publication.json")
    with open(review_path, encoding="utf-8") as f:
        pub = json.load(f)

    title = pub.get("title", "")
    abstract = reconstruct_abstract(pub.get("abstract_inverted_index", {}))
    return {"title": title, "abstract": abstract}


def build_screening_messages(review_context: dict, paper_title: str, paper_abstract: str,
                              examples: list = None) -> list:
    """Build the chat messages for a screening decision.

    `examples` is None for the zero-shot scorer, and a list of
    {title, abstract, label_included} dicts (retrieved similar papers)
    for the RAG scorer — that's the only difference between the two.
    """
    system = (
        "You are assisting with title/abstract screening for a systematic "
        "review. Given the review's scope and a candidate paper, judge "
        "whether the candidate paper should be INCLUDED in the review.\n\n"
        "Respond with ONLY a JSON object of the form:\n"
        '{"reasoning": "<one or two sentence rationale>", '
        '"probability_relevant": <float between 0.0 and 1.0>}\n\n'
        "probability_relevant is your estimated probability that this "
        "paper meets the review's inclusion criteria (1.0 = certainly "
        "include, 0.0 = certainly exclude)."
    )

    user_parts = [
        f"REVIEW TITLE: {review_context['title']}",
        f"REVIEW SCOPE (from the review's own abstract): {review_context['abstract']}",
    ]

    if examples:
        user_parts.append("\nSIMILAR PREVIOUSLY-SCREENED PAPERS (for reference):")
        for ex in examples:
            decision = "INCLUDED" if ex["label_included"] else "EXCLUDED"
            user_parts.append(
                f"- [{decision}] {ex['title']}: {ex['abstract'][:300]}"
            )

    user_parts.append(f"\nCANDIDATE PAPER TITLE: {paper_title}")
    user_parts.append(f"CANDIDATE PAPER ABSTRACT: {paper_abstract}")
    user_parts.append("\nShould this candidate paper be included?")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def parse_llm_response(raw_text: str) -> tuple:
    """Parse the model's JSON response into (probability, reasoning).

    Falls back to a neutral 0.5 probability if parsing fails, rather
    than crashing a 270-call run over one bad response.
    """
    try:
        data = json.loads(raw_text)
        prob = float(data.get("probability_relevant", 0.5))
        prob = min(max(prob, 0.0), 1.0)  # clamp to valid range
        reasoning = str(data.get("reasoning", ""))
        return prob, reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.5, f"PARSE_FAILED: {raw_text[:200]}"


def call_model_with_retry(client, messages: list, model: str = DEFAULT_MODEL,
                           max_retries: int = 3) -> tuple:
    """Call the LLM with simple retry/backoff, return (probability, reasoning)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw_text = response.choices[0].message.content
            return parse_llm_response(raw_text)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [WARN] giving up after {max_retries} attempts: {e}")
                return 0.5, f"API_FAILED: {e}"
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)
