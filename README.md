# Screening Agent

A small research project comparing how different scoring methods perform
at literature screening (the title/abstract triage step of a systematic
review), using a dataset with real human-reviewer ground truth.

This grew out of a blicket-detector project comparing how LLM agents vs.
children explore a causal-learning puzzle. The throughline here is the
same question applied to a different task: **how does an AI's
decision-making process compare to a human's, not just its final
accuracy?**

## Design: two separate studies, not one tangled one

RAG ("does retrieved context improve a classification decision?") and
active learning ("which unlabeled item should I look at next?") are
different research questions. Bundling them into one system makes it
impossible to know what's actually responsible for any performance gain.
So this project is split into two clean studies:

### Study 1 — Classification quality (current focus)
**Fixed:** sample order (a static random sample, no adaptive selection).
**Varied:** the scoring method.
**Question:** does retrieval-augmented context actually improve relevance
judgments, compared to a classic baseline and a zero-shot LLM?

Three scorers compared on the same fixed sample:
1. TF-IDF + logistic regression (the standard published baseline for this
   kind of task)
2. Zero-shot LLM judgment (no retrieved context)
3. RAG: LLM judgment with retrieved similar labeled examples as context

Metrics: precision / recall / F1 / PR-AUC.

### Study 2 — Selection efficiency (next)
**Fixed:** the scoring method (whichever wins Study 1).
**Varied:** the order papers are screened in.
**Question:** does adaptively choosing what to screen next get you to high
recall faster than a fixed/random order?

Compared: random order, relevance sampling (similarity to known-relevant
centroid), uncertainty sampling.

Metric: recall@N curves (% of true relevant papers found after screening
N items), simulated against the dataset's ground truth.

### Stretch: full factorial
If time allows, cross every scorer with every selection policy to measure
main effects and interaction — does a better scorer matter more under
active selection, or does it help regardless?

## Dataset

[SYNERGY](https://github.com/asreview/synergy-dataset) — an open dataset
of 26 systematic reviews with real inclusion/exclusion labels. This
project uses **Sep_2021**: "The rodent object-in-context task: a
systematic review and meta-analysis of important variables" — 270 usable
records, ~14.8% included. Chosen for manageable size (cheap/fast to
iterate on) and topical proximity to cognitive psychology / memory
research.

## Status

- [x] Repo scaffolding
- [x] Dataset acquired and cached (`src/data.py`)
- [ ] TF-IDF + logistic regression baseline
- [ ] Embedding + vector store indexing
- [ ] Zero-shot LLM scorer
- [ ] RAG scorer
- [ ] Study 1 evaluation + comparison plot
- [ ] Study 2 (active selection policies)
- [ ] FastAPI serving layer

## Setup

```bash
pip install -r requirements.txt
python src/data.py        # fetches + caches the review locally
```

You'll need an `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) set as an
environment variable once we get to the LLM scorers — put it in a local
`.env` file (not committed).
