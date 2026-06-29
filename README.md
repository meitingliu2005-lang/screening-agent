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
- [x] TF-IDF + logistic regression baseline (`src/baseline.py`)
- [x] Zero-shot LLM scorer (`src/zero_shot.py`, `src/llm_common.py`) — uses OpenAI `gpt-5.4-mini`
- [x] Embedding + retrieval (`src/embed.py`) — OpenAI `text-embedding-3-small`, in-memory cosine similarity (no vector DB needed at this scale)
- [x] RAG scorer (`src/rag.py`) — same prompt/model as zero-shot, plus 5 retrieved similar papers as context, leakage-controlled via the same CV folds
- [x] Study 1 comparison (`src/compare_study1.py`) — including paired significance testing
- [x] PyTorch MLP classifier on embeddings (`src/pytorch_classifier.py`) — a 4th scorer, added to showcase hands-on PyTorch model-building (the other scorers use scikit-learn or API calls, no PyTorch)
- [ ] Study 2 (active selection policies)
- [ ] FastAPI serving layer

## Results so far

**TF-IDF + logistic regression baseline** (5-fold stratified CV, n=270, 40 positives):

| Precision | Recall | F1 | PR-AUC |
|---|---|---|---|
| 0.478 | 0.275 | 0.349 | 0.445 |

**Zero-shot LLM** (`gpt-5.4-mini`, no retrieved context, n=270):

| Precision | Recall | F1 | PR-AUC |
|---|---|---|---|
| 0.750 | 0.675 | 0.711 | 0.766 |

The zero-shot LLM beats the baseline on every metric — most importantly
recall, which is the metric that matters most for screening (missing a
truly relevant paper is the costly error). The baseline caught 27.5% of
truly relevant papers; the zero-shot LLM caught 67.5%.

**RAG LLM** (`gpt-5.4-mini`, same prompt + 5 retrieved similar papers as context, n=270):

| Precision | Recall | F1 | PR-AUC |
|---|---|---|---|
| 0.730 | 0.675 | 0.701 | 0.763 |

### Study 1 conclusion

Both LLM scorers are significantly better than the TF-IDF baseline
(McNemar's test, p<0.01 for both). But **zero-shot vs. RAG is not
statistically distinguishable** (p=1.0) — the small gap between them is
noise, not a real effect. The honest finding: retrieval-augmented
context did not measurably improve screening accuracy over zero-shot
for this task. See `DEVLOG.md` for the full narrative, including why
that might be true and what was tried along the way.

Study 1 is complete. Next: Study 2 (active selection policies) or a
FastAPI serving layer, depending on which direction is more useful next.

### Why a 4th scorer (PyTorch MLP)

The first three scorers use scikit-learn (baseline) or OpenAI API calls
(zero-shot, RAG) — no hands-on model-building anywhere. The PyTorch MLP
scorer trains a small neural net directly on the embeddings already
cached by the RAG step (no new API calls needed), evaluated through
the same 5-fold CV as everything else.

Worth being explicit about: with 1536-dimensional embeddings and only
~216 training examples per fold, this is exactly the high-dimension/
low-sample-size setup that overfits a careless neural net. Guards
against that: a single small hidden layer (32 units), weight decay,
and early stopping on a validation slice carved out of each fold's
training data (not the test fold itself). Sanity-checked by confirming
that with pure-noise input, out-of-fold PR-AUC sits right at the
dataset's true positive rate rather than being spuriously inflated —
i.e., no leakage, no false confidence from overfitting.

### A subtlety in the significance testing

McNemar's test answers "which model is more often right at a 0.5
threshold" — under class imbalance, that can point the opposite
direction from "which model is better at the metric that actually
matters." Concretely: PyTorch MLP has much higher recall than TF-IDF
but lower precision, so it makes *more total misclassifications*
overall — McNemar's correctly flags this pair as significant, but in
raw-correctness terms, not in recall terms.

Added a second test (`run_bootstrap_tests` in `compare_study1.py`):
bootstrap confidence intervals directly on the metric gap (recall,
PR-AUC) rather than raw correctness — resample the papers with
replacement many times, recompute the gap each time, check whether
zero falls inside the resulting interval. This is the test that
actually answers "is a recall advantage real," separate from "which
scorer is more often right." See `DEVLOG.md` for the full reasoning.

### Final reconciled result (all three significance tests)

- **PyTorch vs. TF-IDF:** PyTorch's recall advantage is real (95% CI
  [+0.125, +0.474]), but its PR-AUC is statistically tied with TF-IDF
  (CI includes 0). Recall measures one threshold; PR-AUC measures the
  whole ranking curve — improving at one specific threshold without
  improving the overall ranking suggests the gain is partly about where
  PyTorch's scores happen to sit relative to 0.5, not a fundamentally
  better-ordered set of predictions.
- **PyTorch vs. the LLM scorers:** recall looks similar (not
  statistically distinguishable — only 40 positive examples makes this
  a low-powered comparison), but PR-AUC clearly and significantly
  favors zero-shot/RAG (CI entirely negative). This is the more
  trustworthy comparison, and it says PyTorch is the weakest of the
  three non-baseline scorers.
- **Why McNemar, recall, and PR-AUC disagree for the same pairs:** they
  answer different questions. McNemar tests raw accuracy (where
  PyTorch's low precision hurts it under class imbalance); the
  bootstrap tests isolate one specific metric each. A model can
  legitimately win one and lose another — that's not a contradiction,
  it's three different lenses on the same predictions.

**Bottom line:** PyTorch is a real, free, fast, locally-trained
alternative to TF-IDF with a genuine recall advantage — but on the more
robust full-curve metric (PR-AUC), it's clearly the weakest of the
three non-baseline approaches, and that gap to the LLM scorers is the
one backed by the strongest statistical evidence in the comparison.

## Setup

```bash
pip install -r requirements.txt
python src/data.py        # fetches + caches the review locally
```

You'll need an `OPENAI_API_KEY` set as an environment variable for the
LLM-based scorers — put it in a local `.env` file in the project root
(not committed to git):
```
OPENAI_API_KEY=sk-...
```
