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

### Study 2 — Selection efficiency

**Fixed:** the relevance signal — the zero-shot LLM's already-computed
scores, plus the embeddings cached from the RAG step. No new API calls.
**Varied:** the order papers get screened in.
**Question:** does adaptively choosing what to screen next reach high
recall faster than a fixed/passive order?

Four policies compared, via `src/study2.py`:
1. **Random order** — baseline, averaged over 100 runs
2. **Greedy by LLM score** — screen in order of the zero-shot LLM's
   precomputed relevance score, highest first (passive, deterministic)
3. **Relevance sampling** — adaptive: always pick the unscreened paper
   most similar (cosine) to the centroid of relevant papers found *so
   far*
4. **Uncertainty sampling** — adaptive: maintain a logistic regression
   on embeddings, retrained as labels get revealed, always pick
   whichever unscreened paper the model is least confident about

Metric: **WSS@95%** (Work Saved over Sampling) — the standard metric in
the actual systematic-review-automation literature. It's the fraction
of the 270 papers you could *skip* while still finding 95% of the truly
relevant ones.

**Significance testing:** a paired bootstrap over the *papers*
themselves (1000 resamples) — not just policy randomness. Each
resample is shared across all four policies, so pairwise WSS@95 gaps
are paired comparisons, and even the deterministic greedy policy gets
a real confidence interval (from sample-composition variation, since
it has no run-to-run randomness of its own). This is the same
"evaluate a fixed ranking under resampling" approach already validated
for the PR-curve bootstrap in Study 1. Validated against two known
cases before trusting it: identical orders produced an exact [0,0] CI;
a constructed perfect-vs-worst-case ordering produced a CI entirely on
the correct side, matching the known direction.

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
- [x] Study 2: active selection policies (`src/study2.py`) — needs zero new API calls, reuses Study 1's cached scores/embeddings
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

Study 1 is complete with four scorers compared. Study 2 (below) builds
on these results directly — see "Study 2 results" further down.

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

## Study 2 results

| Policy | Mean papers to 95% recall | WSS@95% |
|---|---|---|
| Greedy by LLM score | 202.0 | 0.252 |
| Uncertainty sampling | 216.4 | 0.199 |
| Relevance sampling | 229.6 | 0.150 |
| Random order | 251.6 | 0.068 |

All three real policies beat random, but gains are modest, not dramatic
— the best (greedy by LLM score) only saves 25% of screening effort,
far below the 40-60%+ sometimes reported in the active-learning
literature. Worth being explicit about why: WSS@95 is bottlenecked by
the *worst-ranked* true positives, not average ranking quality — a
policy can rank most relevant papers well and still need to screen deep
to catch a long tail of atypical ones. With only 270 papers and 40
positives, there's also less room for any policy to show a dramatic
effect than on a corpus of thousands.

**The more interesting finding:** greedy by LLM score beats both
embedding-based adaptive policies, and relevance sampling (also
embedding-similarity-driven) is the *weakest* of the three real
policies. This is the same limitation Study 1 found with RAG —
embedding similarity in this domain captures topical resemblance, not
whatever deeper judgment the LLM's direct evaluation is making. Same
conclusion, surfacing independently in two different experiments.

**Significance testing — and this changed the conclusion substantially.**
The point estimates above suggested all three real policies beat random.
The rigorous version says something different: **only "Random vs.
Greedy by LLM score" is statistically significant.** Relevance sampling
and uncertainty sampling's apparent advantages over random (point
estimates of 0.150 and 0.199 vs. random's 0.068) are **not
distinguishable from noise** once sampling uncertainty is accounted for
— with only 40 positive papers, those gaps could just as easily be this
particular sample's luck as a real effect.

| Policy | Bootstrap mean WSS@95 | 95% CI |
|---|---|---|
| Greedy by LLM score | 0.313 | [0.174, 0.737] |
| Uncertainty sampling | 0.191 | [0.052, 0.515] |
| Relevance sampling | 0.137 | [0.048, 0.304] |
| Random order | 0.065 | [0.007, 0.182] |

**The real conclusion: only the LLM's direct score provides a
statistically defensible advantage over random screening.** Neither
embedding-based policy clears that bar.

**This connects directly to Study 1, and it's the cleanest finding in
the whole project.** In Study 1, the LLM's direct zero-shot judgment
beat embedding-based RAG retrieval (no measurable gain from retrieval)
and beat a classifier trained on embeddings (PyTorch MLP, the weakest
scorer). In Study 2, the LLM's direct score is the only policy that
statistically beats random — both the embedding-similarity policy and
the locally-trained-model policy fail to clear that bar. **Across both
experiments, anything built on top of the embeddings — retrieval, a
trained classifier, an adaptive sampling strategy — consistently
underperforms or fails to show a robust advantage over the LLM's direct
judgment.** That conclusion shows up independently in two separate
studies, which is what makes it credible rather than a coincidence.

One honest caveat: greedy's bootstrap mean (0.313) is noticeably higher
than its raw point estimate (0.252), with a wide CI ([0.174, 0.737]).
Not a bug — with only 40 positives, "papers needed to hit 95% recall"
is a discontinuous statistic that can swing under resampling. The
direction of the effect is solid; its exact magnitude is genuinely
uncertain given this sample size.

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
