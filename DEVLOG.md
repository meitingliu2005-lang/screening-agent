# Development Log

A running narrative of what was built, why, what broke, and what we
found — written for human reading (interview prep), as a
companion to the terser git commit history.

---

## Session 1 — Scaffolding, dataset, baseline

**Goal:** stand up the repo and get one real number on the board.

- Initialized the repo structure and picked **SYNERGY** as the dataset
  source — a public benchmark of 26 systematic reviews with real
  human-reviewer ground truth labels. Chose the **Sep_2021** review
  (rodent object-in-context memory task) specifically: 270 papers,
  ~14.8% included — small enough to iterate on cheaply, and topically
  close to cognitive science.
- Built the **TF-IDF + logistic regression baseline** — the standard
  published baseline for this exact task. Used 5-fold stratified
  cross-validation rather than a single train/test split, since with
  only 40 positive examples total, one held-out split would be too
  noisy to trust. Saved the fold assignments to `cv_folds.csv`
  specifically so later scorers could reuse the *exact same splits* —
  that's what makes the eventual three-way comparison fair.
- **Result:** Precision 0.478, Recall 0.275, F1 0.349, PR-AUC 0.445.

**Debugging along the way:**
- A cached data file (parquet) written in one environment wasn't
  readable in another — different `pyarrow` versions write
  incompatible file metadata. Switched the cache format to plain CSV,
  which has no such cross-version risk.
- The dataset package's default download pulled the *entire* 26-review
  archive (427MB) just to get the ~900KB our one review actually needs
  — and a single unresumable download over that much data was also why
  the first attempt died mid-connection. Rewrote it to fetch only the
  ~4 small files the one review needs, directly from the repo. Download
  time went from several fragile minutes to about 2 seconds.

---

## Session 2 — Zero-shot LLM scorer

**Goal:** same dataset, same metrics, only the scoring method changes.

- Built a shared prompt-building module (`llm_common.py`) so the
  zero-shot and (later) RAG scorers would differ *only* in what context
  they retrieve, not in how they talk to the model or parse responses —
  necessary for a fair comparison.
- SYNERGY doesn't ship formal inclusion/exclusion criteria text, so the
  review's own published abstract is used as the criteria proxy — an
  honest stand-in, not a fabrication, since review abstracts state
  their scope directly. Had to reconstruct that abstract from
  OpenAlex's "inverted index" storage format (`{word: [positions]}`)
  back into real sentences.
- Model: `gpt-5.4-mini` — cheap enough that 270 calls cost under $0.50.
- Added retry/backoff and incremental saving (every 20 papers) so a
  ~10-minute, 270-call run could survive a dropped connection without
  starting over.
- **Result:** Precision 0.750, Recall 0.675, F1 0.711, PR-AUC 0.766 —
  beat the baseline on every metric, most importantly recall (missing a
  truly relevant paper is the costly error in screening).

**Bug found later, fixed in Session 3:** the progress counter printed
the same number repeatedly instead of counting up — cosmetic only,
didn't affect correctness of final results.

---

## Session 3 — RAG scorer + Study 1 conclusion

**Goal:** same setup again, but each decision also gets retrieved
similar labeled papers as context. This is the actual point of Study
1 — isolating whether retrieval specifically helps, separate from the
active-learning question Study 2 will ask later.

- **Embeddings:** used OpenAI's `text-embedding-3-small` API rather
  than installing `sentence-transformers` + a local vector DB. For 270
  vectors, an in-memory cosine-similarity search is just as correct as
  a full vector database, and avoids a multi-GB local install after the
  archive-download pain in Session 1.
- **Leakage control:** for a paper in CV fold *i*, retrieved examples
  come only from papers *not* in fold *i* — reusing the baseline's
  exact fold assignments. Without this, the RAG scorer would
  effectively get to peek at its own test set.
- **Bug:** the embeddings API call crashed the whole run on a single
  connection timeout — it had no retry logic, unlike the chat
  completion calls. Added the same retry/backoff protection plus a
  longer timeout. Confirmed fixed by simulating a failed-then-succeeded
  call.
- **Result:** Precision 0.730, Recall 0.675, F1 0.701, PR-AUC 0.763.

### The actual finding

RAG did **not** beat zero-shot. Recall was identical; precision/F1/PR-AUC
were very slightly lower with retrieval, not higher — the opposite of
what the project hypothesized going in.

Rather than read too much into a ~1-2 point gap on only 270 papers (40
positive), ran **McNemar's test** — a paired significance test that
looks specifically at the papers where two scorers disagreed, and asks
whether that disagreement is skewed enough to be real rather than a
coin-flip pattern you'd expect from chance at this sample size.

**Results:**
- TF-IDF vs. Zero-shot LLM: p=0.0026 → significant
- TF-IDF vs. RAG LLM: p=0.0029 → significant
- **Zero-shot vs. RAG LLM: p=1.0 → not distinguishable from noise**

**Conclusion:** both LLM scorers are genuinely, significantly better
than the classic baseline. But the gap between zero-shot and RAG is not
statistically real — the honest finding is **"retrieval made no
measurable difference here,"** not "retrieval made it slightly worse."

**Why might that be, honestly:** the review's own abstract may already
give the model enough scope information that retrieved examples add
little on top. Also, embedding similarity finds *topically* similar
papers, not necessarily the *diagnostically* useful ones — the papers
that actually trip up a screening decision are usually the ambiguous
edge cases, and "most similar by embedding" doesn't reliably surface
those.

## What this means for the project as a whole

Study 1's core question is answered. The interesting result isn't
"LLMs beat TF-IDF" (expected) — it's that **retrieval-augmented context
didn't add measurable value over zero-shot for this task**, which is a
real, defensible, slightly counterintuitive finding rather than a clean
success story. That's arguably more interesting to talk about in an
interview than a result that confirmed the hypothesis.

---

## Session 4 — Adding a PyTorch scorer

**The trigger:** realized partway through that none of the three
scorers built so far used PyTorch at all — TF-IDF+LR is scikit-learn,
zero-shot and RAG are both just OpenAI API calls. If demonstrating
hands-on ML/PyTorch skill matters for the portfolio (it does), that's
a real gap, not a minor one.

**Considered and rejected:** swapping the embedding source from
OpenAI's API to a local `sentence-transformers` model. Technically
"uses PyTorch" since that library runs on it under the hood, but it's
a shallow win — calling `model.encode()` doesn't demonstrate building
anything, and it would reintroduce the heavy-install problem from
Session 1 for no real benefit.

**What got built instead:** a small PyTorch MLP classifier trained
directly on the embeddings already cached during the RAG step —
meaning this scorer needs zero new API calls, just local training.
5-fold CV, same as every other scorer, added as a 4th entry to the
comparison.

**The real methodological point, worth being able to explain clearly:**
1536-dimensional embeddings with only ~216 training examples per fold
is exactly the shape that overfits a careless neural net. Specific
guards, not just "added dropout because that's what you do":
- A single small hidden layer (32 units) — deliberately low capacity
- Weight decay (L2) on the optimizer
- Early stopping on a validation slice carved out of each fold's
  training data, restoring the best-validation-loss weights rather
  than running a fixed epoch count

**Sanity check that actually mattered:** tested with embeddings
containing zero real signal (pure noise) and confirmed out-of-fold
PR-AUC landed right at the dataset's true positive rate (~0.148) rather
than being spuriously inflated. That's the check that would have
caught a leakage bug if one existed — a high-dimensional model trained
on noise that still "performs well" out-of-fold is the classic
signature of train/test contamination somewhere in the CV loop.

Study 1 now compares four scorers spanning a real range of approaches —
classic ML (TF-IDF+LR), zero-shot LLM, retrieval-augmented LLM, and a
locally-trained neural net on embeddings — which between them touch
most of the modern NLP/ML toolkit, not just one corner of it.

---

## Session 5 — Catching a misleading significance result

**What happened:** the real PyTorch run came back with Precision 0.299,
Recall 0.575 — clearly better recall than TF-IDF's 0.275, but much
worse precision. McNemar's test flagged TF-IDF vs. PyTorch as
"significant" — easy to misread that as "PyTorch is significantly
better," especially right after seeing it beat TF-IDF on recall and F1.

**The actual issue:** McNemar's test compares raw per-paper correctness
at a 0.5 threshold, not any specific metric like recall. Working
backward from precision/recall: TF-IDF makes ~41 total
misclassifications out of 270 papers; PyTorch makes ~71 — nearly
double, because its much lower precision means a lot more false
positives. The significant McNemar result actually reflects TF-IDF
being more often "correct" in a strict sense, not PyTorch being better
— the opposite of what it's easy to assume at a glance.

**Why this matters more than it looks:** for an imbalanced screening
task, recall is the metric that actually matters (missing a relevant
paper is the costly error), not raw accuracy. A significance test built
around accuracy can point the opposite direction from a significance
test built around the metric you actually care about. This is a
genuinely useful thing to be able to raise unprompted in an interview,
not just a footnote.

**Fix:** added `bootstrap_metric_diff` / `run_bootstrap_tests` to
`compare_study1.py` — bootstrap confidence intervals on the actual
metric gap (recall, PR-AUC) rather than raw correctness. Validated with
two controlled synthetic cases before trusting it on real data:
identical predictions correctly produced a CI of exactly [0, 0]; a
constructed 50-point true recall gap (30% vs. 80%) correctly produced a
CI entirely on the positive side, matching the known direction.

**Takeaway for the project as a whole:** a significance test is only as
informative as the metric it's testing — picking the right one for an
imbalanced task isn't a detail, it changes the conclusion.

**The real results, once run:** confirmed PyTorch's recall advantage
over TF-IDF is genuinely significant (95% CI [+0.125, +0.474]) — the
prediction held. But a more interesting wrinkle showed up that wasn't
anticipated: PyTorch's **PR-AUC** is statistically tied with TF-IDF (CI
includes 0), even though its recall is significantly higher. Since
PR-AUC measures ranking quality across every threshold and recall only
measures it at exactly 0.5, that combination suggests PyTorch's recall
gain is partly a threshold artifact — its scores happen to sit more
generously relative to 0.5, not necessarily a genuinely better-ordered
ranking. Meanwhile PyTorch's recall vs. the LLM scorers came back NOT
significant (too few positives for that single-threshold test to have
power), but its PR-AUC vs. both LLM scorers was clearly, significantly
worse. Net effect: three tests, three different verdicts on the same
pair, each one correct for the question it's actually asking. The
honest conclusion: PyTorch is a real, free improvement over TF-IDF on
recall specifically, but on the more robust full-curve metric, it's the
weakest of the three non-baseline approaches.

---

## Session 6 — Study 2: does selection order matter?

**The question:** Study 1 fixed the order papers were judged in and
varied the scoring method. Study 2 flips that — fixes the relevance
signal and varies the ORDER papers get screened in. Does adaptively
choosing what to screen next reach high recall faster than a passive
order?

**Scoping decision that mattered:** rather than building a new "agent"
that makes fresh LLM calls to decide what to screen next, Study 2
reuses Study 1's already-computed artifacts entirely — the zero-shot
LLM's saved scores and the cached embeddings. Zero new API calls,
zero new cost. This works because the zero-shot scorer judges each
paper independently (its score for paper X doesn't depend on screening
order), so there's no need to re-query it for a simulation.

**Four policies, designed for leakage safety from the start:**
embeddings are known for every paper up front (computed from
title/abstract text, not from labels) — only the LABEL of a paper is
"hidden" until it's selected for screening. Every policy picks the next
paper using only embeddings of unscreened papers and labels of
already-screened ones.
1. Random order (baseline, averaged over 100 runs)
2. Greedy by LLM score — passive, deterministic, no adaptation
3. Relevance sampling — adaptive, picks closest to the relevant-papers
   centroid found so far
4. Uncertainty sampling — adaptive, retrains a cheap logistic
   regression on embeddings as labels get revealed, picks whichever
   remaining paper it's least confident about

**Metric:** WSS@95% (Work Saved over Sampling) — borrowed directly from
the actual systematic-review-automation literature, not invented for
this project. It's the fraction of papers you could skip while still
finding 95% of the truly relevant ones.

**Validation before trusting any of this:**
- Random order's simulated curve matched the theoretical expectation
  (recall@N ≈ N/n) to within 0.01 — confirms no bug in the basic
  simulation mechanics.
- `policy_greedy_score` takes no label argument at all — structurally
  cannot leak, not just "tested to not leak."
- Pure-noise embeddings/scores → all four policies landed in the same
  narrow band (WSS 0.07–0.12), none spuriously inflated. This is the
  same kind of sanity check used for the PyTorch scorer in Session 4 —
  if a leakage bug existed, this is exactly the test that would catch
  performance that's "too good" given there's no real signal to exploit.

**An interesting wrinkle caught during testing, not a bug:** in an
early test with strong synthetic structure (positives clearly
clustered in embedding space), uncertainty sampling performed barely
better than random — while relevance sampling and greedy-by-score
both shot up almost immediately. Diagnosed by checking the
uncertainty-sampling model's accuracy mid-run: it reached a perfect
AUC of 1.0 after only 30 of 270 papers, but only 6 of those 30 were
actually positive. Uncertainty sampling deliberately seeks *ambiguous*
boundary cases to learn an accurate model fast — which is a different
goal from *finding positives* fast. This matches a real, published
result in the screening literature (certainty/relevance-based query
strategies generally outperform uncertainty sampling specifically for
recall-efficient screening) — a good thing to know going in rather
than be surprised by on the real data.

---

## Session 7 — Closing the rigor gap in Study 2

**The problem:** Study 2 had a real asymmetry. Stochastic policies
(random, relevance sampling, uncertainty sampling) already ran 50-100
repeated simulations, so there was variation to learn from. The
deterministic greedy-by-score policy ran exactly once — no run-to-run
variation exists for a fixed ranking, so its reported WSS@95 had no
uncertainty estimate at all. Reporting four point estimates side by
side, sorted by value, implied a confidence that hadn't actually been
earned for at least one of them.

**The fix:** a paired bootstrap over the PAPERS themselves, not over
policy randomness. Resample the 270 papers with replacement, then
evaluate every policy's already-computed preference order against that
resampled population (expanding each policy's fixed ranking according
to how many times each paper appears in the resample). The same
resample is shared across all four policies in a given iteration, so
pairwise gaps are paired comparisons. This works identically for the
deterministic and stochastic policies — for greedy, all the variation
comes from sample composition; for the others, it's a mix of sample
composition and which of their precomputed runs gets used.

**Validated against two known cases before trusting it:**
- Two policies given the literally identical order → CI of exactly
  [0, 0], zero false positive
- A constructed perfect-ordering vs. worst-ordering pair (all positives
  first vs. all positives last) → CI entirely on the correct side
  ([-0.89, -0.81]), matching the obvious true direction

**Scope honesty, worth stating plainly:** this tests whether each
policy's already-observed ranking is robust to a different sample of
papers — it does NOT re-run the full adaptive decision loop under each
resampled population (that would mean redoing the active-learning
simulation per bootstrap draw, a much larger compute cost for a
marginal gain in rigor here). Same scope-vs-cost tradeoff as the PR-AUC
bootstrap in Study 1.

**Takeaway:** the same lesson as Session 5, generalized — a point
estimate without a confidence interval isn't a result, it's a number
that happens to be true of this one run. Closing this gap means Study
2 is now held to the same evidentiary bar as Study 1, not a lesser one.

**The real results, once run — and this is the moment the rigor work
paid for itself.** The point estimates suggested all three real
policies beat random: greedy 0.252, uncertainty 0.199, relevance 0.150,
vs. random's 0.068. The pairwise significance table told a different
story: of six comparisons, only ONE was significant — "Random vs.
Greedy by LLM score." Relevance sampling's and uncertainty sampling's
apparent advantages over random did not survive the bootstrap; both
came back "not distinguishable from noise." Without this test, the
natural (wrong) conclusion would have been "all three adaptive
strategies help, ranked by size." The actual finding: only the LLM's
direct score provides evidence strong enough to trust.

**Why this matters beyond Study 2 specifically:** put next to Study 1's
results, a single clean pattern emerges across the ENTIRE project.
Study 1: the LLM's direct zero-shot judgment beat embedding-based RAG
retrieval (no measurable gain) and beat a classifier trained on
embeddings (PyTorch MLP, weakest scorer). Study 2: the LLM's direct
score is the only selection policy that statistically beats random —
both the embedding-similarity policy (relevance sampling) and the
locally-trained-model policy (uncertainty sampling) fail to clear that
bar. Two independent experiments, same conclusion: in this domain,
nothing built on top of the embeddings — not retrieval, not a trained
classifier, not adaptive sampling — outperforms the LLM's direct
judgment. That's the actual headline finding of the whole project, and
it only became visible because the significance testing was taken
seriously at every step rather than stopping at point estimates.
