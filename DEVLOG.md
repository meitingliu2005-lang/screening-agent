# Development Log

A running narrative of what was built, why, what broke, and what we
found — written for human reading (interview prep, future-you), as a
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
