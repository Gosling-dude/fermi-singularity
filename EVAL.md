# EVAL.md — Evaluation System, Failures, and a Measured Improvement

Everything in this document was produced by running the system. The raw
inputs and outputs are committed under `eval/results/` — no number here was
typed by hand.

```bash
make eval-baseline    # reproduces eval/results/run_baseline/
make eval-improved    # reproduces eval/results/run_improved/
make eval-compare     # prints the table in §7
```

---

## 1. What "success" means for this product

The product's defining promise is that a learner can trust it — both when it
answers and when it declines. So success is defined on four axes, in priority
order:

| Axis | Question it answers | How it is measured |
|---|---|---|
| **Groundedness** | Does every claim come from the episodes? | forbidden-claim detection (deterministic) + judge faithfulness |
| **Honest refusal** | Does it decline when the episodes don't cover it? | refusal accuracy, false-refusal count |
| **Verifiability** | Can the learner check the answer against the audio? | citation presence + timestamp validation |
| **Usefulness** | Is the answer actually good? | judge completeness + clarity |

A fluent, well-cited, confidently wrong answer is a **failure**, and the
scoring is built so it cannot pass: the judge's pass criterion requires
faithfulness ≥ 4, citation correctness ≥ 4 *and* refusal correctness ≥ 4, so a
high clarity score cannot rescue an ungrounded answer.

---

## 2. The evaluation system

Three components, all under `eval/`:

**`cases.yaml` — 20 cases across 8 categories.** Each declares what the
retriever should surface (`must_retrieve`, `episode`), what a good answer
covers (`key_points`), whether it should refuse, and — for out-of-scope cases —
`forbidden_claims`: facts that are *true* and that the model certainly knows,
but that the episodes never state. Their appearance is direct evidence of
ungrounded generation.

| Category | Cases | What it probes |
|---|---|---|
| `factual` | 4 | single-episode recall, including a numeric value |
| `cross_episode` | 3 | synthesis across two or more episodes |
| `explanation` | 2 | "explain this simply" — clarity vs faithfulness |
| `recommendation` | 1 | "which episode should I listen to, and why" |
| `locate` | 2 | "take me to the part where…" |
| `refusal` | 4 | out-of-scope, including two topically adjacent traps |
| `ambiguous` | 1 | a term used differently in two episodes |
| `followup` | 3 | multi-turn; the **second** turn is the one scored |

**`runner.py` — executes cases against the real product.** It constructs the
same `CompanionAgent` that `make chat` uses, with the same retriever and the
same prompts. There is no evaluation-only code path and no hard-coded answer
anywhere. Multi-turn cases replay through a single `Session`, so follow-up
resolution is genuinely exercised.

Every case writes a full record to `raw.jsonl`: the conversation, the retrieved
chunks with their dense/BM25/rerank scores, the answer, parsed citations with
their validation verdicts, latency, token counts and estimated cost.

**Two layers of scoring**, deliberately separated:

- **`checks.py` — deterministic.** No LLM, no network. Catches what is
  objectively decidable: an empty answer, a missing citation, a citation whose
  timestamp doesn't overlap any retrieved passage, a forbidden claim, an
  out-of-scope question that got answered, an in-scope question that got
  refused, and whether retrieval found the right episode.
- **`judge.py` — LLM-as-judge.** Scores faithfulness, completeness, citation
  correctness, refusal correctness and clarity on 1–5. It is given *only* the
  passages that were actually retrieved and is told explicitly that its own
  physics knowledge is not admissible evidence — otherwise it would penalise a
  correct refusal and reward a fluent hallucination.

Relying on the judge alone would be a mistake: it is the same class of system
being tested. The deterministic layer is the backstop, and it is what makes
the offline mode below possible.

### 2.1 Offline mode — why the headline numbers cost $0

`make eval-retrieval` runs every case through the real pipeline but stops after
retrieval and the evidence gate. No LLM is called. This measures exactly the
two things the improvement in §6 targets — retrieval quality and refusal
behaviour — with **no API key and no cost**, which is why it is the mode used
for the baseline/improved comparison in this document.

`make eval` runs the full pipeline including generation and the judge, and
requires an API key. See §9 for what it adds and why it has not been run yet.

---

## 3. Baseline methodology

The baseline is the naive-RAG configuration this project would have shipped
without the improvement: **dense-only retrieval, no reranking, and a fixed
cosine-similarity threshold as the refusal gate** (`MIN_RELEVANCE=0.30`).

It is the real system with two flags flipped, not a separate implementation:

```bash
RETRIEVAL_MODE=dense ENABLE_RERANK=false python -m eval.runner --retrieval-only
```

Corpus, chunking, embeddings and cases are identical between the two runs. The
only difference is the retrieval and gating strategy.

> **On the audio used.** The runs below were executed against three synthetic
> development episodes (`scripts/fixtures/`, ~20 minutes total) generated
> locally, because the supplied Fermi episodes were not available on this
> machine. The audio is real audio and the ASR, retrieval and evaluation are
> all real — but the *absolute* numbers describe this fixture corpus, not the
> Fermi collection. Re-running `make ingest && make eval-baseline &&
> make eval-improved` on the real episodes regenerates everything, including
> the calibrated gate, which is computed per corpus. The methodology and the
> direction of the finding transfer; the specific percentages should be
> re-measured.

---

## 4. Baseline results

`eval/results/run_baseline/summary.json`

```
Pass rate                 15/20   (75%)
Retrieval: episode hit           100%
Retrieval: term coverage          91%
Refusal accuracy            0/4   (0%)
False refusals                     0
Mean latency                   0.01s
```

Per category:

| Category | Baseline |
|---|---|
| factual | 4/4 |
| cross_episode | 3/3 |
| explanation | 2/2 |
| locate | 2/2 |
| recommendation | 1/1 |
| ambiguous | 1/1 |
| followup | 2/3 |
| **refusal** | **0/4** |

Retrieval is strong. Refusal is completely broken — the gate fired for none of
the four out-of-scope questions.

---

## 5. Three real failures, inspected

These are copied from `eval/results/run_baseline/raw.jsonl`.

### Failure 1 — `refusal_02`: a topic the episodes never mention

**Question:** "What do these episodes say about how superconductivity works?"

**Retrieved context (top 2 of 5):**
```
The Birth Of The Quantum  5:17–6:38  (cosine 0.6384)
  "…be built out of indivisible units. Energy comes in packets. That idea
   turns out to be the single most productive idea of 20th century physics…"

The Birth Of The Quantum  6:25–6:58  (cosine 0.6326)
  "…called Planck units today, we will come back to that, because when we do
   the Bell's theorem episode…"
```

**What happened:** max cosine similarity 0.638, far above the 0.30 gate, so the
system treated the question as answerable and passed these passages to the
model.

**Why it failed:** the word "superconductivity" appears nowhere in the corpus,
but the *question* is about 20th-century physics and so are the passages. A
bi-encoder embeds query and passage independently and scores their topical
proximity — it has no way to express "these are about the same field but this
passage does not answer this question."

**Root cause:** cosine similarity between independently-embedded texts measures
aboutness, not answerability. It is the wrong quantity for a refusal decision.

### Failure 2 — `refusal_04`: the adjacent-topic trap

**Question:** "What did Planck say about the double-slit experiment in this
episode?"

**Retrieved context (top 1):**
```
The Birth Of The Quantum  6:25–6:58  (cosine 0.6927)
  "…called Planck units today, we will come back to that, because when we do
   the Bell's theorem episode…"
```

**What happened:** cosine 0.693 — the *highest* score of any refusal case, and
higher than legitimately answerable questions such as `factual_02` (0.684).

**Why it failed:** the question names a real entity from the episode (Planck)
and a real concept from physics (the double slit) that the episode never
discusses. Every lexical and semantic signal says "this is about the Planck
episode." Only reading the passages reveals the double slit is absent.

**Root cause:** the same as Failure 1 but sharper — this case proves no single
threshold on bi-encoder cosine can work, because the score distributions of
in-scope and out-of-scope questions genuinely *overlap*:

```
in-scope  cosine range:  0.595 – 0.809
out-of-scope    range:  0.396 – 0.693      ← overlapping
```

### Failure 3 — `followup_02`: a follow-up with no topical anchor

**Conversation:**
```
turn 1: "Tell me about Shannon's noisy channel theorem."
turn 2: "Why couldn't engineers actually use it at the time?"   ← scored
```

**Query actually sent to the retriever:** `Why couldn't engineers actually use
it at the time?`

**Retrieved context:**
```
Shannon And The Birth Of Information  0:00–1:30  (cosine 0.6203)
  "Welcome back to Great Papers. Last time we looked at Max Planck…"
The Birth Of The Quantum  5:17–6:38   (cosine 0.6031)
```

**What happened:** term coverage 0% — neither "non constructive" nor "codes"
appeared in the retrieved passages. The right answer (the proof is
non-constructive; practical codes took decades) sits in a passage that was
never retrieved.

**Why it failed:** stripped of its conversational context, the second turn
contains no content words at all. The retriever matched generic
episode-introduction language, which is the nearest thing to a contentless
question.

**Root cause:** a distinct problem from the first two — this is query
*formulation*, not relevance *judgement*. The system has a query-rewriting step
for exactly this, but it calls the LLM, and this run had no API key. See §8.

---

## 6. The improvement

Failures 1 and 2 are the same root cause and account for **4 of the 5 baseline
failures**, in the category the product's core promise depends on. That is the
one thing worth fixing.

**Hypothesis.** A cross-encoder reads the query and the passage *together* and
is trained to judge whether the passage answers the query, rather than whether
it is on the same topic. Its scores should separate in-scope from out-of-scope
questions where cosine cannot.

**Change.** Replace the bi-encoder cosine evidence gate with a cross-encoder
relevance gate:

1. Rerank the fused candidate pool with
   `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters, CPU, no API key,
   ~35 ms for 8 passages). This also improves final ordering.
2. Normalise the query before scoring. "Take me to the part where they explain
   what a bit is" is a request wrapped around a topic; scored verbatim the
   cross-encoder grades the wrapper and returns −1.51, but scored as "what a
   bit is" it returns +3.87.
3. Gate on the reranker score instead of cosine.

### 6.1 The threshold — and how I avoided fitting it to the test set

My first attempt used the cross-encoder's natural decision boundary (logit
sign, i.e. 0.0). **It made things worse**: refusal accuracy went 0% → 100% but
it introduced **6 false refusals** and the overall pass rate *fell* from 75% to
70%. That run is kept as evidence in `eval/results/run_rerank_only/`.

The false refusals were informative. Collection-level questions
(`cross_03`, −7.14), a numeric-value question (`factual_04`, −0.54) and
un-rewritten follow-ups all score low on a *passage*-relevance model, because
no single passage "answers" them in the sense the model was trained on. And
`factual_04` (in-scope, −0.54) versus `refusal_04` (out-of-scope, −0.58) are
0.04 apart — so a threshold tuned to separate them would be fitted to noise.

So the threshold is not tuned on the evaluation cases at all. It is
**calibrated from the corpus at ingestion time** (`retrieve/calibrate.py`):
eight fixed, mundane, domain-free probe questions that no podcast collection
could answer ("how do I renew my passport online", "what are the rules of
cricket") are scored against the indexed passages. Their scores describe what
"definitely not covered" looks like *for this corpus*, and the gate sits one
point above that band.

On this corpus the probes cluster tightly at −11.0 to −11.3, giving a gate of
**−9.98**, written into `data/index/index_meta.json`. It is recomputed on every
`make ingest`, so it adapts to whatever audio is supplied.

This makes the gate deliberately **high-precision and low-recall**: it refuses
the clearly unrelated and never refuses something the episodes might support.
Topically adjacent questions are passed to the grounding prompt, which can read
the passages and decide. The two layers have different jobs — the gate is a
cheap pre-filter, the prompt is the semantic judge.

---

## 7. Baseline vs improved

`make eval-compare`

```
               run_baseline  →  run_improved
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                   ┃ Baseline ┃ Improved ┃  Delta ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ Pass rate                │    75.0% │    85.0% │ +10.0% │
│ Retrieval: episode hit   │   100.0% │   100.0% │      0 │
│ Retrieval: term coverage │    90.6% │    90.6% │      0 │
│ Refusal accuracy         │     0.0% │    50.0% │ +50.0% │
│ False refusals           │        0 │        0 │      0 │
│ Mean latency             │    0.01s │    0.57s │ +0.56s │
│ Estimated cost           │  $0.0000 │  $0.0000 │      0 │
└──────────────────────────┴──────────┴──────────┴────────┘

Fixed (2): refusal_01, refusal_02
Regressed: none
```

| Category | Baseline | Improved |
|---|---|---|
| factual | 4/4 | 4/4 |
| cross_episode | 3/3 | 3/3 |
| explanation | 2/2 | 2/2 |
| locate | 2/2 | 2/2 |
| recommendation | 1/1 | 1/1 |
| ambiguous | 1/1 | 1/1 |
| followup | 2/3 | 2/3 |
| **refusal** | **0/4** | **2/4** |

### What improved
Two of four out-of-scope questions are now refused deterministically, before
any model call — so those refusals are free, fast, and cannot be talked out of
by a persuasive prompt.

### What did not
`refusal_03` (Hawking radiation, −8.48) and `refusal_04` (double slit, −0.58)
still clear the gate. This is by design, not an oversight: pulling the gate up
to catch them would cross the in-scope band and start refusing real questions.
They are delegated to the grounding prompt, which offline mode cannot exercise.

### What regressed
Nothing at case level. Mean latency rose 0.01 s → 0.57 s. Almost all of that is
the one-off cross-encoder load on the first query; steady-state reranking is
~35 ms for 8 passages, which is negligible next to an LLM call.

### Honest reading of the headline number
The 85% figure covers only the retrieval and gating layer. It is not a claim
about answer quality, which needs the full run in §9.

---

## 8. Remaining weaknesses

1. **Two adjacent-topic refusals rely entirely on the prompt.** Measured only
   when `make eval` runs with a key. This is the biggest open risk.
2. **Follow-up retrieval degrades without query rewriting** (`followup_02`).
   The rewriting step exists and is enabled by default, but it needs an API
   key, so offline mode cannot show its benefit. A free fallback —
   concatenating the previous question with the follow-up — would likely fix
   this and is the obvious next change.
3. **The corpus is small** (19 chunks, ~20 min). Retrieval metrics are
   optimistic at this scale; `episode_hit` of 100% is much easier with three
   episodes than thirty.
4. **The judge is unvalidated.** I have not measured judge–human agreement, so
   its scores should be read as a signal, not a verdict.
5. **Absolute numbers are corpus-specific.** See the note in §3.
6. **`term_coverage` is a proxy.** It rewards lexical overlap with the terms I
   chose, which under-credits a passage that conveys the idea in other words.

---

## 9. What the full evaluation adds

`make eval` runs generation and the judge, and reports the axes offline mode
cannot: faithfulness, completeness, citation correctness, clarity, citation
validity rate, and forbidden-claim leakage in generated text. It also exercises
query rewriting, which should fix `followup_02`, and the grounding prompt,
which is the only thing that can catch `refusal_03` and `refusal_04`.

It has **not been run**, because no API key was available in this environment.
`eval/results/` therefore contains three real offline runs and no fabricated
full-pipeline numbers. Estimated cost for one full run of all 20 cases with
`claude-sonnet-5` as both chat and judge is **≈ $0.10–0.20**; the runner records
actual token counts and cost in `summary.json`.

---

## 10. Next improvements, in priority order

1. **Free follow-up query expansion** — prepend the previous turn's question to
   a context-dependent follow-up when no LLM is available. Directly targets
   Failure 3 at zero cost.
2. **Run `make eval` and validate the second gate layer** — confirm the
   grounding prompt catches the two adjacent-topic refusals the gate delegates.
3. **A second, semantic gate for adjacent topics** — a single cheap
   `claude-haiku-4-5` call asking "do these passages contain an answer?" would
   likely close `refusal_03`/`refusal_04` for a fraction of a cent.
4. **Sentence-level citation anchoring** — cite the specific sentence rather
   than the whole 75-second chunk, narrowing what a learner must listen to.
5. **Measure judge agreement** against my own labels on the 20 cases, so the
   judge's scores can be trusted quantitatively.
