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
retrieval and the evidence gate. **No LLM is called — even when a key is
configured**, so its numbers are free, deterministic, and identical on any
machine. This measures exactly the two things the improvement in §6 targets:
retrieval quality and refusal behaviour.

`make eval` runs the full pipeline including generation and the LLM judge, and
requires an API key. Results in §9.

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
The 85% figure covers only the retrieval and gating layer — it is not a claim
about answer quality. The full pipeline, measured separately in §9, scores
19/20 with refusal accuracy 4/4 and citation validity 100%.

---

## 8. Remaining weaknesses

1. ~~Two adjacent-topic refusals rely entirely on the prompt.~~ **Resolved and
   measured** — the full run in §9 scores refusal accuracy 4/4, confirming the
   prompt catches what the gate delegates.
2. ~~Follow-up retrieval degrades without query rewriting.~~ **Resolved with a
   key** — `followup_02` passes in the full run. It still fails offline, so a
   free fallback (concatenating the previous question with the follow-up)
   remains worthwhile for keyless operation.
3. **Unsupported narrative completion.** `followup_01` in §9.2 — the model
   inferred a fact the transcript implies but never states. No deterministic
   check catches this class, and the strict grounding prompt did not prevent
   it. This is the biggest open risk in the system.
4. **The corpus is small** (19 chunks, ~20 min). Retrieval metrics are
   optimistic at this scale; `episode_hit` of 100% is much easier with three
   episodes than thirty.
5. **The judge is unvalidated.** I have not measured judge–human agreement, so
   its scores should be read as a signal, not a verdict — though on the two
   cases in §9.1 and the one in §9.2 its rationales were specific and correct
   on inspection.
6. **Absolute numbers are corpus-specific.** See the note in §3.
7. **`term_coverage` is a proxy.** It rewards lexical overlap with the terms I
   chose, which under-credits a passage that conveys the idea in other words.

---

## 9. Full pipeline results (generation + LLM judge)

`eval/results/run_full_openrouter/` — run through OpenRouter with
`anthropic/claude-sonnet-5` for chat and `openai/gpt-5-mini` as judge. The judge
is a **different vendor from the chat model on purpose**: a judge grading its
own family's output is prone to self-preference bias, and OpenRouter makes
cross-vendor judging a one-line change on a single key.

```
Pass rate                19/20   (95%)
Refusal accuracy           4/4  (100%)
False refusals                      0
Citation validity                100%   (0 invalid of 68)
Retrieval: episode hit           100%
Retrieval: term coverage          94%
Judge: faithfulness              4.85
Judge: completeness              5.00
Judge: citation correctness      5.00
Judge: refusal correctness       5.00
Judge: clarity                   5.00
Mean latency                    8.56s
Actual cost                   $0.2735   (reported by OpenRouter, not estimated)
```

Three things are worth drawing out.

**The two-layer refusal design is validated.** Offline, the calibrated gate
catches 2 of 4 refusals and deliberately delegates the two topically adjacent
ones. With generation enabled, refusal accuracy is **4/4** — the grounding
prompt caught exactly the cases the gate passed to it. That is the design
working as intended, and it is the answer to the open risk flagged in §8.

**Citation validity is 100% across 68 citations.** No fabricated episode, no
invented timestamp, nothing stripped by the validator.

**Query rewriting fixes Failure 3.** `followup_02` — the un-anchored follow-up
that failed offline — passes here. "Why couldn't engineers actually use it at
the time?" was rewritten to "Why couldn't engineers actually use Shannon's
noisy channel theorem at the time?" and retrieval found the right passage.

### 9.1 Two defects the evaluation found — in the evaluation and in the product

The first full run scored 18/20. Both failures turned out to be defects in *my
own work*, not in the system's reasoning, and both are worth recording because
finding them is the entire point of building an evaluation. That run is kept at
`eval/results/run_full_pre_fixes/`.

**`refusal_03` — a false positive in my deterministic check.** The system
produced a near-perfect refusal ("None of them mention Hawking radiation, black
holes, or black hole evaporation, so I can't answer this from the material
provided") and the judge scored it 5/5/5/5/5. My forbidden-claim check failed it
for containing the word "Hawking" — a term the *learner* had put in the
question. A refusal has to be able to name what it is declining. Fixed: a
forbidden claim that already appears in the case's own input is not evidence of
ungrounded generation. Terms absent from the question ("event horizon",
"virtual particle") are still caught.

**`refusal_01` — an unverifiable claim in my refusal message.** The judge docked
faithfulness to 3 with a precise rationale: the canned message said *"I searched
the transcripts of every episode in this collection"*, which the retrieved
passages cannot support. The judge was right — the statement is true of the
system but is not grounded in the evidence shown. Fixed in the product, not in
the judge: the message now asserts only the outcome. Keeping the judge strict
was the better trade.

### 9.2 The one real remaining failure

`followup_01` fails on faithfulness (3/5), and this one is genuine.

Asked to explain Planck's "act of desperation" more simply, the answer states
that Planck **never succeeded** in finding a derivation without the quantum
assumption. The transcript says he "spent years trying to do exactly that
himself" — it never says he failed. The claim is historically true, which is
precisely why it is dangerous: the model completed the narrative arc that
"spent years trying" implies, and produced an unsupported fact that reads
perfectly naturally.

This is the exact failure mode the product exists to prevent, it survived a
strict grounding prompt, and it was caught only by the judge — not by any
deterministic check. It is the strongest argument in this document for having
an LLM judge at all, and the most useful open problem in the system.

## 10. Next improvements, in priority order

1. **Attack unsupported narrative completion** (§9.2) — the one real open
   failure. A verification pass that re-reads each claim against the cited
   passage, or a prompt rule specifically forbidding inferences the transcript
   only implies, is the obvious next experiment.
2. **Free follow-up query expansion** — prepend the previous turn's question to
   a context-dependent follow-up when no LLM is available, so keyless operation
   matches the full pipeline on `followup_02`.
3. **Broaden the judge across vendors** — run the same cases through two judges
   from different families and report disagreement, which would also give the
   judge-validation §8.5 asks for. OpenRouter makes this a config change.
4. **Sentence-level citation anchoring** — cite the specific sentence rather
   than the whole 75-second chunk, narrowing what a learner must listen to.
5. **Measure judge agreement** against my own labels on the 20 cases, so the
   judge's scores can be trusted quantitatively.
