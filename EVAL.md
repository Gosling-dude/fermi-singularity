# EVAL.md — Evaluation System, Failures, and a Measured Improvement

Everything in this document was produced by running the system. The raw
inputs and outputs are committed under `eval/results/` — no number here was
typed by hand.

**Corpus.** Every number below was measured on the three supplied Fermi
"Great Papers" episodes — Einstein's Special Relativity (52:14), Bell's
Theorem (34:00) and The Dirac Equation and Antimatter (33:00); 1 h 59 m of
audio, 1,350 ASR segments, 92 passages — transcribed by this system from the
raw MP3s with faster-whisper `medium` in 41 minutes on CPU.

> **Historical runs.** `eval/results/synthetic_*` holds an earlier generation
> of results measured against three short locally-synthesised fixture episodes,
> used to build and debug the pipeline before the real audio was available.
> They describe a **different corpus** and are kept only as a record of the
> development process (including one change that made things worse and was
> reverted). Nothing in this document quotes them except where explicitly
> labelled. See `eval/results/README.md`.

```bash
make eval-baseline    # reproduces eval/results/fermi_baseline/
make eval-improved    # reproduces eval/results/fermi_improved/
make eval-compare     # prints the table in §7
make eval             # reproduces eval/results/fermi_full/  (needs a key)
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

Corpus, chunking, embeddings and cases are identical between the two runs, and
both were run against the real Fermi episodes. The calibrated gate is computed
per corpus at ingestion time; for this collection it came out at **−5.421**.

---

## 4. Baseline results

`eval/results/fermi_baseline/summary.json`

```
Pass rate                 14/20   (70%)
Retrieval: episode hit          93.8%
Retrieval: term coverage        84.4%
Refusal accuracy            0/4   (0%)
False refusals                     0
Mean latency                   0.01s
```

Per category:

| Category | Baseline |
|---|---|
| factual | 4/4 |
| cross_episode | 1/3 |
| explanation | 2/2 |
| locate | 2/2 |
| recommendation | 1/1 |
| ambiguous | 1/1 |
| followup | 3/3 |
| **refusal** | **0/4** |

Single-episode retrieval is strong. Two things are broken: refusal fired for
none of the four out-of-scope questions, and cross-episode retrieval dropped
2 of 3 because dense-only ranking let one episode monopolise the context.

---

## 5. Three real failures, inspected

These are copied from `eval/results/fermi_baseline/raw.jsonl`.

### Failure 1 — `refusal_02`: a topic the episodes never mention

**Question:** "What do these episodes say about how superconductivity works?"

**Retrieved context (top 2 of 5):**
```
Great Papers 12 The Dirac Equation…  0:00–1:49   (cosine 0.6858)
  "Imagine you set out to do something modest and technical. You want to fix
   a known flaw in an equation, not overthrow physics…"

Great Papers 09 Bell's Theorem, 1964  9:45–11:29  (cosine 0.6526)
  "…the light flashes a colour. You do this over and over, millions of times,
   with the switches set however you like…"
```

**What happened:** max cosine similarity 0.686, far above the 0.30 gate, so the
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

**Question:** "What does the Dirac episode say about the Higgs boson giving
particles their mass?"

**Retrieved context (top 1):**
```
Great Papers 12 The Dirac Equation…  20:37–22:01  (cosine 0.6905)
  "…rare radioactive process that would only be possible if the neutrino is
   its own antiparticle, and the answer would bear directly on that matter
   versus antimatter mystery…"
```

**What happened:** cosine 0.691 — the *highest* score of any refusal case, and
higher than legitimately answerable questions such as `followup_03` (0.596)
and `cross_01` (0.652).

**Why it failed:** the question names a real episode from the collection and a
real concept from physics (the Higgs) that the episode never discusses. Every
lexical and semantic signal says "this is about the Dirac episode." Only
reading the passages reveals the Higgs is absent.

**Root cause:** the same as Failure 1 but sharper — this case proves no single
threshold on bi-encoder cosine can work, because the score distributions of
in-scope and out-of-scope questions genuinely *overlap*:

```
in-scope     cosine range:  0.596 – 0.778
out-of-scope cosine range:  0.439 – 0.691      ← overlapping
```

Any threshold that catches the Higgs question at 0.691 would also refuse
`followup_03` (0.596), `cross_01` (0.652) and several other legitimate
questions.

### Failure 3 — `cross_02`: one episode monopolises a comparison

**Question:** "Compare how these episodes use E = mc squared."

**Retrieved context:**
```
Great Papers 01 Einstein's Special Relativity  22:36–23:55  (cosine 0.6517)
  "…to cross in its normal lifetime. So one observer says your clock slowed,
   the other says your distance shrank…"
Great Papers 09 Bell's Theorem, 1964           9:45–11:29   (cosine 0.6485)
```

**What happened:** episode hit 50% — the Dirac episode, which is half the
question, never appeared. Term coverage 0%: neither "mc squared" nor
"annihilate" was in the retrieved text, even though both episodes discuss
E = mc² explicitly.

**Why it failed:** dense-only ranking took the five globally best-scoring
passages. Relativity passages dominate anything phrased in Einstein's
vocabulary, and the Bell episode contributed a passage that is not about mass
at all. Nothing in the strategy guarantees that a comparison question sees both
sides.

**Root cause:** a different problem from the first two — this is candidate
*selection*, not relevance *judgement*. A comparison needs per-episode
representation, which a single global ranking cannot promise.

---

## 6. The improvement

Failures 1 and 2 are the same root cause and account for **4 of the 6 baseline
failures**, in the category the product's core promise depends on. That is the
one thing worth fixing. (Failure 3 is fixed separately and more cheaply, by
balancing retrieval across episodes in `compare` mode.)

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
   the light clock" is a request wrapped around a topic; scored verbatim the
   cross-encoder grades the wrapper and returns **−1.18**, but scored as "the
   light clock" it returns **+3.72**. Likewise "where in the audio do they talk
   about closing the experimental loopholes" goes from **+0.06** to **+4.36**.
3. Gate on the reranker score instead of cosine.

### 6.1 The threshold — and how I avoided fitting it to the test set

The obvious threshold is the cross-encoder's natural decision boundary (logit
sign, i.e. 0.0). During development on the fixture corpus I tried exactly that
and **it made things worse**: refusal accuracy went 0% → 100% but it introduced
**6 false refusals** and the overall pass rate *fell*. That experiment is kept
as evidence in `eval/results/synthetic_rerank_only/` — it is a fixture-corpus
run, and it is quoted here for the lesson, not for its numbers.

The lesson generalises, and the real corpus shows why. Collection-level
questions, numeric-value questions and un-rewritten follow-ups all score low on
a *passage*-relevance model, because no single passage "answers" them in the
sense the model was trained on. On the Fermi corpus the cross-encoder scores
still overlap:

```
in-scope     rerank range:  −6.92 – +5.45   (min: followup_03, un-rewritten)
out-of-scope rerank range: −11.02 – +0.95   (max: refusal_04, the Higgs trap)
```

A threshold placed at 0.0 would refuse eight legitimate questions to catch one
more out-of-scope one. So the threshold is not tuned on the evaluation cases at
all. It is **calibrated from the corpus at ingestion time**
(`retrieve/calibrate.py`): eight fixed, mundane, domain-free probe questions
that no podcast collection could answer ("how do I renew my passport online",
"what are the rules of cricket") are scored against the indexed passages. Their
scores describe what "definitely not covered" looks like *for this corpus*, and
the gate sits one point above that band.

On the Fermi corpus the probe band tops out at **−6.421**, giving a gate of
**−5.421**, written into `data/index/index_meta.json`. It is recomputed on
every `make ingest`, so it adapts to whatever audio is supplied — on the
fixture corpus the same code produced −9.98.

This makes the gate deliberately **high-precision and low-recall**: it refuses
the clearly unrelated and never refuses something the episodes might support.
Topically adjacent questions are passed to the grounding prompt, which can read
the passages and decide. The two layers have different jobs — the gate is a
cheap pre-filter, the prompt is the semantic judge.

---

## 7. Baseline vs improved

`make eval-compare`

```
             fermi_baseline  →  fermi_improved
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                   ┃ Baseline ┃ Improved ┃  Delta ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ Pass rate                │    70.0% │    90.0% │ +20.0% │
│ Retrieval: episode hit   │    93.8% │   100.0% │  +6.2% │
│ Retrieval: term coverage │    84.4% │    84.4% │      0 │
│ Refusal accuracy         │     0.0% │    75.0% │ +75.0% │
│ False refusals           │        0 │        1 │     +1 │
│ Mean latency             │    0.01s │    0.52s │ +0.51s │
│ Estimated cost           │  $0.0000 │  $0.0000 │      0 │
└──────────────────────────┴──────────┴──────────┴────────┘

Fixed (5): cross_02, cross_03, refusal_01, refusal_02, refusal_03
Regressed (1): followup_03
```

| Category | Baseline | Improved |
|---|---|---|
| factual | 4/4 | 4/4 |
| cross_episode | 1/3 | **3/3** |
| explanation | 2/2 | 2/2 |
| locate | 2/2 | 2/2 |
| recommendation | 1/1 | 1/1 |
| ambiguous | 1/1 | 1/1 |
| followup | 3/3 | 2/3 |
| **refusal** | **0/4** | **3/4** |

### What improved
Three of four out-of-scope questions are now refused deterministically, before
any model call — so those refusals are free, fast, and cannot be talked out of
by a persuasive prompt. Cross-episode retrieval went 1/3 → 3/3: episode-balanced
selection in `compare` mode guarantees both sides of a comparison are present,
which lifted mean episode hit from 93.8% to 100%.

### What did not
`refusal_04` (the Higgs trap, rerank **+0.95**) still clears the gate. This is
by design, not an oversight: it is the highest-scoring out-of-scope question in
the set, and pulling the gate up to catch it would cross deep into the in-scope
band and start refusing real questions. It is delegated to the grounding
prompt — which offline mode cannot exercise, but which handles it correctly in
the full run (§9, `refusal_04` passes 5/5 on every judge dimension).

### What regressed
**`followup_03` regressed, and I have left it in the table rather than hiding
it.** The case is a two-turn conversation whose scored turn is "How long did it
take before someone actually found it?" — a question with no topical content of
its own. Offline mode makes no LLM call by design, so no query rewriting
happens, the bare turn scores −6.92, and the gate refuses it. This is a false
refusal and it counts against the improved run.

It is an artefact of the *offline measurement mode*, not of the shipped system:
in the full run, rewriting turns it into "How long after Dirac's equation
predicted the positron did experimental confirmation arrive?" and the case
passes 5/5. The honest statement is that the improved gate is strictly better
at refusing, and slightly more brittle for context-dependent follow-ups when it
has no rewriter — which is exactly the trade-off a cheap pre-filter makes.

Mean latency rose 0.01 s → 0.52 s. Almost all of that is the one-off
cross-encoder load on the first query; steady-state reranking is ~35 ms for 8
passages, which is negligible next to an LLM call.

### Honest reading of the headline number
The 90% figure covers only the retrieval and gating layer — it is not a claim
about answer quality. The full pipeline, measured separately in §9, scores
19/20 with refusal accuracy 4/4 and citation validity 100%.

---

## 8. Remaining weaknesses

1. ~~Adjacent-topic refusals rely entirely on the prompt.~~ **Resolved and
   measured** — the full run in §9 scores refusal accuracy 4/4, confirming the
   prompt catches what the gate delegates (`refusal_04`, 5/5 on every judge
   dimension).
2. ~~Follow-up retrieval degrades without query rewriting.~~ **Resolved with a
   key** — `followup_03` passes in the full run. It still fails offline, so a
   free fallback (concatenating the previous question with the follow-up)
   remains worthwhile for keyless operation.
3. **Unsupported narrative completion.** `cross_01` in §9.2 — the model added a
   word ("instantaneously") and a characterisation the passages do not contain.
   No deterministic check catches this class, and the strict grounding prompt
   did not prevent it. This is the biggest open risk in the system.
4. **ASR error becomes retrieval error on proper nouns.** Whisper `medium`
   transcribes "Michelson-Morley" as **"Mickelson-Morley"** throughout the
   relativity episode. Querying the correct spelling still returns the right
   passage first — the hyphen tokenises and "Morley" still matches, at BM25
   4.20 against 8.40 for the ASR's own spelling — but the margin is halved, and
   a name with no second token would be lost entirely. This class of failure
   is invisible on synthetic audio and only appears on real speech.
5. **The corpus is small** (92 chunks, 1 h 59 m). Retrieval metrics are
   optimistic at this scale; `episode_hit` of 100% is much easier with three
   episodes than thirty.
6. **The judge is unvalidated.** I have not measured judge–human agreement, so
   its scores should be read as a signal, not a verdict — though on the cases
   in §9.1 and §9.2 its rationales were specific and correct on inspection.
7. **Full-pipeline numbers are one sample.** Generation is non-deterministic.
   Across the two full runs recorded here the pass rate was 19/20 both times,
   but the *failing case* differed (`explain_02` before the prompt fix in §9.1,
   `cross_01` after). The offline numbers are deterministic; the full numbers
   are not, and a single run should not be over-read.
8. **`term_coverage` is a proxy.** It rewards lexical overlap with the terms I
   chose, which under-credits a passage that conveys the idea in other words.

---

## 9. Full pipeline results (generation + LLM judge)

`eval/results/fermi_full/` — run through OpenRouter with
`anthropic/claude-sonnet-5` for chat and `openai/gpt-5-mini` as judge. The judge
is a **different vendor from the chat model on purpose**: a judge grading its
own family's output is prone to self-preference bias, and OpenRouter makes
cross-vendor judging a one-line change on a single key.

```
Pass rate                19/20   (95%)
Refusal accuracy           4/4  (100%)
False refusals                      0
Citation validity                100%   (0 invalid of 62)
Forbidden-claim leaks               0
Retrieval: episode hit           100%
Retrieval: term coverage          91%
Judge: faithfulness              4.90
Judge: completeness              4.80
Judge: citation correctness      5.00
Judge: refusal correctness       5.00
Judge: clarity                   5.00
Mean latency                   11.25s
Actual cost                   $0.2995   (reported by OpenRouter, not estimated)
```

Per category: factual 4/4 · cross_episode 2/3 · explanation 2/2 ·
recommendation 1/1 · locate 2/2 · **refusal 4/4** · ambiguous 1/1 ·
followup 3/3.

Three things are worth drawing out.

**The two-layer refusal design is validated.** Offline, the calibrated gate
catches 3 of 4 refusals and deliberately delegates the topically adjacent one.
With generation enabled, refusal accuracy is **4/4** — the grounding prompt
caught exactly the case the gate passed to it. On the Higgs trap the model
answered by *naming what the episode does cover* (negative-energy solutions,
the Dirac Sea, the positron, PET scans) and stating the Higgs is not among it.
That is the design working as intended.

**Citation validity is 100% across 62 citations.** No fabricated episode, no
invented timestamp, nothing stripped by the validator. Re-verified
independently against the ASR segment table — every cited interval overlaps
real transcribed speech in the episode it names, and lies inside that episode's
duration. The 62 citations span all three episodes (23 / 13 / 21).

**Query rewriting fixes the offline regression.** `followup_03` — the
un-anchored follow-up that fails offline — passes here. "How long did it take
before someone actually found it?" was rewritten to "How long after Dirac's
equation predicted the positron did experimental confirmation arrive?" and
retrieval found the right passage.

### 9.1 Two defects the evaluation found — in the evaluation and in the product

Two defects were found by running this evaluation and are worth recording,
because finding them is the entire point of building one.

**A false refusal caused by an ambiguous prompt rule.** The first full run on
the real corpus scored 19/20 with `explain_02` failing — "What was the Dirac
Sea and why don't physicists use it any more?" was flagged `NOT_COVERED`
*despite the model having answered the first half correctly and completely*.
The judge scored it 5/5/5/5/5; only the deterministic check caught it, flagging
an in-scope question refused. The cause was a genuine conflict in the system
prompt: rule 3 said to emit the marker "if the passages do not support an
answer", while rule 4 said to answer partially-supported questions and name the
gap. For a two-part question where retrieval surfaced the "what" but not the
"why", both rules applied. Fixed by scoping rule 3 to "support **no part** of
the question" and stating explicitly that rule 4's case must not emit the
marker. That run is kept at `eval/results/fermi_full_pre_prompt_fix/`; after the
fix `explain_02` passes 5/5 and false refusals went 1 → 0.

Note what this was *not*: the retrieval gate behaved correctly throughout
(rerank −0.55, well above the −5.421 gate). The passage answering "why it was
replaced" — Feynman's reading of antiparticles as particles moving backward in
time — sits in chunk `…_0012`, which ranked below the chunks that were
returned. A retrieval miss and a prompt ambiguity combined to discard an answer
the system was capable of giving.

**Two earlier defects, found on the fixture corpus, still fixed in this code.**
A forbidden-claim check that failed a *correct* refusal for containing a word
the learner had put in the question ("Hawking"), and a canned refusal message
that asserted *"I searched the transcripts of every episode"* — true of the
system, but not supported by the retrieved passages, and correctly docked by
the judge. Both fixes are in the shipped code; the evidence run is
`eval/results/synthetic_full_pre_fixes/`.

### 9.2 The one real remaining failure

`cross_01` fails on faithfulness (3/5), and this one is genuine.

Asked how the relativity and Bell episodes each treat the idea that nothing
outruns light, the answer says that reproducing quantum mechanics forces any
hidden-variable theory to include influence that reaches across distance
**"instantaneously"**, and calls the two positions **"seemingly conflicting"**.
The retrieved Bell passages say the pilot-wave picture is "openly explicitly
non-local with a guiding wave that reaches across any distance" — they do not
say *instantaneously*, and they do not characterise the relationship to
relativity at all.

Both additions are true, and both are the kind of thing a knowledgeable reader
would supply automatically — which is precisely why they are dangerous. The
model completed the argument rather than reporting it. Every deterministic
check passed: no forbidden claim appeared, all four citations were valid, and
the right episodes were retrieved. Only the judge caught it.

There is a compounding retrieval cause. The Bell episode *does* address this
directly — "So no information outraces light… Relativity's speed limit is
completely safe" — and had that passage been retrieved, the answer would have
been both grounded and better. Compare mode balanced across episodes but
selected the entanglement setup and the interpretations passage instead. The
model, lacking the passage that resolves the tension, inferred the tension.

This is the exact failure mode the product exists to prevent, it survived a
strict grounding prompt, and it was caught only by the judge. It is the
strongest argument in this document for having an LLM judge at all, and the
most useful open problem in the system.

## 10. Next improvements, in priority order

1. **Attack unsupported narrative completion** (§9.2) — the one real open
   failure. A verification pass that re-reads each claim against the cited
   passage, or a prompt rule specifically forbidding inferences the transcript
   only implies, is the obvious next experiment.
2. **Improve within-episode passage selection for multi-part questions.** Both
   §9.1 and §9.2 have the same secondary cause: the passage that answers the
   *second* half of a question ranked below passages answering the first. A
   query-decomposition step, or reserving a context slot for the lowest-scoring
   sub-question, would address both.
3. **Free follow-up query expansion** — prepend the previous turn's question to
   a context-dependent follow-up when no LLM is available, so keyless operation
   matches the full pipeline on `followup_03`.
4. **Repeat the full run to quantify variance** (§8.7) — two runs is not enough
   to report a confidence interval on a non-deterministic pass rate.
5. **Broaden the judge across vendors** — run the same cases through two judges
   from different families and report disagreement, which would also give the
   judge validation §8.6 asks for. OpenRouter makes this a config change.
6. **Fuzzy lexical matching for ASR-mangled proper nouns** (§8.4) — a character
   n-gram fallback in BM25 would recover "Michelson" → "Mickelson" without
   touching dense retrieval.
7. **Sentence-level citation anchoring** — cite the specific sentence rather
   than the whole 75-second chunk, narrowing what a learner must listen to.
