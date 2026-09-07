# Demo Script — 3 minutes

Two terminals: one for the product, one for the evaluation.
`make ingest` should already have been run.

---

### 0 · Setup (10s) — say this over the repo

> "Raw MP3 in, own ASR, hybrid retrieval, grounded answers with timestamps you
> can click. One rule: if the audio doesn't support it, it says so."

```bash
make episodes
```
Shows the ingested collection with durations and passage counts.

---

### 1 · Ingestion is real and idempotent (20s)

```bash
make ingest
```
> "Re-running is a no-op — it's keyed on the SHA-256 of each file, so it reuses
> the transcripts. The first run transcribed 20 minutes of audio in about 7."

Point at `data/transcripts/01_the_birth_of_the_quantum.txt`:
> "That's our own Whisper output with timestamps. No external captions
> anywhere in this system."

---

### 2 · A grounded factual answer (25s)

```bash
make chat
```
```
Why did Planck introduce the quantum idea?
```
> "Answer first, then inline citations, then sources with timestamps."

---

### 3 · A multi-turn follow-up (20s)

```
Explain that act of desperation part more simply.
```
> "'That' resolves from the conversation — it's rewritten into a standalone
> search query before retrieval, not just pasted in."

---

### 4 · Cross-episode comparison (25s)

```
How do the Shannon and Bell episodes differ in what they mean by uncertainty?
```
> "Comparison mode balances retrieval across episodes, so one strong match
> can't crowd out the other side. Note the sources span two episodes."

---

### 5 · Take me to the audio (20s)

```
Take me to the part where they explain what a bit is.
```
> "Episode plus timestamp."

Then switch to the web UI:
```bash
make web        # http://127.0.0.1:8000
```
Ask the same thing, **click a source** — the player seeks to that second.
> "This is the verification loop: ask, understand, verify, listen."

---

### 6 · Honest refusal — the important one (30s)

```
What do these episodes say about how superconductivity works?
```
> "Not covered. And notice it was instant — no model call was made. A
> calibrated relevance gate refused before spending a token."

Then the harder one:
```
What did Planck say about the double-slit experiment in this episode?
```
> "This one is nastier — it names a real thing from the episode and a real
> physics concept the episode never mentions. The model knows the answer. It
> still says the episodes don't cover it."

---

### 7 · Evaluation (35s)

Second terminal:
```bash
make eval-compare
```
> "20 cases, 8 categories, run through the real agent — no evaluation-only code
> path. Baseline was naive RAG: dense retrieval, cosine threshold for refusal.
> 75% pass, and refusal accuracy zero out of four."

> "I inspected the failures. The problem was that cosine similarity measures
> whether things are about the same topic, not whether a passage answers the
> question — so 'what did Planck say about the double slit' scored *higher*
> than legitimate questions. The distributions genuinely overlap, so no
> threshold could work."

> "The fix: a cross-encoder that reads query and passage together, with the
> threshold calibrated from the corpus itself at ingestion time rather than
> tuned on these cases. 75% to 85%, refusal 0% to 50%, zero false refusals."

> "My first attempt at this made it *worse* — 6 false refusals. That run is
> still in `eval/results/run_rerank_only/`, and the analysis is in EVAL.md."

---

### 8 · Close (15s)

> "113 tests. Everything except generation runs locally and free — ASR,
> embeddings, reranking, vector store, and the whole offline evaluation.
> Honest gaps are in EVAL.md section 8: the two adjacent-topic refusals still
> rely on the prompt rather than the gate, and I haven't yet run the full
> judged evaluation."
