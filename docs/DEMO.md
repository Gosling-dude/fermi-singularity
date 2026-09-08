# Demo Script — 3 minutes

Two terminals: one for the product, one for the evaluation.
`make ingest` should already have been run.

The corpus is the three supplied Fermi episodes — Einstein's Special
Relativity (52:14), Bell's Theorem (34:00) and The Dirac Equation and
Antimatter (33:00). 1 h 59 m of audio, transcribed by this system.

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
> the transcripts. The first run transcribed two hours of audio in 41 minutes,
> about 3× realtime on CPU."

Point at `data/transcripts/great_papers_01_einstein_s_special_relativity.txt`:
> "That's our own Whisper output with timestamps — 594 segments for this
> episode. No external captions anywhere in this system."

---

### 2 · A grounded factual answer (25s)

```bash
make chat
```
```
What are Einstein's two postulates?
```
> "Answer first, then inline citations, then sources with timestamps. Six
> citations here, every one of them pointing at a real interval of speech you
> can go and listen to."

---

### 3 · A multi-turn follow-up (20s)

```
Why does that second one break common sense?
```
> "'That second one' resolves from the conversation — it's rewritten into a
> standalone search query before retrieval, not just pasted in. Watch the
> retrieval_query in the logs."

---

### 4 · Cross-episode comparison (25s)

```
Compare how each of these episodes portrays Einstein.
```
> "Comparison mode balances retrieval across episodes, so one strong match
> can't crowd out the other side. Einstein is the triumphant architect in the
> relativity episode and the man who turned out to be wrong in the Bell
> episode — and the sources span both."

---

### 5 · Take me to the audio (20s)

```
Where in the audio do they talk about muons?
```
> "Episode plus timestamp — around 20:37 in the relativity episode, where the
> sky is running the time-dilation experiment for free."

Then switch to the web UI:
```bash
make web        # http://127.0.0.1:8000
```
Ask the same thing, **click a source** — the player seeks to that second.
> "This is the verification loop: ask, understand, verify, listen."

---

### 6 · Honest refusal — the important one (30s)

```
What do these episodes say about dark matter?
```
> "Not covered. And notice it was instant — no model call was made. A
> calibrated relevance gate refused before spending a token."

Then the harder one:
```
What does the relativity episode say about black hole event horizons?
```
> "This one is nastier — it names a real episode from the collection and a real
> physics concept the episode never mentions. The model knows the answer. The
> gate deliberately does *not* fire here, because the question is topically
> adjacent; the grounding prompt reads the passages and refuses instead —
> listing what the episode *does* cover. Two layers, different jobs."

---

### 7 · Evaluation (35s)

Second terminal:
```bash
make eval-compare
```
> "20 cases, 8 categories, run through the real agent — no evaluation-only code
> path. Baseline was naive RAG: dense retrieval, cosine threshold for refusal.
> 70% pass, and refusal accuracy zero out of four."

> "I inspected the failures. The problem was that cosine similarity measures
> whether things are about the same topic, not whether a passage answers the
> question. The distributions genuinely overlap, so no threshold could work."

> "The fix: a cross-encoder that reads query and passage together, with the
> threshold calibrated from the corpus itself at ingestion time rather than
> tuned on these cases. 70% to 90%, refusal 0% to 75%. It cost one regression,
> and I left it in the table rather than hiding it — `followup_03` needs query
> rewriting to have a topical anchor, and the offline mode makes no LLM call."

Then, optionally, show the full run:
```bash
python -m eval.report --show fermi_full | head -30
```
> "That's the offline layer. With generation and an LLM judge on top —
> different vendor from the chat model, so it isn't grading its own family —
> the numbers are in EVAL.md §9, including 100% citation validity."

> "The `synthetic_*` runs in `eval/results/` are from the development fixture
> corpus, before the real audio arrived. They're kept for provenance and
> clearly separated — in particular `synthetic_rerank_only`, a change that made
> things worse and got reverted."

---

### 8 · Close (15s)

> "Everything except generation runs locally and free — ASR, embeddings,
> reranking, vector store, and the whole offline evaluation. One key,
> OpenRouter, for chat and the judge."

> "And the honest gaps are written down in EVAL.md §8 — including one the real
> audio exposed that the synthetic corpus never could: Whisper hears
> 'Michelson-Morley' as 'Mickelson-Morley'. Ask for the correct spelling and
> you still get the right passage first, because the hyphen splits into tokens
> and 'Morley' still matches — but at half the lexical score. Proper nouns are
> where ASR quality quietly becomes retrieval quality, and you only find that
> out on real speech."
