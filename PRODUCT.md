# Product Note — Fermi Podcast Companion

## The user

A motivated self-learner — typically a STEM undergraduate — who listens to
long-form educational podcasts and wants to actually *learn* from them, not
just consume them. They are curious, willing to think, and short on time.

## The problem I chose to solve

Long-form audio is the worst possible medium for the thing people most want to
do with it: come back to an idea and check it.

Concretely, after listening to three hour-long episodes, a learner cannot:

- **find** the moment where something was explained ("they said something about
  why energy comes in packets — where?");
- **compare** what two episodes said about the same idea;
- **verify** a half-remembered claim without scrubbing through an hour of audio;
- **tell** whether the collection covers a topic at all.

Search does not fix this, because a learner does not remember the words that
were spoken — they remember the *idea*. And a plain chatbot does not fix it
either, because a language model already knows a great deal of physics and will
happily answer from that knowledge, leaving the learner unable to tell what the
podcast actually said from what the model filled in.

That last point is the one I built the product around.

## The product

A conversational companion over a small podcast collection, where **every
answer is traceable to a moment in the audio**.

A learner can:

- ask questions in natural language and get a direct, clear answer;
- follow up conversationally ("explain that second point more simply");
- ask across episodes ("how do these two differ on what uncertainty means?");
- ask what to listen to and why;
- say "take me to the part where they explain X" and get a timestamp;
- click any source and hear the audio at that exact second;
- and — critically — be told plainly when the episodes simply do not cover
  something.

## The defining principle

> If the supplied audio does not support an answer, the system says so
> instead of guessing.

This is the feature. A companion that is right 90% of the time and confident
100% of the time is worse than useless for learning, because the learner cannot
tell the two apart. Everything else in the system — the grounding prompt, the
citation validator that deletes citations it cannot verify, the calibrated
relevance gate that refuses before the model is even called — exists to make
that principle enforceable rather than aspirational.

The measurable version of this principle is in `EVAL.md`: out-of-scope
questions are a first-class evaluation category, and "answered a question the
episodes don't cover" is a hard failure regardless of how good the answer looks.

## Why this framing, and not another

Two alternatives I considered and rejected:

- **A study-notes generator** (summaries, flashcards, key-point extraction).
  Genuinely useful, but the brief explicitly rules out "a one-time summary",
  and one-shot artefacts do not let a learner chase the specific thing they are
  confused about.
- **A semantic search engine over transcripts.** Honest and verifiable, but it
  puts all the synthesis work back on the learner. The value of a language
  model here is explaining a hard idea *simply* — which is exactly what the
  learner wanted from the podcast in the first place.

The companion sits between them: it explains like a tutor, but every claim
carries a receipt.

## Success criteria

The product works if a learner can go **Ask → Understand → Verify → Listen**
without leaving the conversation, and if they can trust a "no" as much as a
"yes".

## Non-goals

Deliberately out of scope, in line with the brief:

- deployment, authentication, billing, multi-user infrastructure;
- a large or continuously changing catalogue (this is built for a handful of
  episodes and says so);
- training any speech or language model from scratch;
- a polished frontend — the CLI is the primary interface and the web UI exists
  only because clicking a citation to hear the audio is a genuinely better way
  to verify an answer;
- TTS, voice cloning, audio editing, or podcast generation;
- streaming infrastructure — audio is served as a local file with byte-range
  seeking, which is all the "jump to timestamp" feature needs.
