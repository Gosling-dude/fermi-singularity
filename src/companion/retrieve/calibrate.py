"""Calibrating the refusal threshold from the corpus.

Picking a relevance threshold by looking at the evaluation cases would fit the
gate to the test set. Instead the threshold is derived from the corpus itself
at ingestion time.

A fixed set of probe questions that no podcast collection could plausibly
answer is scored against the indexed passages. Their scores describe what
"definitely not covered" looks like *for this corpus and this reranker*. The
gate sits just above that band, which makes it high-precision: it refuses the
clearly unrelated and never refuses a question the episodes might support.

Topically adjacent questions — ones that share vocabulary with the episodes but
are not actually discussed — score well above this band. Those are deliberately
left to the grounding prompt, which can read the passages and decide. The two
layers have different jobs: the gate is a cheap pre-filter, the prompt is the
semantic judge.
"""

from __future__ import annotations

from companion.models import Chunk
from companion.utils.logging import get_logger

log = get_logger("ingest")

# Deliberately mundane and domain-free. These are fixed, ship with the system,
# and are unrelated to the evaluation cases.
NEGATIVE_PROBES = [
    "how do I bake sourdough bread at home",
    "what are the rules of cricket",
    "how do I renew my passport online",
    "best budget laptop for students this year",
    "how to train for a marathon in twelve weeks",
    "recipe for chicken curry with coconut milk",
    "how do I change a flat bicycle tyre",
    "what time does the supermarket close on sunday",
]

# How far above the not-covered band the gate sits. Larger is more
# conservative (fewer refusals, more deferred to the grounding prompt).
DEFAULT_MARGIN = 1.0

# Used when calibration cannot run (no reranker available).
FALLBACK_GATE = -10.0


def calibrate_relevance_gate(
    chunks: list[Chunk], reranker, margin: float = DEFAULT_MARGIN
) -> float | None:
    """Return the relevance score below which a question is 'not covered'.

    ``None`` means calibration could not run and the caller should fall back
    to the bi-encoder threshold.
    """
    if reranker is None or not chunks:
        return None
    texts = [chunk.text for chunk in chunks]
    worst = float("-inf")
    for probe in NEGATIVE_PROBES:
        scores = reranker._model.predict([(probe, text) for text in texts])
        worst = max(worst, float(max(scores)))
    gate = worst + margin
    log.info(
        "calibrated relevance gate",
        probes=len(NEGATIVE_PROBES),
        not_covered_band_max=round(worst, 3),
        gate=round(gate, 3),
    )
    return round(gate, 4)
