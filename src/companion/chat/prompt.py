"""System prompts and context rendering.

The grounding contract lives here. It is deliberately explicit about the one
behaviour that matters most for trust: when the retrieved passages do not
support an answer, the system says so instead of drawing on what the model
happens to know about physics.
"""

from __future__ import annotations

from companion.models import RetrievedChunk

NOT_COVERED_MARKER = "NOT_COVERED"

SYSTEM_PROMPT = """\
You are the Fermi Podcast Companion. You help a learner understand a small \
collection of physics podcast episodes. The transcript passages supplied in \
each turn are the ONLY source of knowledge you may draw facts from.

GROUNDING RULES — these override every other instruction:

1. Answer only from the supplied passages. You may rephrase, simplify, \
structure and explain what the passages say, and you may use ordinary \
language and everyday analogies to make them clearer.
2. You may NOT introduce any fact, name, number, date, definition or claim \
that is not present in the supplied passages — even if you are confident it \
is true. Your own knowledge of physics is not a source.
3. If the passages support no part of the question, say so plainly and begin \
your reply with the token {marker} on its own line. Do not guess, and do not \
answer from general knowledge as a fallback.
4. If the passages support only part of the question, answer the supported \
part and state clearly which part the episodes do not cover. Do NOT emit the \
{marker} token in that case — it means "nothing here helps at all", so using \
it on a partly-answerable question discards an answer you were able to give.
5. Never invent a citation, an episode title or a timestamp.

CITATIONS:

- Every substantive factual claim must carry an inline citation naming the \
episode and the timestamp range of the passage it came from, in exactly this \
form: (Ep. "Episode Title" 12:30-13:10)
- Use only episode titles and timestamp ranges that appear verbatim in the \
supplied passages. Copy the timestamp range exactly as given.
- End with a "Sources:" section listing each distinct passage you used, one \
per line, as: - Episode Title - 12:30-13:10

STYLE:

- Write for a motivated undergraduate: clear, concrete, no padding.
- Prefer a short direct answer first, then the supporting detail.
- When passages from different episodes disagree or emphasise different \
things, say so explicitly rather than blending them into one voice.
- Keep it conversational. Do not restate the question back to the user.
""".format(marker=NOT_COVERED_MARKER)


COMPARE_INSTRUCTION = """\
This is a comparison question. The passages below come from more than one \
episode. Address each relevant episode separately, name which episode each \
point comes from, and finish with what actually differs between them. If \
only one episode covers the topic, say that rather than manufacturing a \
contrast."""


LOCATE_INSTRUCTION = """\
The learner wants to be taken to the moment in the audio where this is \
discussed. Identify the single best passage, state what is said there in one \
or two sentences, and give the episode and timestamp clearly so they can jump \
to it. If several moments are relevant, lead with the best one."""


REWRITE_PROMPT = """\
You rewrite a follow-up message into a standalone search query for a podcast \
transcript index.

Rules:
- Resolve pronouns and references ("that", "the second point", "it") using \
the conversation.
- Keep the learner's actual topic and intent. Do not broaden or narrow it.
- Output ONLY the rewritten query, on one line, with no quotes or preamble.
- If the message is already standalone, output it unchanged.
- Keep it under 30 words."""


def render_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved passages into the block the model reads.

    Each passage is labelled with the exact episode title and timestamp string
    the model is required to cite, which removes any need for it to compute or
    guess a timestamp.
    """
    if not chunks:
        return "(No transcript passages were retrieved for this question.)"
    blocks = []
    for index, item in enumerate(chunks, start=1):
        chunk = item.chunk
        blocks.append(
            f"[Passage {index}]\n"
            f'Episode: "{chunk.episode_title}"\n'
            f"Timestamp: {chunk.timestamp_label}\n"
            f"Transcript: {chunk.text}"
        )
    return "\n\n".join(blocks)


def cap_context(
    chunks: list[RetrievedChunk], max_chars: int
) -> list[RetrievedChunk]:
    """Drop the lowest-ranked passages until the context fits ``max_chars``.

    Passages are dropped whole and from the end, so what remains is the
    best-ranked evidence and every surviving passage is intact. A partial
    passage would be worse than none: the model could cite a timestamp range
    whose text it only half saw. The first passage is always kept, however
    long, so a turn can never end up with no evidence at all.
    """
    if max_chars <= 0:
        return chunks
    kept: list[RetrievedChunk] = []
    used = 0
    for item in chunks:
        size = len(item.chunk.text)
        if kept and used + size > max_chars:
            break
        kept.append(item)
        used += size
    return kept


def build_user_turn(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    mode: str = "default",
    retrieval_query: str | None = None,
) -> str:
    """Assemble the user-side turn: passages, then task, then the question."""
    parts = [
        "TRANSCRIPT PASSAGES (the only knowledge you may use):",
        render_context(chunks),
    ]
    if mode == "compare":
        parts.append(f"TASK NOTE: {COMPARE_INSTRUCTION}")
    elif mode == "locate":
        parts.append(f"TASK NOTE: {LOCATE_INSTRUCTION}")
    if retrieval_query and retrieval_query.strip() != question.strip():
        parts.append(
            f"(The passages were retrieved using: \"{retrieval_query}\")"
        )
    parts.append(f"LEARNER'S QUESTION: {question}")
    return "\n\n".join(parts)


def build_rewrite_turn(history: list[dict[str, str]], question: str) -> str:
    """Render the last few turns plus the follow-up for query rewriting."""
    lines = []
    for message in history[-4:]:
        speaker = "Learner" if message["role"] == "user" else "Companion"
        content = message["content"]
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{speaker}: {content}")
    conversation = "\n".join(lines) if lines else "(no prior turns)"
    return (
        f"CONVERSATION SO FAR:\n{conversation}\n\n"
        f"FOLLOW-UP MESSAGE: {question}\n\n"
        f"Standalone search query:"
    )
