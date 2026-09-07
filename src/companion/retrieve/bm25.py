"""Lexical retrieval over chunk text.

BM25 is the half of the hybrid retriever that handles exact terminology —
names, units, coined phrases — where an embedding model happily returns
something topically adjacent but lexically wrong.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from companion.models import Chunk

# Common English words plus discourse fillers that dominate spoken transcripts
# and would otherwise inflate BM25 scores for conversational chunks.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with as is are was were be been being it its into about over under
    we you they he she i me my our your their them us who whom which what when
    where how why not no nor so such can could would should will shall may
    might must do does did done have has had having there here just really
    actually basically kind sort like yeah okay um uh right well now thing
    things get got go going one two also very much more most some any because
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords and 1-character noise removed."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


@dataclass
class BM25Index:
    """A BM25Okapi index plus the chunk ids its documents correspond to."""

    chunk_ids: list[str]
    _model: object

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("cannot build a BM25 index from zero chunks")
        corpus = [tokenize(chunk.text) for chunk in chunks]
        # A document that tokenises to nothing breaks BM25's averaging; keep a
        # placeholder token so index positions stay aligned with chunk_ids.
        corpus = [tokens or ["__empty__"] for tokens in corpus]
        return cls(chunk_ids=[chunk.chunk_id for chunk in chunks],
                   _model=BM25Okapi(corpus))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` for the best ``top_k`` matches.

        Zero-scoring documents are dropped: a lexical index that matched
        nothing should contribute nothing to fusion rather than pad the
        candidate list with noise.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._model.get_scores(tokens)
        ranked = sorted(
            zip(self.chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0.0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"chunk_ids": self.chunk_ids, "model": self._model}, handle)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(chunk_ids=payload["chunk_ids"], _model=payload["model"])
