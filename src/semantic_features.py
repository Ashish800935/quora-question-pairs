"""
Optional semantic-similarity feature using Sentence-Transformers.

This is the one deliberate "differentiator" feature: it reuses the same
Sentence-Transformers skill already applied in the Hybrid-RAG project,
so it isn't a brand-new, unproven library on the resume — it's the same
tool used a second time for a genuinely different purpose (dense
semantic similarity instead of pure lexical/statistical overlap).

It is wrapped in a try/except and a `SEMANTIC_AVAILABLE` flag so the
rest of the pipeline (train.py, predict.py) can run perfectly fine
without it if the model hasn't been downloaded / the package isn't
installed — the BOW + TF-IDF + engineered-feature pipeline is fully
self-sufficient on its own.
"""
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for short questions
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def semantic_similarity_batch(questions1, questions2) -> np.ndarray:
    """Cosine similarity between sentence embeddings of q1 and q2,
    for a batch of question pairs. Returns an (n,) array of floats
    in [0, 1] (roughly; cosine similarity of MiniLM embeddings)."""
    if not SEMANTIC_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run `pip install sentence-transformers` to enable this feature, "
            "or skip it (train.py --no-semantic)."
        )
    model = _get_model()
    emb1 = model.encode(list(questions1), show_progress_bar=False, batch_size=64)
    emb2 = model.encode(list(questions2), show_progress_bar=False, batch_size=64)
    sims = np.array([
        cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
        for a, b in zip(emb1, emb2)
    ])
    return sims


def semantic_similarity_pair(q1: str, q2: str) -> float:
    return float(semantic_similarity_batch([q1], [q2])[0])
