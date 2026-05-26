import numpy as np
from sentence_transformers import SentenceTransformer, util


def compute_anchored_sentiment_scores(
    model: SentenceTransformer,
    segments: list[str],
    positive_anchors: list[str],
    negative_anchors: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute anchored sentiment scores for text segments.

    Score = mean cosine similarity to positive anchors
            minus mean cosine similarity to negative anchors.

    Positive scores indicate that a segment is closer to the positive anchors.
    Negative scores indicate that a segment is closer to the negative anchors.
    """
    if not positive_anchors:
        raise ValueError("positive_anchors is empty.")

    if not negative_anchors:
        raise ValueError("negative_anchors is empty.")

    positive_anchor_vectors = model.encode(
        positive_anchors,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    negative_anchor_vectors = model.encode(
        negative_anchors,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    segment_vectors = model.encode(
        segments,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    positive_similarity = util.cos_sim(
        segment_vectors,
        positive_anchor_vectors,
    ).mean(dim=1)

    negative_similarity = util.cos_sim(
        segment_vectors,
        negative_anchor_vectors,
    ).mean(dim=1)

    scores = positive_similarity - negative_similarity

    return scores.cpu().numpy()