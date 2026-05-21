# Methodological Decisions

This document records methodological choices that are not fully visible from the public notebooks because the underlying corpus text and anchor sentences cannot be redistributed.

## Sentiment anchor set

The sentiment anchor set used in the current analysis contains 12 positive and 12 negative sentences.

The anchor sentences were manually selected from the target corpus as short, relatively unambiguous examples of positive/restorative and negative/threatening affective language. Ambiguous, mixed, or strongly context-dependent sentences were excluded from the anchor set.

The anchor texts are not included in this repository because they contain copyrighted material.

The semantic interpretation of positive and negative affect was informed by existing Tolkien scholarship, particularly work on light, darkness, evil, good, rhythm, and narrative movement. Relevant interpretive background includes Verlyn Flieger's *Splintered Light*, Tom Shippey's *J.R.R. Tolkien: Author of the Century*, and Ursula K. Le Guin's "Rhythmic Pattern in The Lord of the Rings", among other scholarship.

## Anchor and test separation

Anchor sentences and held-out test sentences are kept separate.

Anchor sentences are used as reference examples for the scoring procedure. Held-out test sentences are used only for evaluation. This separation is intended to avoid evaluating the model on the same examples that define the positive and negative reference centroids.

## Corpus segmentation

The trajectory notebook uses a prepared local plain-text version of the corpus. The current parser assumes:

- chapter markers in the form `###CHAPTER:`;
- a local paragraph-start convention based on leading spaces;
- blank lines as hard paragraph breaks.

These assumptions are source-specific and should be checked before applying the notebook to another corpus.

The default segmentation procedure is dialogue-aware and preserves chapter boundaries. Raw paragraphs are merged into larger analysis segments using separate token-window settings for narration and dialogue-like passages.

## Smoothing window

The trajectory notebook uses a default rolling smoothing window of 20 segments.

This is a visualisation parameter rather than part of the model itself. Figures generated from the notebook should report the smoothing window used.

Future work may include a sensitivity comparison across multiple smoothing windows, such as 10, 20, and 50 segments.

## Segment-level evaluation

The current evaluation notebook uses held-out sentence-level examples. This provides a controlled test of whether the model distinguishes manually labelled positive and negative examples.

The downstream trajectory analysis, however, operates on merged text segments. Segment-level evaluation is therefore a planned future extension.

## Light–Dark semantic-axis anchor set

The Light–Dark semantic axis uses two manually selected sets of anchor sentences from the target corpus. One set represents the Light pole and the other represents the Dark pole.

The anchor sentences were selected as relatively clear examples of light, darkness, shadow, illumination, obscurity, and related imagery in the target text. Ambiguous or strongly context-dependent examples were excluded where possible.

The anchor texts are not included in the repository because they contain copyrighted material.

The axis is oriented so that positive values indicate movement toward the Light pole and negative values indicate movement toward the Dark pole.