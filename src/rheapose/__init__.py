"""The RheaPose package."""

import typing
import numpy
import polus.images.formats.label_to_vector as l2v
import polus.images.formats.vector_to_label as v2l
import tqdm


__version__ = "0.1.0"


def label_to_vector(labels: list[numpy.ndarray]) -> list[numpy.ndarray]:
    """Convert a list of label masks of objects to a list of vector-fields.

    Args:
        masks: A list of label masks of objects.

    Returns:
        A list of vector-fields.
    """
    vectors = []
    for label in tqdm.tqdm(labels):
        vectors.append(l2v.convert(label))
    return vectors


def vector_to_label(
    vectors: list[numpy.ndarray],
    masks: list[typing.Optional[numpy.ndarray]],
) -> list[numpy.ndarray]:
    """Convert a list of vector-fields to a list of label masks of objects.

    Args:
        vectors_masks: A list of vector-fields, and optional binary mask of objects.

    Returns:
        A list of label masks of objects.
    """
    if len(vectors) != len(masks):
        raise ValueError(
            f"The number of vectors and masks must be the same. Got {len(vectors)} vectors and {len(masks)} masks."
        )

    labels = []
    for vs, ms in tqdm.tqdm(zip(vectors, masks), total=len(vectors)):
        if ms is None:
            v_normed = l2v.dynamics.common.vector_norm(vs, axis=0)
            v_norm = numpy.linalg.norm(v_normed, axis=0)
            ms = v_norm > 0.5  # foreground should have unit vectors
        label = v2l.dynamics.convert(vs, ms, window_radius=3)
        labels.append(label)
    return labels
