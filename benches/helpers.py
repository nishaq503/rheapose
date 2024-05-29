"""Helpers for wrapping functions from cellpose and omnipose."""

import numpy
import cellpose.models as cp_models
import cellpose.dynamics as cp_dynamics
import polus.images.formats.label_to_vector as l2v
import omnipose.core as op_core
import tqdm
import rheapose


def cp_l2v(labels: list[numpy.ndarray], diameter: int | None) -> list[numpy.ndarray]:
    """Convert a list of label masks of objects to a list of vector-fields.

    If the `diameter` is not provided, it is estimated using the `SizeModel` from Cellpose.

    Args:
        masks: A list of label masks of objects.
        diameter: The typical diameter of the objects.

    Returns:
        A list of vector-fields.
    """
    if diameter is None:
        cp_model = cp_models.CellposeModel(gpu=False, model_type="nuclei")
        sz_model = cp_models.SizeModel(cp_model)
        diameter, style = sz_model.eval(labels)
        rescale = diameter / style
        _, flows, _ = cp_model.eval(labels, rescale=rescale)
    else:
        model = cp_models.Cellpose(gpu=False, model_type="nuclei")
        _, flows, _, _ = model.eval(labels, diameter=diameter)

    return [f[1] for f in flows]


def cp_v2l(
    vectors: list[numpy.ndarray],
    masks: list[numpy.ndarray] | None,
) -> list[numpy.ndarray]:
    """Convert a list of vector-fields to a list of label masks of objects.

    Args:
        vectors: A list of vector-fields.
        masks: A list of binary masks of objects.

    Returns:
        A list of label masks of objects.
    """
    if masks is None:
        masks = []
        for vs in vectors:
            v_normed = l2v.dynamics.common.vector_norm(vs, axis=0)
            masks.append(numpy.linalg.norm(v_normed, axis=0))

    if len(vectors) != len(masks):
        raise ValueError(
            f"The number of vectors and masks must be the same. Got {len(vectors)} vectors and {len(masks)} masks."
        )

    labels = []
    for vs, ms in tqdm.tqdm(zip(vectors, masks), total=len(vectors)):
        label, _ = cp_dynamics.compute_masks(vs, ms)
        labels.append(label)
    return labels


def cp_round_trip(
    labels: list[numpy.ndarray], diameter: int | None, reuse_masks: bool
) -> list[numpy.ndarray]:
    """Perform a round-trip conversion of label masks of objects using Cellpose.

    Args:
        labels: A list of label masks of objects.
        diameter: The typical diameter of the objects.
        reuse_masks: Whether to reuse the masks for converting vectors to labels.

    Returns:
        A list of label masks of objects.
    """
    vectors = cp_l2v(labels, diameter)
    assert (
        type(vectors[0]) == numpy.ndarray
    ), f"Expected numpy.ndarray, got {type(vectors[0])}"
    if reuse_masks:
        masks = [(label > 0) for label in labels]
    else:
        masks = None
    return cp_v2l(vectors, masks)


def om_l2v(
    labels: list[numpy.ndarray],
) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
    """Convert a list of label masks of objects to a list of vector-fields.

    Args:
        masks: A list of label masks of objects.
        diameter: The typical diameter of the objects.

    Returns:
        A list of distance fields, and a list of vector-fields.
    """
    flows = op_core.labels_to_flows(labels)
    distance_fields = [f[0] for f in flows]
    vectors = [f[1, 2] for f in flows]
    return distance_fields, vectors


def om_v2l(
    vectors: list[numpy.ndarray],
    distance_fields: list[numpy.ndarray],
) -> list[numpy.ndarray]:
    """Convert a list of vector-fields to a list of label masks of objects.

    Args:
        vectors: A list of vector-fields.
        distance_fields: A list of distance fields.

    Returns:
        A list of label masks of objects.
    """
    if len(vectors) != len(distance_fields):
        raise ValueError(
            f"The number of vectors and distance fields must be the same. Got {len(vectors)} vectors and {len(distance_fields)} distance fields."
        )

    labels = []
    for vs, ds in zip(vectors, distance_fields):
        label = op_core.compute_masks(vs, ds)[0]
        labels.append(label)
    return labels


def om_round_trip(labels: list[numpy.ndarray]) -> list[numpy.ndarray]:
    """Perform a round-trip conversion of label masks of objects using Omnipose.

    Args:
        labels: A list of label masks of objects.

    Returns:
        A list of label masks of objects.
    """
    distance_fields, vectors = om_l2v(labels)
    return om_v2l(vectors, distance_fields)


def rp_round_trip(
    labels: list[numpy.ndarray], reuse_masks: bool
) -> list[numpy.ndarray]:
    """Perform a round-trip conversion of label masks of objects using RheaPose.

    Args:
        labels: A list of label masks of objects.
        reuse_masks: Whether to reuse the masks for converting vectors to labels.

    Returns:
        A list of label masks of objects.
    """
    vectors = rheapose.label_to_vector(labels)
    if reuse_masks:
        masks = [(label > 0) for label in labels]
    else:
        masks = [None] * len(labels)
    return rheapose.vector_to_label(vectors, masks)
