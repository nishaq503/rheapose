"""Test the vector-field converters."""

import numpy
import rheapose

from . import helpers


def test_label_to_vector():
    """Test the label-to-vector converter."""
    _, masks, _ = helpers.gen_image(1024, 16)
    vectors = rheapose.label_to_vector([masks])
    assert len(vectors) == 1

    vectors = vectors[0]
    assert vectors.shape == (2,) + masks.shape


def test_vector_to_label():
    """Test the vector-to-label converter."""
    _, masks, _ = helpers.gen_image(1024, 16)
    vectors = rheapose.label_to_vector([masks])

    bin_masks = masks > 0
    new_masks = rheapose.vector_to_label(vectors, [None])
    assert len(new_masks) == 1

    new_masks = new_masks[0]
    assert new_masks.shape == masks.shape
    assert new_masks.dtype == masks.dtype

    new_bin_mask = new_masks > 0
    mean_diff = numpy.mean(new_bin_mask != bin_masks)
    assert mean_diff < 1e-4, f"Mean difference: {mean_diff:.2e}"
