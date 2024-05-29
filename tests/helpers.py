"""Helpers for generating test data."""

import numpy
import skimage.data as sk_data
import skimage.measure as sk_measure


def gen_image(length: int, diameter: int) -> tuple[numpy.ndarray, numpy.ndarray, int]:
    """Generate an image for testing.

    The image will have rounded blobs as objects.
    The image will have dtype float32 and values between 0 and 1.
    The label mask will have dtype int32 and values between 0 and `n_objects`.

    Args:
        length: The length and width of the image.
        diameter: The typical diameter of the objects.

    Returns:
        A test image, the corresponding label mask, and the number of objects in the mask.
    """

    blob_size_fraction = diameter / length
    mask: numpy.ndarray = sk_data.binary_blobs(
        length=length,
        blob_size_fraction=blob_size_fraction,
        n_dim=2,
        volume_fraction=0.2,
    )

    # use poisson noise to make the image more realistic
    rng = numpy.random.default_rng(42)
    background = rng.poisson(2, mask.shape)
    foreground = rng.poisson(10, mask.shape) * mask
    image = (background + foreground).astype(numpy.float32)

    # normalize the image
    im_max = numpy.max(image)
    im_min = numpy.min(image)
    image = (image - im_min) / (im_max - im_min)

    # label the mask
    labels, num_objects = sk_measure.label(
        mask.astype(numpy.int32), connectivity=1, return_num=True
    )

    return image, labels.astype(numpy.int32), num_objects
