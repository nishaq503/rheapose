"""Run benchmarks for vector converters."""

import os
import pathlib
import time
import typing

import numpy

import helpers


TISSUENET_DIR = pathlib.Path(
    os.environ.get(
        "TISSUENET_DIR",
        str(
            pathlib.Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath("data", "tissuenet_v1.1")
        ),
    )
).resolve()

DATA = typing.Literal["TissueNet"]
SPLIT = typing.Literal["train", "val", "test"]


def read_tissuenet(split: SPLIT) -> list[numpy.ndarray]:
    """Read the TissueNet dataset.

    Args:
        split: The split to read.

    Returns:
        A list of label masks of objects.
    """

    assert (
        TISSUENET_DIR.exists()
    ), f"Could not find the TissueNet dataset at {TISSUENET_DIR}."
    assert TISSUENET_DIR.is_dir(), f"The path {TISSUENET_DIR} is not a directory."

    split_name = f"tissuenet_v1.1_{split}.npz"
    assert (
        TISSUENET_DIR / split_name
    ).exists(), f"Could not find the split {split_name} in {TISSUENET_DIR}."

    data_dict = numpy.load(str(TISSUENET_DIR / split_name))
    labels: numpy.ndarray = data_dict["y"]
    nuclear_labels = labels[..., 1:2].squeeze()
    return [nl for nl in nuclear_labels]


def bench_rheapose(data: DATA, split: SPLIT, reuse_masks: bool) -> tuple[float, float]:
    """Benchmark the round-trip conversion of label masks of objects using RheaPose.

    Args:
        data: The dataset to read.
        split: The split to read.
        reuse_masks: Whether to reuse the masks.

    Returns:
        The time taken and the mean loss.
    """
    print(f"Reading {data} {split}...")
    if data == "TissueNet":
        labels = read_tissuenet(split)
    else:
        raise ValueError(f"Unknown data {data}.")
    print(f"Read {len(labels)} images.")

    print(f"Running RheaPose round-trip conversion on {data} {split}...")
    start = time.perf_counter()
    out_labels = helpers.rp_round_trip(labels, reuse_masks)
    time_taken = time.perf_counter() - start

    mean_loss = numpy.mean(
        [((in_l > 0) != (out_l > 0)).mean() for in_l, out_l in zip(labels, out_labels)]
    )

    print(
        f"RheaPose round-trip conversion on {data} {split} with {len(labels)} images took {time_taken:.2e}s with a mean loss of {mean_loss:.2e}."
    )

    return time_taken, mean_loss


def bench_rheapose_multi_trips(
    data: DATA,
    split: SPLIT,
    num_trips: int,
) -> list[tuple[float, float]]:
    """Benchmark the round-trip conversion of label masks of objects using RheaPose with multiple round trips.

    Args:
        data: The dataset to read.
        split: The split to read.

    Returns:
        A list of losses after each round trip. Each element is a tuple of:
            - loss vs previous round trip
            - loss vs original labels
    """
    print(f"Reading {data} {split}...")
    if data == "TissueNet":
        labels = read_tissuenet(split)
    else:
        raise ValueError(f"Unknown data {data}.")
    print(f"Read {len(labels)} images.")

    prev_labels = labels
    losses = []
    for i in range(num_trips):
        print(f"Running RheaPose round-trip {i + 1}/{num_trips} on {data} {split}...")
        out_labels = helpers.rp_round_trip(labels, reuse_masks=False)
        prev_loss = numpy.mean(
            [
                ((in_l > 0) != (out_l > 0)).mean()
                for in_l, out_l in zip(prev_labels, out_labels)
            ]
        )
        orig_loss = numpy.mean(
            [
                ((in_l > 0) != (out_l > 0)).mean()
                for in_l, out_l in zip(labels, out_labels)
            ]
        )
        losses.append((prev_loss, orig_loss))
        prev_labels = out_labels

        print(f"Round trip {i + 1} losses: {prev_loss:.2e}, {orig_loss:.2e}")

    print(f"losses: [{', '.join([f'({l_[0]:.2e}, {l_[1]:.2e})' for l_ in losses])}]")

    return losses


def bench_cellpose(data: DATA, split: SPLIT) -> tuple[float, float]:
    """Benchmark the round-trip conversion of label masks of objects using Cellpose.

    Args:
        data: The dataset to read.
        split: The split to read.

    Returns:
        The time taken and the mean loss.
    """
    print(f"Reading {data} {split}...")
    if data == "TissueNet":
        labels = read_tissuenet(split)
    else:
        raise ValueError(f"Unknown data {data}.")
    print(f"Read {len(labels)} images.")

    print(f"Running Cellpose round-trip conversion on {data} {split}...")
    start = time.perf_counter()
    out_labels = helpers.cp_round_trip(labels, diameter=16, reuse_masks=False)
    time_taken = time.perf_counter() - start

    mean_loss = numpy.mean(
        [((in_l > 0) != (out_l > 0)).mean() for in_l, out_l in zip(labels, out_labels)]
    )

    print(
        f"Cellpose round-trip conversion on {data} {split} with {len(labels)} images took {time_taken:.2e}s with a mean loss of {mean_loss:.2e}."
    )

    return time_taken, mean_loss


def bench_cellpose_multi_trips(
    data: DATA,
    split: SPLIT,
    num_trips: int,
) -> list[tuple[float, float]]:
    """Benchmark the round-trip conversion of label masks of objects using Cellpose with multiple round trips.

    Args:
        data: The dataset to read.
        split: The split to read.

    Returns:
        A list of losses after each round trip. Each element is a tuple of:
            - loss vs previous round trip
            - loss vs original labels
    """
    print(f"Reading {data} {split}...")
    if data == "TissueNet":
        labels = read_tissuenet(split)[:10]
    else:
        raise ValueError(f"Unknown data {data}.")
    print(f"Read {len(labels)} images.")

    prev_labels = labels
    losses = []
    for i in range(num_trips):
        print(f"Running Cellpose round-trip {i + 1}/{num_trips} on {data} {split}...")
        out_labels = helpers.cp_round_trip(labels, diameter=16, reuse_masks=False)
        prev_loss = numpy.mean(
            [
                ((in_l > 0) != (out_l > 0)).mean()
                for in_l, out_l in zip(prev_labels, out_labels)
            ]
        )
        orig_loss = numpy.mean(
            [
                ((in_l > 0) != (out_l > 0)).mean()
                for in_l, out_l in zip(labels, out_labels)
            ]
        )
        losses.append((prev_loss, orig_loss))
        prev_labels = out_labels

        print(f"Round trip {i + 1} losses: {prev_loss:.2e}, {orig_loss:.2e}")

    print(f"losses: [{', '.join([f'({l_[0]:.2e}, {l_[1]:.2e})' for l_ in losses])}]")

    return losses


def bench_omnipose(data: DATA, split: SPLIT) -> tuple[float, float]:
    """Benchmark the round-trip conversion of label masks of objects using OmniPose.

    Args:
        data: The dataset to read.
        split: The split to read.

    Returns:
        The time taken and the mean loss.
    """
    # TODO: Omnipose is currently broken. The error is:
    #   File ".../rheapose/.venv/lib/python3.11/site-packages/omnipose/core.py", line 267, in labels_to_flows
    #     labels, dist, heat, veci = map(list,zip(*[masks_to_flows(labels[n], links=links[n], use_gpu=use_gpu,
    #     ^^^^^^^^^^^^^^^^^^^^^^^^
    # ValueError: too many values to unpack (expected 4)
    print(f"Reading {data} {split}...")
    if data == "TissueNet":
        labels = read_tissuenet(split)[:10]  # TODO: After fixing, remove the slicing
    else:
        raise ValueError(f"Unknown data {data}.")
    print(f"Read {len(labels)} images.")

    print(f"Running OmniPose round-trip conversion on {data} {split}...")
    start = time.perf_counter()
    out_labels = helpers.om_round_trip(labels)
    time_taken = time.perf_counter() - start

    mean_loss = numpy.mean(
        [((in_l > 0) != (out_l > 0)).mean() for in_l, out_l in zip(labels, out_labels)]
    )

    print(
        f"OmniPose round-trip conversion on {data} {split} with {len(labels)} images took {time_taken:.2e}s with a mean loss of {mean_loss:.2e}."
    )

    return time_taken, mean_loss


def bench_all() -> None:
    """Run all benchmarks."""
    bench_rheapose("TissueNet", "train", False)
    bench_rheapose("TissueNet", "val", False)
    bench_rheapose("TissueNet", "test", False)
    bench_cellpose("TissueNet", "train")
    bench_cellpose("TissueNet", "val")
    bench_cellpose("TissueNet", "test")
    # Omnipose is currently broken
    # bench_omnipose("TissueNet", "train")
    # bench_omnipose("TissueNet", "val")
    # bench_omnipose("TissueNet", "test")


def bench_all_multi_trips() -> None:
    """Run all benchmarks with multiple round trips."""
    bench_rheapose_multi_trips("TissueNet", "train", 10)
    bench_rheapose_multi_trips("TissueNet", "val", 10)
    bench_rheapose_multi_trips("TissueNet", "test", 10)
    bench_cellpose_multi_trips("TissueNet", "train", 10)
    bench_cellpose_multi_trips("TissueNet", "val", 10)
    bench_cellpose_multi_trips("TissueNet", "test", 10)


if __name__ == "__main__":
    # bench_all()
    bench_all_multi_trips()
