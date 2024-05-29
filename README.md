# RheaPose

## Benchmarks

This repository uses git submodules to include the Vector-Field converter tools from [PolusAI](https://github.com/PolusAI/image-tools).
You will need to update the submodules to get the latest version of the tools:

```bash
git submodule update --init
```

We used Python 3.11.8 on a Linux to run all the benchmarks.
The benchmarks were run using a single CPU core and no GPU.

To get started, create a virtual environment and install the package with poetry:

```bash
python -m venv .venv
source .venv/bin/activate
poetry install
```

Then, you can run the benchmarks with:

```bash
python benches/bench_vector_converters.py
```

Note that the Omnipose package is currently broken and will not run.

### TissueNet Data

1. Download the TissueNet dataset (v1.1) from [DeepCell Datasets](https://datasets.deepcell.org/).
1. Extract the dataset to a location of your choice.
1. Pass the path to the dataset to the `TISSUENET_DIR` environment variable.

| split | num_images | image_size |
| ----- | ---------- | ---------- |
| train | 2580       | 512        |
| val   | 3118       | 512        |
| test  | 1324       | 256        |

Summary of the benchmarks:

### RheaPose Performance

| split | time(s)  | loss     |
| ----- | -------- | -------- |
| train | 2.74e+02 | 5.42e-05 |
| val   | 7.28e+01 | 5.70e-05 |
| test  | 3.64e+01 | 7.63e-05 |

### Cellpose

| split | time(s)  | loss     |
| ----- | -------- | -------- |
| train | 4.32e+03 | 3.57e-01 |
| val   | 2.06e+03 | 4.25e-01 |
| test  | 8.28e+02 | 3.96e-01 |

### RheaPose Multi-Trip

| split | num_trips | losses                                                                                                                                                                                                                       |
| ----- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train | 10        | [(8.62e-05, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05), (0.00e+00, 8.62e-05)] |
| val   | 10        | [(7.63e-05, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05), (0.00e+00, 7.63e-05)] |
| test  | 10        | [(5.65e-05, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05), (0.00e+00, 5.65e-05)] |

### Cellpose Multi-Trip

| split | num_trips | losses                                                                                                                                                                                                                       |
| ----- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train | 10        | [(3.85e-01, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01), (0.00e+00, 3.85e-01)] |
| val   | 10        | [(4.12e-01, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01), (0.00e+00, 4.12e-01)] |
| test  | 10        | [(3.78e-01, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01), (0.00e+00, 3.78e-01)] |
