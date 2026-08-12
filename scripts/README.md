# Scientific workflows

Run commands from the repository root with the project virtual environment active.

- `catalogue/`: build, repair, and validate halo catalogues.
- `powerspectra/`: compute spectra, covariance products, and masking tests.
- `inference/`: run and summarize parameter chains.
- `figures/`: generate diagnostic and paper plots.

Each workflow has one canonical command; threshold-specific cases use command-line options such as `--q-cuts` or `--single-cut`.
