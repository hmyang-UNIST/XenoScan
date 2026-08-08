# Trained weights

The `.h5` weight files are **not stored in this repository** — together they are 229 MB,
which makes cloning slow for no benefit. They are archived on Zenodo instead, in the
same record as the image dataset: https://doi.org/10.5281/zenodo.21845686

From the repository root:

```bash
python download_models.py            # fetch every missing weight file
python download_models.py --check    # report what is present
```

Each download is verified against the MD5 that Zenodo publishes for it.

These are the exact weights used for every result in the paper. Retraining will not reproduce
them bit-for-bit: random seeds were not fixed. See `../../MODIFICATIONS.md`.
