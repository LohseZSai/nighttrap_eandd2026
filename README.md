# NightTrap Code Package

This is an anonymous staging copy of NightTrap scripts and paper-side reproducibility material.

## Included

- `remote_scripts/nighttrap_ops_v1/`: paper artifact generation and diagnostic scripts.
- `paper/`: sanitized LaTeX source for local reference.
- `requirements.txt` and `environment.yml`: minimal Python environment for table/audit scripts.
- `smoke_test_release.py`: package-integrity smoke test for the paired dataset repository.

## Paired dataset repository

The derived dataset package is staged at:

https://huggingface.co/datasets/LohseRre/nighttrap-eandd2026

## Not included

Large training checkpoints, raw images, private API keys, and source-dataset archives are not included.

The included scripts are the paper-side artifact, diagnostic, and table-generation scripts. Some full experiment reruns require the original benchmark workspace, source images, model checkpoints, or embeddings that are not redistributed here.

## Reproducibility scope

This repository is executable for release checks and paper-side artifact scripts. It is not a full raw-image training environment. Full reruns of VLM inference, detector-crop baselines, or tuned-model training require upstream raw images, checkpoints, and embeddings described in the paper and dataset repository.

## Smoke test

Run this from the code repository after placing the dataset repository beside it, or set `NIGHTTRAP_DATASET_ROOT` to a local clone of the Hugging Face dataset package:

```bash
python3 -m pip install -r requirements.txt
NIGHTTRAP_DATASET_ROOT=../dataset_repo python3 smoke_test_release.py
```

The smoke test checks that the derived dataset package is present, contains the frozen catalog and task files, excludes raw media files, and matches the paper-level catalog counts.
