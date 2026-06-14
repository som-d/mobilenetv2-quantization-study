# Models directory

Checkpoints saved when `SAVE_MODEL_FILES = True` in the main notebook.

Typical files after a full run:

| File | Description |
|------|-------------|
| `mobilenetv2_fp32.pth` | FP32 baseline |
| `mobilenetv2_ptq_int8.pth` | PTQ int8 |
| `mobilenetv2_mixed.pth` | Mixed FX model |
| `mobilenetv2_qat_int8.pth` | QAT int8 |

These are **tracked in git** so professors can see your trained artifacts.  
If a file exceeds GitHub’s 100 MB limit, use [Git LFS](https://git-lfs.github.com/) or keep only the best checkpoint.
