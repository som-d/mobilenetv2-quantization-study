# Models directory

Checkpoints saved when `SAVE_MODEL_FILES = True` in the main notebook.

Typical files after a full run:

| File | Description |
|------|-------------|
| `mobilenetv2_fp32.pth` | FP32 baseline (8.76 MB) |
| `mobilenetv2_ptq_int8.pth` | PTQ int8 (2.53 MB) |
| `mobilenetv2_mixed.pth` | Mixed FX model (2.57 MB) |
| `mobilenetv2_qat_int8.pth` | QAT int8 (2.53 MB) |

These are local only — not tracked in git. Run the notebook to regenerate them.
