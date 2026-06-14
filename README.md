# MobileNetV2 Quantization Study (CIFAR-10)

Self-directed study comparing **FP32**, **PTQ (int8)**, **mixed FP32+int8 (FX)**, and **QAT** on MobileNetV2 for CIFAR-10 classification.

**Author:** Soham Deshmukh  
**Context:** PhD outreach portfolio (edge AI, efficient inference, deployed ML systems)

---

## Results Summary

| Model | Accuracy | vs FP32 | Disk Size | Latency/batch | Speedup |
|-------|----------|---------|-----------|---------------|---------|
| FP32 | **88.53%** | — | 8.76 MB | 700.4 ms | 1.00x |
| PTQ int8 | **85.82%** | -2.71% | 2.53 MB | 167.1 ms | **4.19x** |
| Mixed | **87.81%** | -0.72% | 2.57 MB | 345.3 ms | **2.03x** |
| QAT int8 | **93.10%** | +4.57% | 2.53 MB | 162.7 ms | **4.30x** |

Full per-class F1 breakdown and confusion matrices are in the [technical report](technical_report.md).

---

## Repository Layout

| File / Folder | Role |
|---------------|------|
| `mobilenetv2_ptq_qat_mixed_cifar10.ipynb` | **Main pipeline** — full study notebook |
| `technical_report.md` | Written report with results, analysis, and key takeaways |
| `results/research_report.csv` | One row per model (accuracy, F1, disk, latency, speedup) |
| `models/` | Saved `.pth` checkpoints (local only, not in git) |
| `data/` | CIFAR-10 cache (local only, not in git — downloaded on first run) |
| `requirements.txt` | Python dependencies |

---

## Quick Start

1. **Python 3.12** (PyTorch 2.6 + CUDA optional).
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `mobilenetv2_ptq_qat_mixed_cifar10.ipynb`.
4. Run **cell 1** (environment setup), then **config**, then phases **0 → 5**.

---

## Pipeline (Main Notebook)

| Phase | Content |
|-------|---------|
| 0 | Toy quant math (scale, zero-point) |
| 1 | Train FP32 baseline |
| 2 | PTQ: calibrate + convert to int8 |
| 3 | Mixed FP32 + int8 (FX partial quant) |
| 4 | QAT: fake-quant training + convert |
| 5 | Report: accuracy, F1, disk size, latency, curves, confusion matrices |

Output: `results/research_report.csv` (one row per model).

---

## Data

CIFAR-10 is downloaded automatically into `data/` on first run. Raw data stays out of git.

---

## Citation / Contact

If you use this work, please credit the repository link.  
Questions: Soham Deshmukh — sohamdeshmukh611@gmail.com
