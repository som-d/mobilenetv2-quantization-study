# MobileNetV2 on CIFAR-10: A DevOps Engineer's First Look at Model Quantization

**Author:** Soham Deshmukh
**Date:** June 2026
**Code:** [GitHub — github.com/som-d/mobilenetv2-quantization-study](https://github.com/som-d/mobilenetv2-quantization-study)
**Environment:** Python 3.12, PyTorch 2.6.0 (CPU), TorchAO (fbgemm)

---

## 1. Why I Did This

I work as a DevOps engineer. My day job is about deployment pipelines, not training neural networks. But I wanted to understand the deployment side of ML — specifically, how models get small and fast enough to run on edge devices.

Model quantization is the bridge between "it works in a notebook" and "it runs on a Raspberry Pi." I built this project to see that bridge with my own eyes. I picked MobileNetV2 because it is already a lightweight architecture, and CIFAR-10 because it is small enough to train on a laptop CPU.

This is not a research paper. It is a learning project, documented honestly for anyone who wants to follow the same path.

---

## 2. What I Did

I compared four versions of the same model on the same test set:

| Phase | Method | Description |
|-------|--------|-------------|
| 1 | FP32 baseline | ImageNet-pretrained MobileNetV2, fine-tuned on CIFAR-10 |
| 2 | PTQ (Post-Training Quantization) | FP32 model -> calibrate on 2048 samples -> convert to int8 |
| 3 | Mixed int8 + FP32 | Stem + 2 blocks + classifier stay FP32, rest converted to int8 |
| 4 | QAT (Quantization-Aware Training) | Fake-quant during 2 more training epochs, then convert to int8 |

**Training details:**
- Resized from 32x32 CIFAR-10 to 224x224 for ImageNet-pretrained backbone
- 2 epochs fine-tuning for FP32 (SGD, lr=0.01, momentum=0.9)
- 2 epochs QAT from FP32 checkpoint (SGD, lr=0.0001, momentum=0.9)
- Calibration: 2048 training images with eval transforms
- All training on CPU (no GPU available), quantized models evaluated on CPU

---

## 3. Results

### Main Comparison

| Model | Accuracy | vs FP32 | Disk Size | Latency/batch | Speedup |
|-------|----------|---------|-----------|---------------|---------|
| FP32 | **88.53%** | — | 8.76 MB | 700.4 ms | 1.00x |
| PTQ int8 | **85.82%** | -2.71% | 2.53 MB | 167.1 ms | **4.19x** |
| Mixed | **87.81%** | -0.72% | 2.57 MB | 345.3 ms | **2.03x** |
| QAT int8 | **93.10%** | **+4.57%** | 2.53 MB | 162.7 ms | **4.30x** |

### Per-Class F1 Scores

| Class | FP32 | PTQ | Mixed | QAT |
|-------|------|-----|-------|-----|
| airplane | 0.895 | 0.874 | 0.885 | 0.937 |
| automobile | 0.946 | 0.940 | 0.948 | 0.966 |
| bird | 0.858 | 0.817 | 0.850 | 0.918 |
| cat | 0.762 | 0.750 | 0.753 | 0.857 |
| deer | 0.869 | 0.827 | 0.856 | 0.927 |
| dog | 0.819 | 0.796 | 0.809 | 0.880 |
| frog | 0.919 | 0.900 | 0.913 | 0.961 |
| horse | 0.912 | 0.875 | 0.905 | 0.949 |
| ship | 0.931 | 0.903 | 0.928 | 0.956 |
| truck | 0.933 | 0.903 | 0.930 | 0.958 |

---

## 4. What Surprised Me

### QAT beat FP32 by 4.57%

This was unexpected. The QAT model achieved **93.10%** vs the FP32 baseline of **88.53%**. I had to double-check the code to make sure I wasn't making a mistake.

My best explanation: the FP32 baseline was only fine-tuned for 2 epochs (limited by CPU training time — each epoch took ~54 minutes). The QAT phase added 2 more epochs of training with simulated quantization noise, which acted as a regularizer and let the model continue learning. If I had trained the FP32 baseline longer, it would likely have closed the gap. So the +4.57% is not QAT being magic — it is QAT + more training.

But it does show that quantization-aware training can produce models that are not just "good enough" but actually competitive with, or better than, their FP32 counterparts, especially when training time is limited.

### PTQ loss was uneven across classes

The PTQ model lost 2.71% overall, but some classes suffered more than others. Bird went from 0.858 to 0.817 (-4.8%). Deer went from 0.869 to 0.827 (-4.8%). Meanwhile automobile barely budged (0.946 to 0.940, -0.6%). This makes sense — classes with finer-grained features (bird species, deer antlers) are more sensitive to quantization noise than classes with distinctive shapes (automobiles, trucks).

### Mixed model as a practical compromise

The mixed model kept 2 middle blocks in FP32 along with the stem and classifier. It lost only 0.72% accuracy while still getting 2x speedup and 3.4x compression. For deployment scenarios where every percentage point matters, this is a good option.

---

## 5. Key Takeaways

1. **PTQ is the easiest path to deployment.** No retraining needed. You lose 2-3% accuracy but gain 4x speed and 3.5x compression. Good enough for many applications.

2. **QAT is worth the extra effort if you can train.** The model learns to be robust to quantization. If you are already training, add fake-quant nodes and train a bit longer.

3. **Mixed precision is a practical middle ground.** If you know which layers are sensitive, keep them in FP32 and quantize the rest. You get most of the benefit with minimal accuracy loss.

4. **CPU training is slow but viable for small models.** Each epoch took ~55 minutes on a laptop. MobileNetV2 is small enough. I would not try this with ResNet-50.

5. **The quant math is not magic.** Under the hood, it is just scale * (int8 - zero_point). I included a toy example in Phase 0 of the notebook that walks through the math on a tiny array. Seeing it by hand demystified the whole process.

---

## 6. What I Would Do Next

If I continue this project, here is what I would try:

- **Train the FP32 baseline to convergence** (10-20 epochs) so the QAT comparison is fair
- **Try int8 dynamic quantization** (activations stay FP32, weights are int8) — simpler than PTQ
- **Deploy to a real edge device** (Raspberry Pi, Arduino Portenta) and measure actual inference time
- **Profile which layers lose the most accuracy** under PTQ and target them with mixed precision
- **Try quantization on a larger model** (ResNet-18, EfficientNet-B0) to see if the patterns hold

---

## 7. How This Fits Into My PhD Goals

I am applying for PhD positions in **Edge AI and adaptive systems**. My background is DevOps — I know how to deploy and monitor systems at scale. This project was my first deep dive into the ML deployment side. It connects directly to questions I want to explore:

- How do you deploy models to resource-constrained devices without sacrificing accuracy?
- Can quantization be made adaptive — adjusting precision based on input complexity or device state?
- How do you monitor deployed quantized models for concept drift when labels are scarce?

These questions are at the intersection of systems engineering and ML. That is where I want to work.

---

*For the full code, including training loops, evaluation, and visualizations, see the companion Jupyter notebook: `mobilenetv2_ptq_qat_mixed_cifar10.ipynb`*

*This report was written in June 2026. Feedback and corrections are welcome.*
