# Data directory

## What lives here

After you run `mobilenetv2_ptq_qat_mixed_cifar10.ipynb`, PyTorch downloads **CIFAR-10** into this folder:

- `cifar-10-python.tar.gz` (~163 MB)
- `cifar-10-batches-py/` (extracted batches)

Total size is about **340 MB**.

## Is it in git?

**No.** Raw CIFAR-10 is **not** committed to this repository.

| Approach | Recommendation |
|----------|----------------|
| **Commit full `data/`** | Not recommended. Slow clones, GitHub file limits (100 MB/file), no real benefit (public dataset). |
| **Commit only this README** | Yes. Documents how to reproduce. |
| **Git LFS for data** | Optional if you insist on a frozen copy; costs LFS bandwidth on GitHub. |
| **Re-download on new machine** | Best default. Notebook downloads CIFAR-10 automatically on first run. |

## Reproduce on a new machine

1. Clone the repo.
2. Run the main notebook from the top (cell 1 + config + data cell).
3. CIFAR-10 will appear here again in a few minutes.

Official dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
