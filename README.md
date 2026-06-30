# WALD-SSIM

Reference implementation for the paper:

> **WALD-SSIM: Worst-Case and Axis-Directional Structural Similarity for Image Reconstruction**
> Samir Brahim Belhaouari and Farida Mohsen.

WALD-SSIM is a family of training-free, full-reference structural similarity
measures derived from SSIM. It replaces SSIM's spatial averaging with a
worst-case (minimax) rule over image patches, so the score is governed by the
most degraded region rather than the average, and adds an axis-directional
decomposition that scores row-wise and column-wise structure separately. A
multi-scale variant unifies both across resolutions.

This repository reproduces every figure and table in the paper.

## Installation

```bash
git clone https://github.com/<user>/wald-ssim.git
cd wald-ssim
pip install -r requirements.txt
```

Tested with Python 3.9–3.12. The metric itself depends only on NumPy and OpenCV;
`piq` and `torch` are used only to compute the SSIM/MS-SSIM/FSIM/IW-SSIM
baselines for comparison.

## Reproducing the paper

| Script | Produces |
|---|---|
| `ssim_experiments_paper.py` | Metric-sensitivity analysis (Table I, Fig. 3) and denoising evaluation (Table IV, Fig. 5) |
| `directional_ablation.py` | Directional-artifact validation (Table II, Fig. 4) and component ablation (Table III) |


Run any script directly, for example:

```bash
python ssim_experiments_paper.py
python directional_ablation.py
python make_teaser.py
```

Each script prints the numbers reported in the paper and writes its figures to
`figs/`. All experiments use the same five standard grayscale test images
(`cameraman`, `moon`, `coins`, `clock`, and a luminance-converted `astronaut`)
from `scikit-image`, with non-overlapping 16×16 patches, α = 0.5, and the
standard SSIM constants C1 = (0.01·255)^2 and C2 = (0.03·255)^2.

## The metric

The core functions implement the formulation in Section III of the paper:

- `sub_ssim` — worst-case sub-image SSIM (Eq. 1)
- `ssim_row`, `ssim_col`, `dir_ssim` — axis-directional SSIM (Eqs. 2–3)
- `combined` — worst-case over directional patches (Eq. 4)
- `ms_combined` — multi-scale local-directional measure (Eq. 5)

All measures are defined as losses internally; the scripts report the
corresponding scores (1 − loss) so that higher values mean greater similarity.
The worst-case score can be negative when a patch SSIM is negative, which
signals a severely degraded local region.




## License

Released under the MIT License. See [LICENSE](LICENSE).
