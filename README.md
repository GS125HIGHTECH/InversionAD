# Modified InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models for Time Series Data</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.11838-b31b1b.svg)](https://arxiv.org/abs/2504.05662)&nbsp;
[![Project Page](https://img.shields.io/badge/Project%20Page-visit-blue.svg)](https://invad-project.com)
[![AlphaArXiv](https://img.shields.io/badge/AlphaArXiv-2504.05662-b31b1b.svg)](https://www.alphaxiv.org/abs/2504.05662)

# Train
```
python src/train.py --config configs/oil_config.yaml 
```

# Test
```
$env:PYTHONPATH = "."
python src/evaluate.py --save_dir results/oil_diffusion_experiment --use_best_model --visualize_samples
```
