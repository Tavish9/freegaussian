# FreeGaussian: Annotation-free Control of Articulated Objects via 3D Gaussian Splats with Flow Derivatives

> FreeGaussian: Annotation-free Control of Articulated Objects via 3D Gaussian Splats with Flow Derivatives <br />
> [Qizhi Chen*](https://github.com/Tavish9), [Delin Qu*](https://delinqu.github.io), Junli Liu, Yiwen Tang, Haoming Song, Dong Wang, Binzhao†, [Xuelong Li†](https://scholar.google.com/citations?user=ahUibskAAAAJ)

### [Paper](https://arxiv.org/abs/2410.22070) | [Project](https://tavish9.github.io/freegaussian) | [Video](https://youtu.be/)

<div align='center'>
<img src="https://github.com/Tavish9/freegaussian/raw/main/docs/assets/img/freegaussian/pipeline.png" >
</div>

## Roadmap
- [x] Project Pages for FreeGaussian [2025-11-28]
- [x] Code for FreeGaussian [2025-12-01]


## ⚙️ Installation
### 1. Install Nerfstudio
Follow the official instructions to install the latest version of [nerfstudio](https://docs.nerf.studio/quickstart/installation.html).
### 2. Install freegaussian
Clone the repository and install the package:
```
git clone https://github.com/Tavish9/freegaussian.git
pip install -e .
```

## 📥 Download Dataset

- [CoNeRF Dataset](https://github.com/kacperkan/conerf?tab=readme-ov-file#dataset-update-17-dec-2024)
- [LiveScene Dataset](https://huggingface.co/datasets/IPEC-COMMUNITY/freegaussian)
- [DyNeRF Dataset](https://github.com/facebookresearch/Neural_3D_Video)

## 🚀 Running freegaussian
### 1. Check running options
To view available options for training:
```
ns-train freegaussian --help
```

### 2. Launch Training

1. Stage 1: pretraining
```
ns-train freegaussian --data /path/to/your/freegaussian_dataset/scene_name freegaussian-real/sim-data
```
or
```
bash scripts/run.sh
```

2. Stage 2: post-training
```
ns-train freegaussian-control --data /path/to/your/freegaussian_dataset/scene_name freegaussian-real/sim-data --extra-kwargs
```
or
```
bash scripts/run_control.sh
```

## 📜 Citation

If you find our work useful, please cite:

```bibtex
@misc{chen2025freegaussianannotationfreecontrolarticulated,
      title={FreeGaussian: Annotation-free Control of Articulated Objects via 3D Gaussian Splats with Flow Derivatives}, 
      author={Qizhi Chen and Delin Qu and Junli Liu and Yiwen Tang and Haoming Song and Dong Wang and Bin Zhao and Xuelong Li},
      year={2025},
      eprint={2410.22070},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.22070}, 
}
```

## 🤝 Acknowledgement

We adapt codes from some awesome repositories, including [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [LiveScene](https://github.com/Tavish9/livescene). Thanks for making the code available! 🤗
