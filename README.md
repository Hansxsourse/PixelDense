# PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion

**NeurIPS 2026**

[![arXiv](https://img.shields.io/badge/Arxiv-2610.00483-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2610.00483)
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://bio.lehanyang.info/PixelDense/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow.svg)](https://huggingface.co/xsourse/PixelDense)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **PixelDense** (NeurIPS 2026).

Representation alignment (REPA) usually aligns a diffusion transformer to a single semantic encoder such as DINOv2. PixelDense uses frozen dense-prediction models as alignment targets for pixel-space diffusion: DINOv2 and SAM2 supervise a **semantic projection stream**, Depth Anything v2 and Metric3D v2 supervise a **geometric projection stream**, and a weight-space **orthogonality penalty** keeps the two streams in disjoint subspaces of the denoiser feature. All teachers and alignment projections are used only during training; at inference the model is the 1.1B pixel-space denoiser sampled with the PixelGen sampler.

This repository is built on [PixelGen](https://github.com/Zehong-Ma/PixelGen).

## TODO

- [x] ~~arXiv~~
- [x] ~~Release code~~
- [x] ~~Release checkpoint~~
- [ ] Release demo

## Checkpoint

| Model | Params | Resolution | Download |
|-------|--------|------------|----------|
| PixelDense-XXL T2I | 1.1B | 512 | [🤗 HuggingFace](https://huggingface.co/xsourse/PixelDense/blob/main/PixelDense_XXL_T2I.ckpt) |

The checkpoint contains only the EMA denoiser weights. Load it with this repository: the denoiser has one learned embedding (`visual_type_embedding`) that upstream PixelGen does not have.

## Installation

```bash
# recommend python 3.10
pip install -r requirements.txt
```

## Inference

Download `PixelDense_XXL_T2I.ckpt` into `./ckpts/`:

```bash
hf download xsourse/PixelDense PixelDense_XXL_T2I.ckpt --local-dir ./ckpts
```

then run

```bash
python main.py predict -c ./configs_t2i/pixeldense_inference_res512.yaml
```

Images are written to `./workdirs/exp_pixeldense_inference/images/predict/`. To use your own prompts, pass a text file with one prompt per line:

```bash
python main.py predict -c ./configs_t2i/pixeldense_inference_res512.yaml \
    --data.pred_dataset.init_args.conditions ./prompts.txt
```

Inference only needs the denoiser and the Qwen3 text encoder; the dense teachers are not loaded.

## Training

PixelDense fine-tunes the released PixelGen-XXL T2I checkpoint on BLIP3o-60k.

**1. Teacher code.** SAM2 is installed as a package; Depth Anything V2 and Metric3D are used from their official repositories.

```bash
SAM2_BUILD_CUDA=0 pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

mkdir -p third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2
git -C third_party/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
git clone https://github.com/YvanYin/Metric3D third_party/Metric3D
git -C third_party/Metric3D checkout eb5b6fac0dc155e4e52f576e304fbf11655ff339
```

**2. Weights.** Download the following files into `./ckpts/`. DINOv2 ViT-B/14 (torch hub), Qwen3-1.7B and the LPIPS weights are downloaded automatically.

| File | Source |
|------|--------|
| `PixelGen_XXL_T2I.ckpt` | [zehongma/PixelGen](https://huggingface.co/zehongma/PixelGen/blob/main/PixelGen_XXL_T2I.ckpt) |
| `sam2.1_hiera_large.pt` | [SAM 2.1 Hiera-L](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |
| `depth_anything_v2_vitl.pth` | [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/blob/main/depth_anything_v2_vitl.pth) |
| `metric_depth_vit_large_800k.pth` | [Metric3D v2 ViT-L](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_large_800k.pth) |

**3. Data.** Download [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) and extract all tar shards into `./data/BLIP3o-60k`, so that every image has a caption file with the same name (`xxx.jpg` + `xxx.txt`).

**4. Train.**

```bash
python main.py fit -c ./configs_t2i/pixeldense_sft_res512.yaml
```

The config uses a per-GPU batch size of 16, 8 gradient-accumulation steps and `devices: auto` (all visible GPUs). The paper uses an effective batch size of 256; set `--trainer.devices` and `--trainer.accumulate_grad_batches` so that `num_gpus × 16 × accumulate_grad_batches = 256`. Training logs to Weights & Biases; run `wandb login` first or set `WANDB_MODE=offline`.

The denoiser is warm-started from the PixelGen-XXL T2I checkpoint. The warning about `visual_type_embedding` printed at start-up is expected: this parameter is not part of the PixelGen checkpoint and is initialized from scratch.

To sample from a checkpoint you trained, point the inference config to it:

```bash
python main.py predict -c ./configs_t2i/pixeldense_inference_res512.yaml \
    --model.denoiser.init_args.weight_path /path/to/your.ckpt
```

## Acknowledgement

This repository is built on [PixelGen](https://github.com/Zehong-Ma/PixelGen) and [DeCo](https://github.com/Zehong-Ma/DeCo). The teachers come from [DINOv2](https://github.com/facebookresearch/dinov2), [SAM 2](https://github.com/facebookresearch/sam2), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [Metric3D](https://github.com/YvanYin/Metric3D). We thank the authors for releasing their code and models.

## License

This project is released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{yang2026pixeldense,
  title={PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion},
  author={Yang, Lehan and Qi, Daiqing and Zhang, Wenhao and Li, Avery and Yang, Yiqing and Li, Yifan and Kong, Yu and Zheng, Haitian and Zhang, Zhifei and Lin, Zhe and Jampani, Varun and Li, Sheng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
