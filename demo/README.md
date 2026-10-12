---
title: PixelDense
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: NeurIPS 2026 pixel-space text-to-image diffusion
models:
  - xsourse/PixelDense
  - Qwen/Qwen3-1.7B
---

# PixelDense demo

Gradio text-to-image demo for **PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion** (NeurIPS 2026).

[🤗 Online demo](https://huggingface.co/spaces/mzhu1026/PixelDense) | [📄 Paper](https://arxiv.org/abs/2610.00483) | [🌐 Project Page](https://bio.lehanyang.info/PixelDense/) | [💻 Code](https://github.com/Hansxsourse/PixelDense) | [🤗 Model](https://huggingface.co/xsourse/PixelDense)

The demo reads `configs_t2i/pixeldense_inference_res512.yaml` and samples the released `PixelDense_XXL_T2I.ckpt` EMA denoiser the same way as `python main.py predict`, without Lightning: fp32 denoiser, bf16 Qwen3-1.7B text encoder, `AdamLMSamplerJiT` under bf16 autocast. It generates 512 × 512 images.

## Run locally

Requires a CUDA GPU.

```bash
cd demo
pip install -r requirements.txt
python app.py
```

When run from `demo/` inside the repository, the demo uses the repository code and `./ckpts/PixelDense_XXL_T2I.ckpt` if present. Otherwise the checkpoint is downloaded from [xsourse/PixelDense](https://huggingface.co/xsourse/PixelDense). Qwen3-1.7B is downloaded automatically.

## Deploy to Hugging Face Spaces

Upload the contents of this folder to a Gradio Space with ZeroGPU hardware. Outside the repository, `pipeline.py` clones the official code at the commit pinned in `REPO_COMMIT`.
