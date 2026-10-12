"""Standalone PixelDense text-to-image inference.

Reproduces `python main.py predict -c configs_t2i/pixeldense_inference_res512.yaml`
from the official repository without Lightning: the released JiT-T2I denoiser
(EMA weights, fp32), the Qwen3-1.7B text encoder (bf16), the AdamLM sampler and
bf16 autocast, matching the `bf16-mixed` trainer precision of the config.
"""

import os
import shutil
import subprocess
import sys

import torch
import yaml
from huggingface_hub import hf_hub_download

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "configs_t2i/pixeldense_inference_res512.yaml"

# Code fetched when the demo runs outside a PixelDense checkout (e.g. on Hugging Face Spaces).
REPO_URL = "https://github.com/Hansxsourse/PixelDense"
REPO_COMMIT = "2d5617633940bef700fffcc4f93f7d8c7ec42bce"

MODEL_REPO = "xsourse/PixelDense"
MODEL_FILE = "PixelDense_XXL_T2I.ckpt"


def find_repo():
    """Return the PixelDense code root and make `src` importable.

    Uses the enclosing checkout when run from `demo/` in the repository; otherwise
    clones the official code at a pinned commit next to this file.
    """
    repo_dir = os.path.dirname(DEMO_DIR)
    if not os.path.isfile(os.path.join(repo_dir, CONFIG_PATH)):
        repo_dir = os.path.join(DEMO_DIR, "PixelDense")
        if not os.path.isfile(os.path.join(repo_dir, CONFIG_PATH)):
            shutil.rmtree(repo_dir, ignore_errors=True)
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, repo_dir], check=True)
            subprocess.run(["git", "-C", repo_dir, "checkout", "--quiet", REPO_COMMIT], check=True)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir


REPO_DIR = find_repo()

from transformers import Qwen2Tokenizer, Qwen3Model  # noqa: E402

from src.diffusion.base.guidance import simple_guidance_fn  # noqa: E402
from src.diffusion.flow_matching.adam_sampling import AdamLMSamplerJiT  # noqa: E402
from src.diffusion.flow_matching.scheduling import LinearScheduler  # noqa: E402
from src.models.autoencoder.base import fp2uint8  # noqa: E402
from src.models.transformer.JiT_T2I import JiT_T2I  # noqa: E402

# Same global setting as src/utils/patch_bugs.py, imported by main.py.
torch.set_float32_matmul_precision("medium")


def load_config():
    with open(os.path.join(REPO_DIR, CONFIG_PATH)) as f:
        return yaml.safe_load(f)


class PixelDensePipeline:
    def __init__(self, denoiser, tokenizer, text_encoder, config, device):
        self.denoiser = denoiser
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = torch.device(device)

        model_cfg = config["model"]
        self.embed_dim = model_cfg["conditioner"]["init_args"]["embed_dim"]
        self.max_length = model_cfg["conditioner"]["init_args"]["max_length"]
        self.latent_shape = tuple(config["data"]["pred_dataset"]["init_args"]["latent_shape"])

        sampler_cfg = model_cfg["diffusion_sampler"]["init_args"]
        self.default_steps = sampler_cfg["num_steps"]
        self.default_guidance = sampler_cfg["guidance"]
        self.default_timeshift = sampler_cfg["timeshift"]
        self.order = sampler_cfg["order"]

    @classmethod
    def from_pretrained(cls, device="cuda"):
        config = load_config()
        model_cfg = config["model"]

        denoiser_args = dict(model_cfg["denoiser"]["init_args"])
        denoiser_args.pop("weight_path")
        denoiser_args.pop("load_ema")
        with torch.device("meta"):
            denoiser = JiT_T2I(**denoiser_args)

        # The checkpoint holds {"state_dict": {"ema_denoiser.<param>": tensor}}.
        # Reuse ./ckpts/ from the repository README if present, otherwise download.
        ckpt_path = os.path.join(REPO_DIR, "ckpts", MODEL_FILE)
        if not os.path.isfile(ckpt_path):
            ckpt_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
        prefix = "ema_denoiser."
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        denoiser.load_state_dict(state_dict, strict=True, assign=True)
        denoiser = denoiser.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

        cond_args = model_cfg["conditioner"]["init_args"]
        tokenizer = Qwen2Tokenizer.from_pretrained(
            cond_args["weight_path"], max_length=cond_args["max_length"], padding_side="right"
        )
        text_encoder = Qwen3Model.from_pretrained(cond_args["weight_path"], dtype=torch.bfloat16)
        text_encoder = text_encoder.to(device).eval().requires_grad_(False)

        return cls(denoiser, tokenizer, text_encoder, config, device)

    def encode(self, prompts):
        # Mirrors Qwen3TextEncoder._impl_condition.
        tokenized = self.tokenizer(
            prompts, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        y = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        if y.shape[2] < self.embed_dim:
            y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], self.embed_dim - y.shape[2]).to(y.device, y.dtype)], dim=-1)
        if y.shape[2] > self.embed_dim:
            y = y[:, :, :self.embed_dim]
        return y.to(torch.bfloat16)

    @torch.inference_mode()
    def __call__(
        self,
        prompts,
        seeds,
        negative_prompt="",
        num_steps=None,
        guidance=None,
        timeshift=None,
        callback=None,
    ):
        """Generate one 512x512 image per (prompt, seed) pair. Returns uint8 HWC numpy arrays."""
        num_steps = num_steps or self.default_steps
        guidance = self.default_guidance if guidance is None else guidance
        timeshift = self.default_timeshift if timeshift is None else timeshift

        # Same noise as RandomNDataset for a given seed.
        noise = torch.stack([
            torch.randn(self.latent_shape, generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float32)
            for seed in seeds
        ]).to(self.device)

        sampler = AdamLMSamplerJiT(
            num_steps=num_steps,
            guidance=guidance,
            timeshift=timeshift,
            order=self.order,
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
        )

        net = self.denoiser
        if callback is not None:
            step = [0]

            def net(x, t, y):
                out = self.denoiser(x, t, y)
                step[0] += 1
                callback(step[0], num_steps)
                return out

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            condition = self.encode(list(prompts))
            uncondition = self.encode([negative_prompt or ""]).repeat(len(prompts), 1, 1)
            samples = sampler(net, noise, condition, uncondition)

        # PixelAE decode is the identity (scale=1, shift=0); then [-1, 1] -> uint8.
        samples = fp2uint8(samples.float())
        return list(samples.permute(0, 2, 3, 1).cpu().numpy())
