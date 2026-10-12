import spaces  # noqa: F401  (import before torch so ZeroGPU can patch CUDA)

import random

import gradio as gr

from pipeline import PixelDensePipeline

MAX_SEED = 2**31 - 1
MAX_IMAGES = 4

pipe = PixelDensePipeline.from_pretrained(device="cuda")

TITLE = "PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion"

DESCRIPTION = """
<div style="text-align: center;">
<h1>PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion</h1>
<p><b>NeurIPS 2026</b></p>
<p>
<a href="https://arxiv.org/abs/2610.00483">📄 Paper</a> &nbsp;|&nbsp;
<a href="https://bio.lehanyang.info/PixelDense/">🌐 Project Page</a> &nbsp;|&nbsp;
<a href="https://github.com/Hansxsourse/PixelDense">💻 Code</a> &nbsp;|&nbsp;
<a href="https://huggingface.co/xsourse/PixelDense">🤗 Model</a>
</p>
</div>

PixelDense aligns a pixel-space diffusion transformer to frozen dense-prediction models during training:
DINOv2 and SAM2 supervise a **semantic projection stream**, Depth Anything v2 and Metric3D v2 supervise a
**geometric projection stream**, and an orthogonality penalty keeps the two streams apart. The teachers are
used only during training, so this demo runs the plain 1.1B pixel-space denoiser (PixelGen-XXL architecture)
with the Qwen3-1.7B text encoder and the 25-step Adam LM sampler, generating 512 × 512 images directly in RGB space.
"""

CITATION = """
```bibtex
@inproceedings{yang2026pixeldense,
  title={PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion},
  author={Yang, Lehan and Qi, Daiqing and Zhang, Wenhao and Li, Avery and Yang, Yiqing and Li, Yifan and Kong, Yu and Zheng, Haitian and Zhang, Zhifei and Lin, Zhe and Jampani, Varun and Li, Sheng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
"""

EXAMPLES = [
    "A red fox sitting in a snowy forest at sunrise.",
    "A vintage bicycle leaning against a brick wall covered in ivy.",
    "A cup of cappuccino with latte art on a wooden table next to an open book.",
    "A futuristic city skyline at night, reflected in a calm river.",
    "Three yellow rubber ducks floating in a blue bathtub.",
    "A red book on the left of a green apple on a white table.",
    "A cozy reading nook with a large window, a cat sleeping on an armchair, warm afternoon light.",
    "A lighthouse on a rocky cliff during a storm, waves crashing, dramatic clouds.",
]


def get_duration(prompt, negative_prompt, num_images, num_steps, guidance, timeshift, seed, randomize_seed, progress=None):
    return int(20 + 0.4 * num_steps * num_images)


@spaces.GPU(duration=get_duration)
def generate(
    prompt,
    negative_prompt,
    num_images,
    num_steps,
    guidance,
    timeshift,
    seed,
    randomize_seed,
    progress=gr.Progress(),
):
    # Empty textboxes arrive as None from the web UI.
    prompt = (prompt or "").strip()
    negative_prompt = (negative_prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)
    num_images = int(num_images)
    num_steps = int(num_steps)
    seeds = [(seed + i) % (MAX_SEED + 1) for i in range(num_images)]

    progress(0, desc="Sampling")
    images = pipe(
        [prompt] * num_images,
        seeds,
        negative_prompt=negative_prompt,
        num_steps=num_steps,
        guidance=guidance,
        timeshift=timeshift,
        callback=lambda step, total: progress(step / total, desc=f"Step {step}/{total}"),
    )
    return [(image, f"seed {s}") for image, s in zip(images, seeds)], seed


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                value="",
                label="Prompt",
                placeholder="Describe the image you want to generate",
                lines=3,
                info="Prompts longer than 128 tokens are truncated.",
            )
            run_button = gr.Button("Generate", variant="primary")
            with gr.Accordion("Advanced settings", open=False):
                negative_prompt = gr.Textbox(
                    value="",
                    label="Negative prompt",
                    placeholder="Optional. Used as the unconditional branch of classifier-free guidance.",
                    lines=1,
                )
                num_images = gr.Slider(label="Number of images", minimum=1, maximum=MAX_IMAGES, step=1, value=1)
                num_steps = gr.Slider(label="Sampling steps", minimum=10, maximum=50, step=1, value=pipe.default_steps)
                guidance = gr.Slider(label="Guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=pipe.default_guidance)
                timeshift = gr.Slider(label="Timestep shift", minimum=1.0, maximum=6.0, step=0.1, value=pipe.default_timeshift)
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Samples", columns=2, height=560, format="png", object_fit="contain")
            used_seed = gr.Number(label="Seed", info="Images use seed, seed+1, ...", interactive=False, precision=0)

    gr.Examples(examples=EXAMPLES, inputs=[prompt], label="Example prompts")
    gr.Markdown(CITATION)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=generate,
        inputs=[prompt, negative_prompt, num_images, num_steps, guidance, timeshift, seed, randomize_seed],
        outputs=[gallery, used_seed],
    )


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
