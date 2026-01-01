#@title Utils Code
# %cd /content/ComfyUI

import os, random, time
import torch
import numpy as np
from PIL import Image
import re, uuid
import gradio as gr

from nodes import NODE_CLASS_MAPPINGS

# ----------------------------
# Node setup
# ----------------------------
UNETLoader       = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader       = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader        = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode   = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler         = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode        = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader       = NODE_CLASS_MAPPINGS["LoraLoader"]()

# ----------------------------
# Base model loads (once)
# ----------------------------
with torch.inference_mode():
    unet_base = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip_base = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae       = VAELoader.load_vae("ae.safetensors")[0]

# ----------------------------
# Save helpers
# ----------------------------
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt: str) -> str:
    safe_prompt = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt)[:25]
    uid = uuid.uuid4().hex[:6]
    filename = f"{safe_prompt}_{uid}.png"
    return os.path.join(save_dir, filename)

# ----------------------------
# LoRA utilities (recursive subfolders)
# ----------------------------
LORA_DIR = "./models/loras"

def list_loras():
    if not os.path.isdir(LORA_DIR):
        return ["None"]
    exts = (".safetensors", ".ckpt", ".pt")
    loras = []
    for root, _, files in os.walk(LORA_DIR):
        for f in files:
            if f.lower().endswith(exts):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, LORA_DIR)
                loras.append(rel_path)
    loras.sort(key=str.lower)
    return ["None"] + loras

def apply_loras(model, clip_obj, loras):
    for l in (loras or []):
        name = (l.get("name") or "None").strip()
        if name == "None":
            continue
        sm = float(l.get("strength_model", 0.0))
        sc = float(l.get("strength_clip", 0.0))
        if sm == 0.0 and sc == 0.0:
            continue
        model, clip_obj = LoraLoader.load_lora(model, clip_obj, name, sm, sc)
    return model, clip_obj

# ----------------------------
# Core generation (shared by UI and notebook use)
# ----------------------------
@torch.inference_mode()
def generate(input):
    values = input["input"]

    positive_prompt = values["positive_prompt"]
    negative_prompt = values["negative_prompt"]
    seed            = int(values["seed"])
    steps           = int(values["steps"])
    cfg             = float(values["cfg"])
    sampler_name    = values["sampler_name"]
    scheduler       = values["scheduler"]
    denoise         = float(values["denoise"])
    width           = int(values["width"])
    height          = int(values["height"])
    batch_size      = int(values["batch_size"])
    loras           = values.get("loras", [])

    if seed == 0:
        seed = random.randint(0, 2**64 - 1)

    model, clip_local = apply_loras(unet_base, clip_base, loras)

    positive = CLIPTextEncode.encode(clip_local, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip_local, negative_prompt)[0]

    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    samples = KSampler.sample(
        model, seed, steps, cfg,
        sampler_name, scheduler,
        positive, negative,
        latent_image,
        denoise=denoise
    )[0]

    decoded = VAEDecode.decode(vae, samples)[0].detach()
    save_path = get_save_path(positive_prompt)

    img = np.array(decoded * 255, dtype=np.uint8)[0]
    Image.fromarray(img).save(save_path)

    return save_path, seed

# ----------------------------
# Gradio UI wrapper
# ----------------------------
def generate_ui(
    positive_prompt,
    negative_prompt,
    aspect_ratio,
    seed,
    steps,
    cfg,
    denoise,

    lora1, lora1_sm, lora1_sc,
    lora2, lora2_sm, lora2_sc,
    lora3, lora3_sm, lora3_sc,
    lora4, lora4_sm, lora4_sc,

    batch_size=1,
    sampler_name="euler",
    scheduler="simple"
):
    width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "loras": [
                {"name": lora1, "strength_model": float(lora1_sm), "strength_clip": float(lora1_sc)},
                {"name": lora2, "strength_model": float(lora2_sm), "strength_clip": float(lora2_sc)},
                {"name": lora3, "strength_model": float(lora3_sm), "strength_clip": float(lora3_sc)},
                {"name": lora4, "strength_model": float(lora4_sm), "strength_clip": float(lora4_sc)},
            ],
        }
    }

    image_path, used_seed = generate(input_data)
    return image_path, image_path, str(used_seed)

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_POSITIVE = """Insert your prompt here"""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus,pixelated"""

ASPECTS = [
    "1024x1024 (1:1)", "1080x1080 (1:1)",
    "720x1280 (9:16)", "832x1248 (2:3)", "864x1152 (3:4)", "896x1152 (7:9)",
    "1080x1440 (3:4)", "1080x1920 (9:16)",
    "1280x720 (16:9)", "1344x576 (21:9)", "1920x1080 (16:9)",
]

# ----------------------------
# Build UI (safe to import; does not launch)
# ----------------------------
custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"
loras_list = list_loras()

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
    <h1 style="font-size:2.5em; margin-bottom:10px;">Z-Image-Turbo</h1>
    <a href="https://github.com/Tongyi-MAI/Z-Image" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white"
             style="height:15px;">
    </a>
</div>
""")

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

            with gr.Accordion("LoRAs (up to 4)", open=False):
                lora1 = gr.Dropdown(loras_list, value="None", label="LoRA 1")
                lora1_sm = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 1 UNet strength")
                lora1_sc = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 1 CLIP strength")

                lora2 = gr.Dropdown(loras_list, value="None", label="LoRA 2")
                lora2_sm = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 2 UNet strength")
                lora2_sc = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 2 CLIP strength")

                lora3 = gr.Dropdown(loras_list, value="None", label="LoRA 3")
                lora3_sm = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 3 UNet strength")
                lora3_sc = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 3 CLIP strength")

                lora4 = gr.Dropdown(loras_list, value="None", label="LoRA 4")
                lora4_sm = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 4 UNet strength")
                lora4_sc = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 4 CLIP strength")

            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="720x1280 (9:16)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")

            with gr.Row():
                run = gr.Button("🚀 Generate", variant="primary")

            with gr.Accordion("Image Settings", open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)

        with gr.Column():
            download_image = gr.File(label="Download Image")
            output_img = gr.Image(label="Generated Image", height=480)
            used_seed = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)

    run.click(
        fn=generate_ui,
        inputs=[
            positive, negative, aspect, seed, steps, cfg, denoise,
            lora1, lora1_sm, lora1_sc,
            lora2, lora2_sm, lora2_sc,
            lora3, lora3_sm, lora3_sc,
            lora4, lora4_sm, lora4_sc,
        ],
        outputs=[download_image, output_img, used_seed]
    )

def main():
    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    main()
