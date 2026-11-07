"""
MOSS-Image + Edit LoRA on Modal (ComfyUI UI).

Usage:
    pip install modal && modal setup
    modal run moss_edit_modal.py::download      # one-time: fetch models into a Volume
    modal serve moss_edit_modal.py              # dev URL, hot reload
    modal deploy moss_edit_modal.py             # persistent URL

Then in the ComfyUI UI: Templates -> ComfyUI-Moss-Image-Edit-Native
(single-reference or multi-reference, SFT variant) and run.

UNVERIFIED: I could not read the Comfy-Org/MOSS-Image repo layout, so the
download step prints the file tree and links files by their parent folder
(split_files/diffusion_models, text_encoders, vae). Check the printed tree
and adjust ALLOW_PATTERNS / folder mapping if something is missing.
"""
import os
import subprocess
import modal

GPU = "L40S"  # 48 GB; the int8 SFT build should fit comfortably on smaller GPUs too
COMFY = "/root/comfy/ComfyUI"
VOL_PATH = "/models"

vol = modal.Volume.from_name("moss-image-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("comfy-cli", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Native MOSS support needs a recent core (author verified v0.42.3; v0.41.3 lacks it)
    # Pinned: ComfyUI 0.44.0 removed MOSSImage.pe_embedder, which breaks the
    # Chenkin-x edit node pack. v0.42.3 is the version its README verified.
    .run_commands("comfy --skip-prompt install --nvidia --version 0.42.3")
    .run_commands(
        f"git clone https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit.git {COMFY}/custom_nodes/ComfyUI-Moss-Image-Edit",
        # Also the 'Native' node pack the LoRA repack's card refers to
        f"git clone https://github.com/PedroMarinhoDev/ComfyUI-Moss-Image-Edit-Native.git {COMFY}/custom_nodes/ComfyUI-Moss-Image-Edit-Native || true",
    )
)

app = modal.App("moss-image-edit", image=image)

ALLOW_PATTERNS = ["*sft_int8_convrot*", "*text_encoders*", "*vae*"]


@app.function(volumes={VOL_PATH: vol}, timeout=60 * 60, secrets=[])
def download():
    from huggingface_hub import snapshot_download, hf_hub_download

    base = snapshot_download(
        "Comfy-Org/MOSS-Image",
        local_dir=f"{VOL_PATH}/Comfy-Org-MOSS-Image",
        allow_patterns=ALLOW_PATTERNS,
    )
    print("== Comfy-Org/MOSS-Image tree ==")
    for root, _, files in os.walk(base):
        for f in files:
            print(os.path.relpath(os.path.join(root, f), base))

    os.makedirs(f"{VOL_PATH}/loras", exist_ok=True)
    hf_hub_download(
        "PedroMarinhoDev/Moss-Image-Edit-ComfyUI",
        "moss_image_edit_lora_bf16.safetensors",
        local_dir=f"{VOL_PATH}/loras",
    )
    # Official-keys adapter for the node pack's Load Model Patch route
    os.makedirs(f"{VOL_PATH}/model_patches", exist_ok=True)
    hf_hub_download(
        "chinoll/Moss-Image-Edit",
        "adapter_model.safetensors",
        local_dir=f"{VOL_PATH}/model_patches",
    )
    os.replace(
        f"{VOL_PATH}/model_patches/adapter_model.safetensors",
        f"{VOL_PATH}/model_patches/moss_image_edit.safetensors",
    )
    vol.commit()


def _link_models():
    """Symlink Volume contents into ComfyUI's model folders."""
    base = f"{VOL_PATH}/Comfy-Org-MOSS-Image"
    for folder in ("diffusion_models", "text_encoders", "vae"):
        dst = f"{COMFY}/models/{folder}"
        os.makedirs(dst, exist_ok=True)
        for root, _, files in os.walk(base):
            if os.path.basename(root) == folder:
                for f in files:
                    link = os.path.join(dst, f)
                    if not os.path.exists(link):
                        os.symlink(os.path.join(root, f), link)
    for folder in ("loras", "model_patches"):
        src = f"{VOL_PATH}/{folder}"
        dst = f"{COMFY}/models/{folder}"
        os.makedirs(dst, exist_ok=True)
        if os.path.isdir(src):
            for f in os.listdir(src):
                link = os.path.join(dst, f)
                if f.endswith(".safetensors") and not os.path.exists(link):
                    os.symlink(os.path.join(src, f), link)


@app.function(
    gpu=GPU,
    volumes={VOL_PATH: vol},
    max_containers=1,
    scaledown_window=300,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8188, startup_timeout=180)
def ui():
    _link_models()
    subprocess.Popen(
        f"cd {COMFY} && python main.py --listen 0.0.0.0 --port 8188", shell=True
    )
