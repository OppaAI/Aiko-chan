#!/usr/bin/env python3
"""
Call the deployed MOSS-Image-Edit ComfyUI on Modal from the command line.

    python moss_edit_client.py test.jpg "Change the scene to a cozy cafe interior. Keep the person's face, hair and clothing unchanged." -o out.png

Needs: pip install requests
"""
import argparse
import random
import sys
import time

import requests

DEFAULT_URL = "https://oppa-ai-org--moss-image-edit-ui.modal.run"


def build_workflow(image_name, prompt, negative, seed, steps, cfg, ref_edge, out_prefix,
                   ref_names=()):
    """API-format graph converted from moss_edit_single_reference.json.
    ref_names: extra reference images, wired to reference_0, reference_1, ...
    (the dotted input name comes from the workflow JSON; unverified in API form)."""
    wf = _base_workflow(image_name, prompt, negative, seed, steps, cfg, ref_edge, out_prefix)
    for i, name in enumerate(ref_names):
        node_id = str(20 + i)
        wf[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["5"]["inputs"][f"reference_images.reference_{i}"] = [node_id, 0]
    return wf


def _base_workflow(image_name, prompt, negative, seed, steps, cfg, ref_edge, out_prefix):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "moss_image_sft_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3.5_9b_bf16.safetensors",
                         "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "flux2-vae_small.safetensors"}},
        "4": {"class_type": "ModelPatchLoader",
              "inputs": {"name": "moss_image_edit.safetensors"}},
        "5": {"class_type": "ApplyMossImageEdit",
              "inputs": {"model": ["1", 0], "model_patch": ["4", 0], "vae": ["3", 0],
                         "positive": ["9", 0], "negative": ["10", 0],
                         "image": ["6", 0],
                         "reference_max_edge": ref_edge, "strength": 1.0}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["6", 0], "upscale_method": "lanczos",
                         "megapixels": 1.0, "resolution_steps": 16}},
        "9": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["2", 0], "text": negative}},
        "13": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": ["17", 0], "height": ["17", 1], "batch_size": 1}},
        "14": {"class_type": "KSampler",
               "inputs": {"model": ["18", 0], "positive": ["5", 1], "negative": ["5", 2],
                          "latent_image": ["13", 0], "seed": seed, "steps": steps,
                          "cfg": cfg, "sampler_name": "euler", "scheduler": "normal",
                          "denoise": 1.0}},
        "15": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage",
               "inputs": {"images": ["15", 0], "filename_prefix": out_prefix}},
        "17": {"class_type": "GetImageSize", "inputs": {"image": ["7", 0]}},
        "18": {"class_type": "SkipLayerGuidanceDiTSimple",
               "inputs": {"model": ["5", 0], "double_layers": "", "single_layers": "22",
                          "start_percent": 0.0, "end_percent": 1.0}},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("prompt")
    p.add_argument("-o", "--out", default="out.png")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--ref", action="append", default=[],
                   help="extra reference image (repeat for more, up to 3)")
    p.add_argument("--negative", default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--ref-edge", type=int, default=512)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()

    url = a.url.rstrip("/")
    seed = a.seed if a.seed is not None else random.randint(0, 2**31 - 1)

    if len(a.ref) > 3:
        sys.exit("the edit LoRA supports up to 4 images total (source + 3 references)")

    # 1. upload (first request may wait for a cold start)
    def upload(path):
        with open(path, "rb") as f:
            r = requests.post(f"{url}/upload/image",
                              files={"image": (path.split("/")[-1], f)},
                              data={"overwrite": "true"}, timeout=300)
        r.raise_for_status()
        return r.json()["name"]

    name = upload(a.image)
    ref_names = [upload(p) for p in a.ref]
    print(f"uploaded {name} + {len(ref_names)} reference(s)")

    # 2. queue
    wf = build_workflow(name, a.prompt, a.negative, seed, a.steps, a.cfg,
                        a.ref_edge, "MOSS-Image-Edit/api", ref_names)
    r = requests.post(f"{url}/prompt", json={"prompt": wf}, timeout=60)
    if r.status_code != 200:
        sys.exit(f"queue failed {r.status_code}: {r.text}")
    pid = r.json()["prompt_id"]
    print(f"queued {pid} (seed {seed})")

    # 3. poll
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        h = requests.get(f"{url}/history/{pid}", timeout=30).json()
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                sys.exit(f"workflow error: {status}")
            images = entry.get("outputs", {}).get("16", {}).get("images", [])
            if images:
                break
        time.sleep(3)
    else:
        sys.exit("timed out waiting for result")

    # 4. download
    im = images[0]
    r = requests.get(f"{url}/view",
                     params={"filename": im["filename"], "subfolder": im["subfolder"],
                             "type": im["type"]}, timeout=120)
    r.raise_for_status()
    with open(a.out, "wb") as f:
        f.write(r.content)
    print(f"saved {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
