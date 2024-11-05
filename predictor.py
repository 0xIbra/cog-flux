import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, List

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20

import logging
# from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from fp8.flux_pipeline import FluxPipeline
from fp8.util import LoadedModels
import numpy as np
# from einops import rearrange
from PIL import Image
from torchvision import transforms
from flux.util import load_ae, load_clip, load_flow_model, load_t5, download_weights
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
# from transformers import (
#     CLIPImageProcessor,
#     AutoModelForImageClassification,
#     ViTImageProcessor,
# )

# import fp8.lora_loading as lora_loading

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


class ImagePredictor:
    def __init__(self, flow_model_name: str, compile_fp8: bool = True, compile_bf16: bool = False, disable_fp8: bool = False):
        self.flow_model_name = flow_model_name
        print(f"Booting model {self.flow_model_name}")
        gpu_name = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits").read().strip()
        print("Detected GPU:", gpu_name)

        self.offload = True
        device = "cuda"
        max_length = 256 if self.flow_model_name == "flux-schnell" else 512
        self.t5 = load_t5(device, max_length=max_length)
        self.clip = load_clip(device)
        self.flux = load_flow_model(self.flow_model_name, device="cpu" if self.offload else device)
        self.flux = self.flux.eval()
        self.ae = load_ae(self.flow_model_name, device="cpu" if self.offload else device)
        self.num_steps = 4 if self.flow_model_name == "flux-schnell" else 28
        self.shift = self.flow_model_name != "flux-schnell"
        self.compile_run = False

        shared_models = LoadedModels(flow=None, ae=self.ae, clip=self.clip, t5=self.t5, config=None)

        self.disable_fp8 = disable_fp8 or torch.cuda.get_device_capability() < (8, 9)
        print("Disable fp8:", self.disable_fp8)


        if not self.disable_fp8:
            self.fp8_pipe = FluxPipeline.load_pipeline_from_config_path(
                f"fp8/configs/config-1-{flow_model_name}-L40.json",
                shared_models=shared_models,
            )

            self.fp8_pipe.load_lokr("/home/ubuntu/cog-flux-faster/model-cache/lora/pytorch_lora_weights.safetensors", 1)
            self.fp8_pipe.load_lora("/home/ubuntu/cog-flux-faster/model-cache/lora/8step.safetensors", 1)

            if compile_fp8:
                self.compile_fp8()

        # if compile_bf16:
        #     self.compile_bf16()

    def compile_fp8(self):
        print("compiling fp8 model")
        st = time.time()
        self.fp8_pipe.generate(
            prompt="a cool dog",
            width=1344,
            height=768,
            num_steps=self.num_steps,
            guidance=3,
            seed=123,
            compiling=compile,
        )

        for k in ASPECT_RATIOS:
            print(f"warming kernel for {k}")
            width, height = self.aspect_ratio_to_width_height(k)
            self.fp8_pipe.generate(
                prompt="godzilla!", width=width, height=height, num_steps=4, guidance=3
            )
            self.fp8_pipe.generate(
                prompt="godzilla!",
                width=width // 2,
                height=height // 2,
                num_steps=4,
                guidance=3,
            )

        print("compiled in ", time.time() - st)

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        return ASPECT_RATIOS.get(aspect_ratio)

    def postprocess(
        self,
        images: List[Image.Image],
        output_format: str,
        output_quality: int,
        np_images: Optional[List[Image.Image]] = None,
    ) -> List[Path]:
        import random

        output_paths = []
        for i, img in enumerate(images):
            output_path = f"./{random.randint(0, 100)}out-{i}.{output_format}"
            img.save(output_path)

            output_paths.append(Path(output_path))

        return output_paths

    def fp8_predict(
        self,
        prompt: str,
        num_outputs: int,
        num_inference_steps: int,
        guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
        image: Path = None,  # img2img for flux-dev
        prompt_strength: float = 0.8,
        seed: int = None,
        width: int = 1024,
        height: int = 1024,
    ) -> List[Image.Image]:
        """Run a single prediction on the model"""
        print("running quantized prediction")

        imgs, np_imgs = self.fp8_pipe.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            init_image=image,
            strength=prompt_strength,
            num_images=num_outputs,
        )

        output_format = "png"
        output_quality = 80 # not relevant for .png

        return self.postprocess(
            imgs,
            output_format,
            output_quality,
            np_images=np_imgs,
        )
