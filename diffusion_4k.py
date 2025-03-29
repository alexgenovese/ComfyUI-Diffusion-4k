import torch
import os
import numpy as np
from huggingface_hub import snapshot_download
from .pipeline_flux_4k_wavelet import FluxPipeline

class FluxImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "height": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 64}),
                "width": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "checkpoint_path": ("STRING", {"default": "./checkpoint/flux_wavelet"}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "partitioned": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "generators"

    def check_checkpoint(self, checkpoint_path):
        """Check if the checkpoint exists, download if not."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                # Download the model from Hugging Face
                snapshot_download(
                    repo_id="zhang0jhon/flux_wavelet", 
                    local_dir=checkpoint_path
                )
                print(f"Checkpoint downloaded successfully to {checkpoint_path}")
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
                raise

    def generate(self, prompt, height, width, guidance_scale, num_inference_steps, seed, 
                checkpoint_path="./checkpoint/flux_wavelet", max_sequence_length=512, partitioned=True):
        # Check if checkpoint exists and download if needed
        self.check_checkpoint(checkpoint_path)
        
        # Carica il modello Flux
        pipe = FluxPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        
        # Genera l'immagine
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(seed),
            partitioned=partitioned,
        ).images[0]
        
        # Converti l'immagine nel formato richiesto da ComfyUI
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        return (image_tensor,)
