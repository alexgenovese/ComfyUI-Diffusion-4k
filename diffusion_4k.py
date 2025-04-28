import torch
import os, sys

# Get the absolute path to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the diffusion_4k directory to the Python path
diffusion_4k_path = os.path.join(script_dir, "diffusion_4k")
sys.path.append(diffusion_4k_path)

import numpy as np
from huggingface_hub import snapshot_download
from pipeline_flux_4k_wavelet import FluxPipeline

# Import ComfyUI's folder_paths module
import folder_paths

class FluxImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        # Use ComfyUI's folder_paths to get the proper checkpoints directory
        checkpoint_list = []
        
        # Get the default path for flux model checkpoints
        flux_dir = os.path.join(folder_paths.models_dir, "checkpoints")
        os.makedirs(flux_dir, exist_ok=True)
        
        # Scan for available checkpoints
        for item in os.listdir(flux_dir):
            item_path = os.path.join(flux_dir, item)
            if os.path.isdir(item_path) or item.lower().endswith(('.safetensors', '.sft')):
                checkpoint_list.append(item)
                
        # If no checkpoints found, add a default option
        if not checkpoint_list:
            checkpoint_list = ["flux_wavelet"]
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "checkpoint": (checkpoint_list, {"default": checkpoint_list[0]}),
                "height": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 64}),
                "width": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "partitioned": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "generators"

    def check_checkpoint(self, checkpoint_name):
        """Check if the checkpoint exists, download if not."""
        checkpoint_path = os.path.join(folder_paths.models_dir, "checkpoints", checkpoint_name)
        
        # Handle both file-based and directory-based checkpoints
        if os.path.exists(checkpoint_path):
            if os.path.isdir(checkpoint_path):
                return checkpoint_path
            else:
                # If it's a file, return the directory containing the file
                return os.path.dirname(checkpoint_path)
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
            try:
                # For file-based checkpoints, use the base name (without extension) for the folder
                if checkpoint_name.lower().endswith(('.safetensors', '.sft')):
                    base_name = os.path.splitext(checkpoint_name)[0]
                else:
                    base_name = checkpoint_name
                
                download_path = os.path.join(folder_paths.models_dir, "checkpoints", "flux", base_name)
                
                # Create directory if it doesn't exist
                os.makedirs(download_path, exist_ok=True)
                
                # Download the model from Hugging Face
                snapshot_download(
                    repo_id="zhang0jhon/flux_wavelet", 
                    local_dir=download_path
                )
                print(f"Checkpoint downloaded successfully to {download_path}")
                return download_path
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
                raise

    def generate(self, prompt, checkpoint, height, width, guidance_scale, num_inference_steps, seed,
                max_sequence_length=512, partitioned=True):
        # Get the full checkpoint path and check if it exists
        checkpoint_path = self.check_checkpoint(checkpoint)
        
        # Load the Flux model
        pipe = FluxPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        
        # Generate the image
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
        
        # Convert the image to the format required by ComfyUI
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        return (image_tensor,)
