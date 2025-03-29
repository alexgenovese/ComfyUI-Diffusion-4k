from .diffusion_4k import FluxImageGenerator


# Mappatura dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxImageGenerator": FluxImageGenerator,
}

# Nomi visualizzati per i nodi
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxImageGenerator": "Diffusion 4K",
}