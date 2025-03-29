import os, sys

# Installazione automatica delle dipendenze
def install_dependencies():
    try:
        import diffusers
    except ImportError:
        print("Installazione delle dipendenze per Diffusion 4k...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dipendenze installate con successo!")

# Esegui l'installazione delle dipendenze quando il modulo viene caricato
install_dependencies()


from .diffusion_4k import FluxImageGenerator

# Mappatura dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxImageGenerator": FluxImageGenerator,
}

# Nomi visualizzati per i nodi
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxImageGenerator": "Diffusion 4K",
}