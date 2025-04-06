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

# function that download the diffusion_4k repo if it doesn't exist
def download_diffusion4_repo():
    """
    Downloads the Diffusion 4k repository if it doesn't exist in the local folder.
    """
    try:
        import subprocess
        print("Downloading Diffusion 4k repository...")
        # Get the absolute path to this module's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define target directory
        target_dir = os.path.join(current_dir, "diffusion_4k")
        # Replace with the actual repository URL
        repo_url = "https://github.com/zhang0jhon/diffusion-4k.git"
        subprocess.check_call(["git", "clone", repo_url, target_dir])
        sys.path.append(target_dir)
        print("Diffusion 4k repository downloaded successfully!")
    except Exception as e:
        print(f"Error downloading Diffusion 4k repository: {e}")
        print("Please download it manually and place it in the ComfyUI-Diffusion-4k directory.")
        sys.exit(1)

# If localfolder diffusion_4k is not found, download the repo
current_dir = os.path.dirname(os.path.abspath(__file__))
diffusion_4k_path = os.path.join(current_dir, "diffusion_4k")
if not os.path.exists(diffusion_4k_path):
    download_diffusion4_repo()


from .diffusion_4k import FluxImageGenerator

# Mappatura dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxImageGenerator": FluxImageGenerator,
}

# Nomi visualizzati per i nodi
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxImageGenerator": "Diffusion 4K",
}