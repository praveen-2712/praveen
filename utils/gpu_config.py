import torch_directml

def get_nvidia_device():
    """
    Scans available DirectML devices and exclusively selects the NVIDIA GPU,
    bypassing integrated AMD/Intel graphics.
    """
    for i in range(torch_directml.device_count()):
        name = torch_directml.device_name(i).lower()
        if "nvidia" in name or "rtx" in name:
            print(f"[Device Selection] Forced mapping to High-Performance GPU: {torch_directml.device_name(i)}")
            return torch_directml.device(i)
            
    print("[Device Selection] Warning: NVIDIA GPU not found via DirectML. Falling back to default.")
    return torch_directml.device()
