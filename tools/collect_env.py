"""
Script used to generate environment information for the purpose of
creating bug reports. See `.github/ISSUE_TEMPLATE/bug_report.md`
"""

import platform
import sys
import importlib

def get_version(pkg_name):
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "None"

def get_torch_hardware_info():
    try:
        import torch
        cuda_devices = []
        amd_devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                if "AMD" in name.upper():
                    amd_devices.append(name)
                else:
                    cuda_devices.append(name)
        return cuda_devices, amd_devices
    except ImportError:
        return [], []

def collect_environment_info():
    cuda_devices, amd_devices = get_torch_hardware_info()

    info = {
        "Operating System": platform.platform(),
        "Python Version": sys.version.replace("\n", " "),
        "llm-compressor Version": get_version("llmcompressor"),
        "compressed-tensors Version": get_version("compressed_tensors"),
        "transformers Version": get_version("transformers"),
        "torch Version": get_version("torch"),
        "CUDA Devices": cuda_devices if cuda_devices else "None",
        "AMD Devices": amd_devices if amd_devices else "None",
    }

    print("### Environment Information ###")
    for key, value in info.items():
        print(f"{key}: `{value}`")

if __name__ == "__main__":
    collect_environment_info()
