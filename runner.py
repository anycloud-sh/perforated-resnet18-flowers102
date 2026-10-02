"""Run Perforated AI's upstream Flowers-102 comparison with pinned model files."""

import hashlib
import json
import math
import os

import torch
import torch.nn as nn
import torchvision
from huggingface_hub import hf_hub_download
from perforatedai import library_perforatedai as LPA
from perforatedai import network_perforatedai as NPA
from perforatedai import utils_perforatedai as UPA
from safetensors.torch import load_file

from upstream import flowers_comparison as upstream


MODEL_ID = "perforated-ai/resnet-18-perforated-cascor"
MODEL_REVISION = "8a59acfc5285e72257e8d322da51bf486c3ce060"
MODEL_SHA256 = "114d3ed72896110ff229bd5a7c4ae8e7d048eb185ba1cdeb75cd76557d8674bc"


def pinned_file(name):
    return hf_hub_download(repo_id=MODEL_ID, filename=name, revision=MODEL_REVISION)


def build_pinned_perforated_model(num_classes):
    """Use the upstream 3.2.0 loader steps with an explicit model revision."""
    base_model = torchvision.models.get_model("resnet18", weights=None, num_classes=1000)
    model = LPA.ResNetPAIPreFC(base_model)

    with open(pinned_file("config.json"), encoding="utf-8") as config_file:
        config = json.load(config_file)
    UPA.set_gpa_config(config["pai_config"])

    model_path = pinned_file("model.safetensors")
    with open(model_path, "rb") as weights_file:
        digest = hashlib.file_digest(weights_file, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"Model SHA-256 mismatch: {digest}")

    model = NPA.convert_network(model)
    model = NPA.load_pai_model_from_dict(model, load_file(model_path))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def verify_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU artifact")
    left = torch.randn((128, 128), device="cuda")
    right = torch.randn((128, 128), device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    checksum = result.sum().item()
    if not math.isfinite(checksum):
        raise RuntimeError("CUDA kernel returned a non-finite result")
    print(
        f"CUDA_PROOF device={torch.cuda.get_device_name(0)!r} "
        f"allocated_bytes={torch.cuda.memory_allocated()} "
        f"matmul_checksum={checksum:.6f} synchronized=true",
        flush=True,
    )


if __name__ == "__main__":
    verify_cuda()
    print(f"MODEL_REVISION={MODEL_REVISION} MODEL_SHA256={MODEL_SHA256}", flush=True)
    upstream.build_perforated_cascor_resnet18 = build_pinned_perforated_model
    if os.environ.get("SPOT_RESUME") == "1":
        from spot_checkpoint import run_training

        upstream.run_training = run_training
    upstream.main()
