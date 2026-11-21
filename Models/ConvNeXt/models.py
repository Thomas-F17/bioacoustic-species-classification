"""
This is the modified src/models.py. The only thing added to the original paper's github code implemention is
ConvNeXt-Base via timm.
So, only change: import timm, add convnext_base to hidden_dim_dict, add elif block.
"""
import torchvision
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import json
import os
import timm


hidden_dim_dict = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "vit": 768,
    "mobilenet": 1280,
    "convnext_base": 1024,
}

def get_model(model_name, output_dim, pretrained=True, get_last_dim=False):
    last_dim = hidden_dim_dict[model_name]
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "vit":
        model = torchvision.models.vit_b_16(pretrained=pretrained)
        model.heads.head = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "mobilenet":
        model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "convnext_base":
        model = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=output_dim,
        )
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

    if not get_last_dim:
        return model
    else:
        return model, last_dim