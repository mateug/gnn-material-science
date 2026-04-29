"""Utility module for loading an M3GNet model from matgl."""

import os
from typing import Optional

import torch
import torch.nn as nn


def _import_m3gnet_class():
    try:
        from matgl.models import M3GNet
        return M3GNet
    except (ImportError, FileNotFoundError, OSError):
        try:
            from matgl.models._m3gnet import M3GNet
            return M3GNet
        except (ImportError, FileNotFoundError, OSError) as exc_inner:
            raise ImportError(
                "M3GNet cannot be imported because the required DGL backend is missing or not available. "
                "Install the appropriate DGL package for your Python/PyTorch version, then restart the kernel. "
                "If you only need a working model interface, use model_type='GCNN'."
            ) from exc_inner


def load_pretrained_m3gnet(model_name: str = "M3GNet-MP-2018.6.1-Eform"):
    """Load a pre-trained M3GNet model from matgl."""
    M3GNet = _import_m3gnet_class()
    return M3GNet.from_pretrained(model_name)


def load_model(
        n_node_features=None,
        pdropout=0,
        device='cpu',
        model_name=None,
        mode='train',
        pretrained_name: Optional[str] = "M3GNet-MP-2018.6.1-Eform"
):
    """Load M3GNet with a runtime-compatible loader signature."""
    if pretrained_name is not None:
        model = load_pretrained_m3gnet(pretrained_name)
    else:
        M3GNet = _import_m3gnet_class()
        model = M3GNet()

    if model_name is not None and os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

    model = model.to(device)
    if mode == 'eval':
        model.eval()
    else:
        model.train()

    model = nn.DataParallel(model)
    return model


if __name__ == "__main__":
    model = load_pretrained_m3gnet()
    print(f"Loaded M3GNet model: {model.__class__.__name__}")
