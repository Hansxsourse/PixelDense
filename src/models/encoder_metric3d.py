import os
import importlib.util
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

METRIC3D_MEAN = [0.485, 0.456, 0.406]
METRIC3D_STD = [0.229, 0.224, 0.225]


class Metric3Dv2Encoder(nn.Module):
    def __init__(self, repo_path: str, weight_path: str, base_patch_size: int = 16,
                 extract_layer: int = 17, num_register_tokens: int = 4):
        super().__init__()
        spec = importlib.util.spec_from_file_location(
            "ViT_DINO_reg",
            os.path.join(repo_path, "mono", "model", "backbones", "ViT_DINO_reg.py"),
        )
        vit_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vit_module)

        self.base_patch_size = base_patch_size
        self.extract_layer = extract_layer
        self.num_register_tokens = num_register_tokens
        self.normalize = Normalize(METRIC3D_MEAN, METRIC3D_STD)
        self.encoder = vit_module.vit_large_reg(checkpoint=None)
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)["model_state_dict"]
        self.encoder.load_state_dict({
            k.replace("depth_model.encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("depth_model.encoder.")
        }, strict=False)
        self.encoder.eval()

    def get_features(self, x):
        b, c, h, w = x.shape
        x = self.normalize(x)
        target_h = int(14 * h / self.base_patch_size)
        target_w = int(14 * w / self.base_patch_size)
        x = torch.nn.functional.interpolate(x, (target_h, target_w), mode='bicubic')
        features = self.encoder.get_intermediate_layers(
            x, n=[self.extract_layer], reshape=False, return_class_token=False
        )
        return features[0][:, self.num_register_tokens:]
