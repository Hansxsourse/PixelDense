import sys
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

DA2_MEAN = [0.485, 0.456, 0.406]
DA2_STD = [0.229, 0.224, 0.225]


class DepthAnythingV2Encoder(nn.Module):
    def __init__(self, repo_path: str, weight_path: str, base_patch_size: int = 16, extract_layer: int = 17):
        super().__init__()
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from depth_anything_v2.dinov2 import DINOv2

        self.base_patch_size = base_patch_size
        self.extract_layer = extract_layer
        self.normalize = Normalize(DA2_MEAN, DA2_STD)
        self.encoder = DINOv2(model_name="vitl")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict({
            k.replace("pretrained.", ""): v
            for k, v in state_dict.items()
            if k.startswith("pretrained.")
        })
        self.encoder.eval()

    @torch.compile
    def get_features(self, x):
        b, c, h, w = x.shape
        x = self.normalize(x)
        target_h = int(14 * h / self.base_patch_size)
        target_w = int(14 * w / self.base_patch_size)
        x = torch.nn.functional.interpolate(x, (target_h, target_w), mode='bicubic')
        features = self.encoder.get_intermediate_layers(
            x, n=[self.extract_layer], reshape=False, return_class_token=False
        )
        return features[0]
