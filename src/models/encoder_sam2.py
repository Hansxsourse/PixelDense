import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.position_encoding import PositionEmbeddingSine

SAM2_MEAN = [0.485, 0.456, 0.406]
SAM2_STD = [0.229, 0.224, 0.225]


class SAM2Encoder(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        self.normalize = Normalize(SAM2_MEAN, SAM2_STD)
        trunk = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8],
        )
        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256, normalize=True, scale=None, temperature=10000,
            ),
            d_model=256,
            backbone_channel_list=[1152, 576, 288, 144],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )
        self.image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        self.image_encoder.load_state_dict({
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.")
        })
        self.image_encoder.eval()

    @torch.compile
    def get_features(self, x):
        x = self.normalize(x)
        feature = self.image_encoder(x)["vision_features"]
        return feature.flatten(2).transpose(1, 2)
