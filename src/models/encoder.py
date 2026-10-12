import copy
import torch
import torch.nn as nn
import timm
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import os

class IndentityMapping(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        return x

class DINOv2(nn.Module):
    def __init__(self, weight_path:str=None, base_patch_size=16):
        super(DINOv2, self).__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # need to visit github for each run.
        # self.encoder = torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14', source="local")

        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()
        self.base_patch_size = base_patch_size
        self.encoder.compile()

    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)), mode='bicubic')
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        return feature
    
    @torch.compile
    def get_intermediate_feats(self, x, resize=True, n=[2, 5, 8, 11], reshape=False, return_class_token=False):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        
        if resize:
            target_h = int(14 * h / self.base_patch_size)
            target_w = int(14 * w / self.base_patch_size)
            x = torch.nn.functional.interpolate(x, (target_h, target_w), mode='bicubic')

        features = self.encoder.get_intermediate_layers(x, n=n, reshape=reshape, return_class_token=return_class_token)
        return features
    
    def forward_with_cls(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)), mode='bicubic')
        out = self.encoder.forward_features(x)
        feature, cls_token = out['x_norm_patchtokens'], out["x_norm_clstoken"].unsqueeze(1)
        return feature, cls_token


from transformers import CLIPModel, CLIPTokenizer
class CLIP(nn.Module):
    def __init__(self, weight_path:str):
        super(CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained(weight_path).to(torch.bfloat16)
        self.tokenizer = CLIPTokenizer.from_pretrained(weight_path)
        self.height = self.model.config.vision_config.image_size
        self.width = self.model.config.vision_config.image_size

        self.model.vision_model.compile()
        self.model.text_model.compile()
    def forward(self, x, text, resize=True):
        tokens = self.tokenizer(text, truncation=True, return_tensors='pt', padding="max_length", max_length=self.tokenizer.model_max_length).input_ids.cuda()
        text_output = self.model.text_model(input_ids=tokens).last_hidden_state
        text_output = self.model.text_projection(text_output)
        text_output = torch.nn.functional.normalize(text_output, dim=-1, p=2)
        if resize:
            x = torch.nn.functional.interpolate(x, (self.height, self.width), mode='bicubic')
        x = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)(x)
        vision_output = self.model.vision_model(x).last_hidden_state[:, 1:]
        vision_output = self.model.visual_projection(vision_output)
        vision_output = torch.nn.functional.normalize(vision_output, dim=-1, p=2)
        output = torch.bmm(vision_output, text_output.transpose(1, 2))
        return output

from transformers import SiglipModel, GemmaTokenizer, SiglipTokenizer
class SigLIP(nn.Module):
    def __init__(self, weight_path:str):
        super(SigLIP, self).__init__()
        if "siglip2" in weight_path:
            self.tokenizer = GemmaTokenizer.from_pretrained(weight_path)
        else:
            self.tokenizer = SiglipTokenizer.from_pretrained(weight_path)
        self.model = SiglipModel.from_pretrained(weight_path).to(torch.bfloat16)

        self.mean = 0.5
        self.std = 0.5

        self.model.vision_model.compile()
        self.model.text_model.compile()
    def forward(self, x, text, resize=True):
        tokens = self.tokenizer(text, truncation=True, return_tensors='pt', padding="max_length", max_length=64).input_ids.cuda()
        text_output = self.model.text_model(input_ids=tokens).last_hidden_state
        text_output = torch.nn.functional.normalize(text_output, dim=-1, p=2)
        if resize:
            x = torch.nn.functional.interpolate(x, (self.height, self.width), mode='bicubic')
        x = (x - self.mean)/self.std
        vision_output = self.model.vision_model(x).last_hidden_state
        vision_output = torch.nn.functional.normalize(vision_output, dim=-1, p=2)
        output = torch.bmm(vision_output, text_output.transpose(1, 2))
        return output

from transformers import SiglipVisionModel
class SigLIPVision(nn.Module):
    def __init__(self, weight_path:str, base_patch_size=16):
        super(SigLIPVision, self).__init__()
        self.model = SiglipVisionModel.from_pretrained(weight_path).to(torch.bfloat16)
        self.height = self.model.config.image_size
        self.width = self.model.config.image_size
        self.patch_size = self.model.config.patch_size
        self.base_patch_size = base_patch_size
        self.model.compile()
        self.mean = 0.5
        self.std = 0.5
    def forward(self, x, resize=True):
        if resize:
            h, w = x.shape[-2:]
            new_h = int(self.patch_size * h / self.base_patch_size)
            new_w = int(self.patch_size * w / self.base_patch_size)
            x = torch.nn.functional.interpolate(x, (new_h, new_w), mode='bicubic')
        x = (x - self.mean)/self.std
        vision_output = self.model.vision_model(x).last_hidden_state
        return vision_output
    

import torch.nn as nn
from torchvision import models
from collections import namedtuple
import os

# 官方 LPIPS (VGG) 权重下载地址
LPIPS_VGG_WEIGHTS_URL = "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth"

def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # ImageNet normalization statistics
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # 加载 torchvision 的预训练 VGG16
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True, pretrained=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        
        if pretrained:
            self.load_from_pretrained()
            
        # 冻结参数，因为通常只作为 Loss 使用
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        """
        自动下载并加载官方权重
        """
        try:
            print(f"Loading LPIPS weights from {LPIPS_VGG_WEIGHTS_URL}...")
            # 使用 torch.hub 自动下载并缓存
            state_dict = torch.hub.load_state_dict_from_url(
                LPIPS_VGG_WEIGHTS_URL, 
                progress=True, 
                map_location=torch.device("cpu")
            )
            self.load_state_dict(state_dict, strict=False)
            print("LPIPS weights loaded successfully.")
        except Exception as e:
            print(f"Error loading LPIPS weights: {e}")
            print("Running without trained linear weights (NOT RECOMMENDED for metric computation).")

    def forward(self, input, target):
        # input, target 应该是范围在 [-1, 1] 的 tensor
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

    def forward_with_feats(self, input, target):
        # input, target 应该是范围在 [-1, 1] 的 tensor
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val, outs0, outs1