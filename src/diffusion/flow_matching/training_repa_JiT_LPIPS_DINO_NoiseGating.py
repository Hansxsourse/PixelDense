import torch
import copy
import timm
from torch.nn import Parameter

from src.utils.no_grad import no_grad, freeze_model
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler
import lpips
import math

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)


class REPATrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            feat_loss_weight: float=0.5,
            lognorm_t=False,
            timeshift=1.0,
            encoder:nn.Module=None,
            align_layer=8,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            P_mean=-0.8,
            P_std=0.8,
            t_eps=0.05,
            num_classes=1000,
            lpips_weight: float=1.0,
            dino_weight: float=1.0,
            percept_t_threshold: float=0.0,
            noise_scale: float=1.0,
            patch_size: int=16,
            percept_ratio: float=1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.encoder = encoder
        freeze_model(self.encoder)
        
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').eval()
        self.lpips_loss_fn.compile()
        freeze_model(self.lpips_loss_fn)
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.lpips_weight = lpips_weight
        self.dino_weight = dino_weight
        self.percept_t_threshold = percept_t_threshold
        self.dino_layers = [11] #list(range(12)) # [2, 5, 8, 11]
        self.noise_scale = noise_scale
        self.num_classes = num_classes
        self.percept_ratio = percept_ratio
        self.cached_percept_weight = 1.0

    def _calculate_adaptive_weight(self, rec_loss, g_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 100.0).detach()
        return d_weight
    
    def compute_dino_loss(self, pred_dino_feats, gt_dino_feats, percept_mask=None):
        cos_losses = {}
        final_cos_loss = 0
        batch_size = pred_dino_feats[0].shape[0]
        for i, (pred_feat, gt_feat) in enumerate(zip(pred_dino_feats, gt_dino_feats)):
            if percept_mask is not None:
                percept_mask = percept_mask.reshape(batch_size, 1, 1)
                cos_sim = (torch.nn.functional.cosine_similarity(pred_feat, gt_feat, dim=-1)*percept_mask).mean(dim=(1,2))
                cos_sim = cos_sim.sum()/percept_mask.sum() if percept_mask.sum()>0.0 else cos_sim.sum()
                cos_loss = 1-cos_sim
            else:
                cos_loss = 1-torch.nn.functional.cosine_similarity(pred_feat, gt_feat, dim=-1).view(batch_size, -1).mean()
            cos_losses[f"inter_cos_{i}"] = cos_loss
            final_cos_loss+=cos_loss
        cos_losses["dino_percept_loss"] = final_cos_loss/len(pred_dino_feats)
        return cos_losses
    
    def compute_lpips_loss(self, pred_img, x, percept_mask=None):
        batch_size, _, height, width = pred_img.shape
        if self.patch_size!=16:
            new_scale = int(height*16//self.patch_size)
            pred_img = torch.nn.functional.interpolate(pred_img, size=(new_scale, new_scale), mode='bilinear', align_corners=False, antialias=True)
            x = torch.nn.functional.interpolate(x, size=(new_scale, new_scale), mode='bilinear', align_corners=False, antialias=True)
        if percept_mask is not None: # discarding steps with much noise
            lpips_loss = (self.lpips_loss_fn(pred_img, x).view(batch_size, -1)*percept_mask).mean(dim=1)
            lpips_loss = lpips_loss.sum()/percept_mask.sum() if percept_mask.sum()>0.0 else lpips_loss.sum()
        else:
            lpips_loss = self.lpips_loss_fn(pred_img, x).mean()
        return lpips_loss
    
    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raw_images = metadata["raw_image"]
        current_step = metadata.get("global_step", 0)
        batch_size, c, height, width = x.shape
        self.lpips_loss_fn.eval()
        if self.lognorm_t:
            base_t = (torch.randn(batch_size, device=x.device, dtype=torch.float32)*self.P_std+self.P_mean).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)
        noise = self.noise_scale*torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)
        
        
        x_t = alpha * x + noise * sigma

        v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        pred_img, src_feature = net(x_t, t, y, return_layer=self.align_layer)
        src_feature = self.proj(src_feature)
        out = (pred_img - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps) # compute v from pred x

        with torch.no_grad():
            dst_features = self.encoder.get_intermediate_feats(raw_images, n=self.dino_layers)
        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_features[-1], dim=-1)
        cos_loss = 1 - cos_sim

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2
        
        ###################  Noise Gating ################
        if self.percept_t_threshold>0.0: # discarding steps with much noise
            percept_mask = (t >= self.percept_t_threshold).float().reshape(batch_size, -1)
        else:
            percept_mask = None
            
        ################### LPIPS loss ###################
        lpips_loss = self.compute_lpips_loss(pred_img, x, percept_mask)
        ###################  DINO loss ###################
        raw_pred_img = (pred_img+1)/2
        pred_feats = self.encoder.get_intermediate_feats(raw_pred_img, n=self.dino_layers)
        dino_losses = self.compute_dino_loss(pred_feats, dst_features, percept_mask)
        
        rec_loss = fm_loss.mean()
        percept_loss = self.lpips_weight*lpips_loss + self.dino_weight*dino_losses["dino_percept_loss"]
        
        if current_step>=400000 and current_step%50==0: # balance fm loss and perceptual loss at later training steps
            last_layer = net.final_layer.linear.weight
            percept_weight = self._calculate_adaptive_weight(rec_loss, percept_loss, last_layer)
            self.cached_percept_weight = 0.8*self.cached_percept_weight + 0.2*percept_weight
      
        final_loss = fm_loss.mean() + self.feat_loss_weight*cos_loss.mean() + self.percept_ratio*self.cached_percept_weight*percept_loss 
        out = dict(
            fm_loss=fm_loss.mean(),
            cos_loss=cos_loss.mean(),
            percept_weight=self.cached_percept_weight,
            lpips_loss=lpips_loss,
            loss=final_loss,
        )
        out.update(dino_losses)
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars)