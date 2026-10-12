import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from typing import Callable

from src.utils.no_grad import freeze_model
from src.diffusion.base.training import BaseTrainer
from src.diffusion.base.scheduling import BaseScheduler


def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)


class PixelDenseTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn: Callable = constant,
            lognorm_t=False,
            timeshift=1.0,
            encoder: nn.Module = None,
            sam_encoder: nn.Module = None,
            depth_repa_encoder: nn.Module = None,
            normal_repa_encoder: nn.Module = None,
            align_layer=8,
            proj_denoiser_dim=256,
            factored_semantic_dim=768,
            factored_geometric_dim=1024,
            proj_encoder_dim=768,
            proj_sam_encoder_dim=256,
            proj_depth_repa_encoder_dim=1024,
            proj_normal_repa_encoder_dim=1024,
            feat_loss_weight: float = 0.5,
            sam_loss_weight: float = 0.3,
            depth_repa_loss_weight: float = 0.3,
            normal_repa_loss_weight: float = 0.3,
            orth_reg_weight: float = 0.01,
            repa_curriculum_decay_steps: int = 300000,
            P_mean=-0.8,
            P_std=0.8,
            t_eps=0.05,
            lpips_weight: float = 1.0,
            dino_weight: float = 1.0,
            percept_t_threshold: float = 0.0,
            noise_scale: float = 1.0,
            patch_size: int = 16,
            percept_ratio: float = 1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.align_layer = align_layer
        self.feat_loss_weight = feat_loss_weight
        self.sam_loss_weight = sam_loss_weight
        self.depth_repa_loss_weight = depth_repa_loss_weight
        self.normal_repa_loss_weight = normal_repa_loss_weight
        self.orth_reg_weight = orth_reg_weight
        self.repa_curriculum_decay_steps = repa_curriculum_decay_steps

        self.encoder = encoder
        freeze_model(self.encoder)

        self.lpips_loss_fn = lpips.LPIPS(net='vgg').eval()
        self.lpips_loss_fn.compile()
        freeze_model(self.lpips_loss_fn)

        self.sam_encoder = sam_encoder
        freeze_model(self.sam_encoder)
        self.depth_repa_encoder = depth_repa_encoder
        freeze_model(self.depth_repa_encoder)
        self.normal_repa_encoder = normal_repa_encoder
        freeze_model(self.normal_repa_encoder)

        self.proj_semantic = nn.Sequential(
            nn.Linear(proj_denoiser_dim, factored_semantic_dim),
            nn.LayerNorm(factored_semantic_dim),
            nn.SiLU(),
            nn.Linear(factored_semantic_dim, factored_semantic_dim),
        )
        self.proj_geometric = nn.Sequential(
            nn.Linear(proj_denoiser_dim, factored_geometric_dim),
            nn.LayerNorm(factored_geometric_dim),
            nn.SiLU(),
            nn.Linear(factored_geometric_dim, factored_geometric_dim),
        )
        self.head_dino = nn.Linear(factored_semantic_dim, proj_encoder_dim)
        self.head_sam = nn.Linear(factored_semantic_dim, proj_sam_encoder_dim)
        self.head_depth = nn.Linear(factored_geometric_dim, proj_depth_repa_encoder_dim)
        self.head_normal = nn.Linear(factored_geometric_dim, proj_normal_repa_encoder_dim)

        self.patch_size = patch_size
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.lpips_weight = lpips_weight
        self.dino_weight = dino_weight
        self.percept_t_threshold = percept_t_threshold
        self.dino_layers = [11]
        self.noise_scale = noise_scale
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
        if percept_mask is not None:
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
        sigma = self.scheduler.sigma(t)

        x_t = alpha * x + noise * sigma

        v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        pred_img, src_feature = net(x_t, t, y, return_layer=self.align_layer)
        out = (pred_img - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        with torch.no_grad():
            dst_features = self.encoder.get_intermediate_feats(raw_images, n=self.dino_layers)
            dst_sam = self.sam_encoder.get_features(raw_images)
            dst_depth = self.depth_repa_encoder.get_features(raw_images)
            dst_normal = self.normal_repa_encoder.get_features(raw_images)
        dst_dino = F.normalize(dst_features[-1], dim=-1)
        dst_sam = F.normalize(dst_sam, dim=-1)
        dst_depth = F.normalize(dst_depth, dim=-1)
        dst_normal = F.normalize(dst_normal, dim=-1)

        h_sem = self.proj_semantic(src_feature)
        h_geo = self.proj_geometric(src_feature)
        cos_loss = 1 - F.cosine_similarity(self.head_dino(h_sem), dst_dino, dim=-1)
        sam_cos_loss = 1 - F.cosine_similarity(self.head_sam(h_sem), dst_sam, dim=-1)
        depth_repa_cos_loss = 1 - F.cosine_similarity(self.head_depth(h_geo), dst_depth, dim=-1)
        normal_repa_cos_loss = 1 - F.cosine_similarity(self.head_normal(h_geo), dst_normal, dim=-1)

        align_loss = (
            self.feat_loss_weight*cos_loss.mean()
            + self.sam_loss_weight*sam_cos_loss.mean()
            + self.depth_repa_loss_weight*depth_repa_cos_loss.mean()
            + self.normal_repa_loss_weight*normal_repa_cos_loss.mean()
        )
        if self.repa_curriculum_decay_steps > 0:
            align_loss = align_loss*max(0.0, 1.0 - current_step/self.repa_curriculum_decay_steps)

        W_sem = self.proj_semantic[0].weight
        W_geo = self.proj_geometric[0].weight
        orth_loss = self.orth_reg_weight*F.cosine_similarity(W_sem.unsqueeze(1), W_geo.unsqueeze(0), dim=-1).abs().mean()

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2

        if self.percept_t_threshold>0.0:
            percept_mask = (t >= self.percept_t_threshold).float().reshape(batch_size, -1)
        else:
            percept_mask = None

        lpips_loss = self.compute_lpips_loss(pred_img, x, percept_mask)
        raw_pred_img = (pred_img+1)/2
        pred_feats = self.encoder.get_intermediate_feats(raw_pred_img, n=self.dino_layers)
        dino_losses = self.compute_dino_loss(pred_feats, dst_features, percept_mask)

        rec_loss = fm_loss.mean()
        percept_loss = self.lpips_weight*lpips_loss + self.dino_weight*dino_losses["dino_percept_loss"]

        if current_step>=400000 and current_step%50==0:
            last_layer = net.final_layer.linear.weight
            percept_weight = self._calculate_adaptive_weight(rec_loss, percept_loss, last_layer)
            self.cached_percept_weight = 0.8*self.cached_percept_weight + 0.2*percept_weight

        final_loss = rec_loss + align_loss + orth_loss + self.percept_ratio*self.cached_percept_weight*percept_loss
        out = dict(
            fm_loss=rec_loss,
            cos_loss=cos_loss.mean(),
            sam_cos_loss=sam_cos_loss.mean(),
            depth_repa_cos_loss=depth_repa_cos_loss.mean(),
            normal_repa_cos_loss=normal_repa_cos_loss.mean(),
            align_loss=align_loss,
            orth_loss=orth_loss,
            percept_weight=self.cached_percept_weight,
            lpips_loss=lpips_loss,
            loss=final_loss,
        )
        out.update(dino_losses)
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj_semantic.state_dict(
            destination=destination,
            prefix=prefix + "proj_semantic.",
            keep_vars=keep_vars)
        self.proj_geometric.state_dict(
            destination=destination,
            prefix=prefix + "proj_geometric.",
            keep_vars=keep_vars)
        self.head_dino.state_dict(
            destination=destination,
            prefix=prefix + "head_dino.",
            keep_vars=keep_vars)
        self.head_sam.state_dict(
            destination=destination,
            prefix=prefix + "head_sam.",
            keep_vars=keep_vars)
        self.head_depth.state_dict(
            destination=destination,
            prefix=prefix + "head_depth.",
            keep_vars=keep_vars)
        self.head_normal.state_dict(
            destination=destination,
            prefix=prefix + "head_normal.",
            keep_vars=keep_vars)
