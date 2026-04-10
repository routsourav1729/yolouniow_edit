"""Simulate BNContrastiveHead math to understand embedding spaces."""
import torch
import torch.nn.functional as F
import numpy as np

CKPT = 'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task1/best_owod_Both_epoch_20.pth'

ckpt = torch.load(CKPT, map_location='cpu')
sd = ckpt['state_dict']

# Text embeddings from checkpoint  
txt = sd['embeddings']  # (22, 512)
txt_known = txt[:20]    # 20 T1 known classes
txt_unk = txt[-2]       # unknown
txt_anchor = txt[-1]    # anchor

txt_norm = F.normalize(txt_known, dim=-1, p=2)

# BN stats level 0
rmean = sd['bbox_head.head_module.one2many_cls_contrasts.0.norm.running_mean']  
rvar = sd['bbox_head.head_module.one2many_cls_contrasts.0.norm.running_var']
gamma = sd['bbox_head.head_module.one2many_cls_contrasts.0.norm.weight']
beta = sd['bbox_head.head_module.one2many_cls_contrasts.0.norm.bias']
logit_scale = sd['bbox_head.head_module.one2many_cls_contrasts.0.logit_scale']
bias_val = sd['bbox_head.head_module.one2many_cls_contrasts.0.bias']

print(f"logit_scale={logit_scale.item():.4f}, exp={logit_scale.exp().item():.4f}")
print(f"bias={bias_val.item():.4f}")
print(f"sigmoid(bias alone) = {torch.sigmoid(bias_val).item():.8f}")
print(f"→ ALL logits start from {bias_val.item():.1f}, need dot-product ≈ {-bias_val.item():.0f} to reach sigmoid=0.5")

# ALL text embeddings including unk and anchor
txt_all = txt  # (22, 512)
txt_all_norm = F.normalize(txt_all, dim=-1, p=2)

# Simulate different anchor types
torch.manual_seed(42)
x_bg = rmean + 0.1 * torch.randn(512)  # background noise
x_obj0 = rmean + 0.5 * txt_known[0]     # class-0 like object
x_mean = rmean.clone()                    # exactly the mean

for label, x in [("background (noise)", x_bg), 
                  ("class-0 object", x_obj0), 
                  ("exactly running_mean", x_mean)]:
    # RAW cosine sim
    raw_cos_known = F.cosine_similarity(x.unsqueeze(0), txt_known, dim=-1)
    raw_cos_unk = F.cosine_similarity(x, txt_unk, dim=0)
    raw_cos_anchor = F.cosine_similarity(x, txt_anchor, dim=0)
    
    # BN(x)
    bn_x = gamma * (x - rmean) / (rvar + 1e-5).sqrt() + beta
    
    # BN cosine sim  
    bn_cos_known = F.cosine_similarity(F.normalize(bn_x, dim=0).unsqueeze(0), txt_norm, dim=-1)
    
    # Full logit (ALL classes including unk and anchor)
    logit_all = (bn_x.unsqueeze(0) @ txt_all_norm.T).squeeze(0) * logit_scale.exp() + bias_val
    prob_all = logit_all.sigmoid()
    
    # Known vs unknown vs anchor
    logit_known = logit_all[:20]
    prob_known = prob_all[:20]
    logit_unk = logit_all[-2]
    prob_unk = prob_all[-2]
    logit_anchor = logit_all[-1]
    prob_anchor = prob_all[-1]
    
    print(f"\n{'='*50}")
    print(f"  {label}  ||x||={x.norm():.2f}")
    print(f"{'='*50}")
    print(f"  RAW cosine (x vs known):  max={raw_cos_known.max():.4f}, argmax={raw_cos_known.argmax()}")
    print(f"  RAW cosine (x vs unk):    {raw_cos_unk:.4f}")
    print(f"  RAW cosine (x vs anchor): {raw_cos_anchor:.4f}")
    print(f"  ||BN(x)||={bn_x.norm():.2f}")
    print(f"  BN cosine (bn_x vs known): max={bn_cos_known.max():.4f}, argmax={bn_cos_known.argmax()}")
    print(f"  LOGIT known: max={logit_known.max():.2f} (class {logit_known.argmax()}), sigmoid={prob_known.max():.6f}")
    print(f"  LOGIT unknown: {logit_unk:.2f}, sigmoid={prob_unk:.6f}")
    print(f"  LOGIT anchor:  {logit_anchor:.2f}, sigmoid={prob_anchor:.6f}")
    print(f"  Gatekeeper: anchor_prob ({prob_anchor:.6f}) > max_known_prob ({prob_known.max():.6f})? {prob_anchor > prob_known.max()}")

print(f"\n{'='*60}")
print(f"ANALYSIS: WHY RAW COSINE SIM IS NEAR ZERO")
print(f"{'='*60}")
print(f"""
The BNContrastiveHead does NOT use cosine similarity at all.

The forward path is:
  cls_embed = cls_pred(img_feat)          # Linear projection, (B, 512, H, W)
  bn_embed = BatchNorm(cls_embed)         # Centering + scaling per channel
  logit = bn_embed · L2norm(txt) * exp(scale) + bias

BN = gamma * (x - running_mean) / sqrt(running_var + eps) + beta

Key numbers from T1 checkpoint:
  ||running_mean|| = {rmean.norm():.2f}     (the common direction of ALL features)
  sqrt(mean(var))  = {rvar.mean().sqrt():.4f}   (typical deviation from mean)
  
  Signal/Noise ratio: {rvar.mean().sqrt() / rmean.norm():.4f}
  
  When you L2-normalize raw cls_embed:
    L2norm(cls_embed) ≈ L2norm(running_mean) for ALL anchors
    because running_mean dominates the raw feature
  
  So cosine_sim(L2norm(raw), L2norm(txt)) ≈ same constant for ALL anchors
  → no discriminative signal → w_r ≈ constant → WAPR is a no-op

The model works because BN REMOVES the common component first:
  BN(x) = gamma * (x - mean) / sqrt(var) + beta
  This is like PCA whitening — it removes the shared direction
  and amplifies the class-specific residual signal.
""")

print(f"{'='*60}")
print(f"FIX: Cache BN(cls_embed) INSTEAD of raw cls_embed")
print(f"{'='*60}")
print(f"""
In yolov10_world_head.py, we currently cache:
  self._cached_cls_embeds_one2many.append(cls_embed.detach())

We should cache the BN OUTPUT instead:
  bn_embed = cls_contrast.norm(cls_embed)  # just the BN part
  self._cached_cls_embeds_one2many.append(bn_embed.detach())

Then in wapr.py, cosine_sim(BN(cls_embed), txt_feats) will actually 
discriminate between known and unknown anchors because:
  - For a 'car' anchor: BN removes mean, residual aligns with 'car' text → high cos sim
  - For genuinely unknown: BN residual doesn't align with any known class → low cos sim
""")
