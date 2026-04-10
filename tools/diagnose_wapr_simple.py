"""Minimal diagnostic: compare raw vs BN-processed embeddings.
Loads checkpoint weights directly, no full model build needed."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

CKPT = 'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task1/best_owod_Both_epoch_20.pth'
T2_EMBED = 'embeddings/uniow-food-voccoco/food_voccoco_t2.npy'
OBJECT_NPY = 'embeddings/uniow-food-voccoco/object_tuned.npy'

def main():
    print("Loading checkpoint...")
    ckpt = torch.load(CKPT, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    # Get text embeddings from checkpoint
    embeddings = sd['embeddings']  # (K, 512) — T1 embeddings
    print(f"\n{'='*60}")
    print(f"CHECKPOINT EMBEDDINGS (T1)")
    print(f"{'='*60}")
    print(f"Shape: {embeddings.shape}")
    norms = embeddings.norm(dim=-1)
    print(f"Norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    print(f"  Known (0..{len(norms)-3}): {norms[:-2].tolist()[:5]}... mean={norms[:-2].mean():.4f}")
    print(f"  T_unk (idx -2): norm={norms[-2]:.4f}")
    print(f"  Anchor (idx -1): norm={norms[-1]:.4f}")
    
    # Load T2 embeddings  
    t2_embeds = torch.from_numpy(np.load(T2_EMBED)).float()
    print(f"\nT2 embeddings: shape={t2_embeds.shape}, norms mean={t2_embeds.norm(dim=-1).mean():.4f}")
    
    # Get BN parameters from checkpoint
    # Find the BN layer in the contrastive head
    bn_keys = [k for k in sd.keys() if 'cls_contrast' in k and ('weight' in k or 'bias' in k or 'running' in k)]
    print(f"\n{'='*60}")
    print(f"BN CONTRASTIVE HEAD PARAMETERS")
    print(f"{'='*60}")
    for k in sorted(bn_keys)[:20]:
        v = sd[k]
        print(f"  {k}: shape={v.shape}, mean={v.float().mean():.4f}, std={v.float().std():.4f}")
    
    # Extract BN running_mean and running_var for analysis
    bn_mean_key = [k for k in bn_keys if 'running_mean' in k and 'one2many' in k]
    bn_var_key = [k for k in bn_keys if 'running_var' in k and 'one2many' in k] 
    bn_weight_key = [k for k in bn_keys if '.weight' in k and 'one2many' in k and 'norm' in k]
    bn_bias_key = [k for k in bn_keys if '.bias' in k and 'one2many' in k and 'norm' in k]
    
    if bn_mean_key:
        rmean = sd[bn_mean_key[0]]
        rvar = sd[bn_var_key[0]]
        gamma = sd[bn_weight_key[0]] if bn_weight_key else torch.ones_like(rmean)
        beta = sd[bn_bias_key[0]] if bn_bias_key else torch.zeros_like(rmean)
        
        print(f"\n{'='*60}")
        print(f"BN STATISTICS (one2many level 0)")
        print(f"{'='*60}")
        print(f"  running_mean: mean={rmean.mean():.4f}, std={rmean.std():.4f}, range=[{rmean.min():.4f}, {rmean.max():.4f}]")
        print(f"  running_var:  mean={rvar.mean():.4f}, std={rvar.std():.4f}, range=[{rvar.min():.4f}, {rvar.max():.4f}]")
        print(f"  gamma (weight): mean={gamma.mean():.4f}, std={gamma.std():.4f}")
        print(f"  beta (bias):  mean={beta.mean():.4f}, std={beta.std():.4f}")
        
        # Simulate: what happens to a random cls_embed vector?
        # BN: output = gamma * (x - mean) / sqrt(var + eps) + beta
        # This changes the magnitude AND direction of the vector
        
        # Create synthetic "cls_embed" vector (use a random unit vector scaled like typical)
        # But better: let's check what raw embed magnitudes look like
        # by examining the BN statistics
        
        # The key insight: BN subtracting the mean is what aligns features with text
        # Without it, all features share a common component → low cosine diversity
        print(f"\n{'='*60}")
        print(f"WHY RAW COSINE SIM IS LOW")
        print(f"{'='*60}")
        print(f"  BN running_mean norm: {rmean.norm():.4f}")
        print(f"  BN running_mean / sqrt(var) norm: {(rmean / (rvar + 1e-5).sqrt()).norm():.4f}")
        print(f"  The raw cls_embed shares a large common component (running_mean).")
        print(f"  L2-normalizing raw embeddings kills class-specific signal.")
        print(f"  BN centering removes this common component, revealing class info.")
        
        # Check: does the BN mean dominate?
        mean_norm = rmean.norm()
        var_mean = rvar.mean().sqrt()
        print(f"\n  Analogy: imagine cls_embed ≈ running_mean + small_class_signal")
        print(f"    ||running_mean|| = {mean_norm:.2f}")
        print(f"    sqrt(mean(var)) = {var_mean:.4f}  (typical deviation from mean)")
        print(f"    Ratio (SNR): {var_mean / mean_norm:.4f}  (class signal is tiny relative to mean)")
        print(f"    → L2-norm(raw) ≈ mean_direction for ALL anchors → cosine sim to text ≈ constant and low")
    
    # Also check logit_scale and bias
    ls_keys = [k for k in sd.keys() if 'logit_scale' in k and 'one2many' in k]
    bias_keys = [k for k in sd.keys() if 'cls_contrast' in k and '.bias' in k and 'one2many' in k and 'norm' not in k]
    if ls_keys:
        ls = sd[ls_keys[0]]
        print(f"\n{'='*60}")
        print(f"LOGIT SCALE AND BIAS")
        print(f"{'='*60}")
        print(f"  logit_scale = {ls.item():.4f}, exp(logit_scale) = {ls.exp().item():.4f}")
        if bias_keys:
            b = sd[bias_keys[0]]
            print(f"  bias = {b.item():.4f}")
        print(f"  logit = BN(x) · L2norm(txt) * {ls.exp().item():.4f} + {b.item() if bias_keys else 0:.4f}")
        print(f"  sigmoid(logit) with scale={ls.exp().item():.3f} compresses everything toward 0.5")
        print(f"  For an anchor with BN(x)·txt={0.5:.1f}: logit={0.5*ls.exp().item()+b.item():.3f} → sigmoid={torch.sigmoid(torch.tensor(0.5*ls.exp().item()+b.item())).item():.4f}")
        print(f"  For an anchor with BN(x)·txt={0.1:.1f}: logit={0.1*ls.exp().item()+b.item():.3f} → sigmoid={torch.sigmoid(torch.tensor(0.1*ls.exp().item()+b.item())).item():.4f}")
        print(f"  For an anchor with BN(x)·txt={-0.1:.1f}: logit={-0.1*ls.exp().item()+b.item():.3f} → sigmoid={torch.sigmoid(torch.tensor(-0.1*ls.exp().item()+b.item())).item():.4f}")

    print(f"\n{'='*60}")
    print(f"CONCLUSION")
    print(f"{'='*60}")
    print(f"  Raw cosine sim is low because cls_embed has a dominant common direction")
    print(f"  (the BN running_mean). L2-norm removes magnitude info but the direction")
    print(f"  is dominated by this mean. Class-specific info only emerges after BN centering.")
    print(f"  ")
    print(f"  To fix WAPR: either")
    print(f"  (a) Cache BN(cls_embed) instead of raw cls_embed → then cosine sim works")
    print(f"  (b) Use the model's own logit margin: w_r = 1 - max_known/anchor_score")
    print(f"      (already calibrated in the model's space, no BN needed)")

if __name__ == '__main__':
    main()
