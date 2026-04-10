"""Diagnostic: why is cosine similarity between cls_embed and txt_feats near zero?

Tests three embedding spaces:
1. Raw cls_embed (pre-BN) vs txt_feats → cosine similarity
2. BN(cls_embed) vs L2norm(txt_feats) → cosine similarity (proper space)
3. Full BNContrastiveHead logits → sigmoid (what model uses)

Loads T1 best checkpoint, runs a few val images, and reports statistics.
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

os.environ['DATASET'] = 'FOOD_VOCCOCO'
os.environ['TASK'] = '2'
os.environ['THRESHOLD'] = '0.05'
os.environ['SAVE'] = 'False'
os.environ['FEWSHOT_K'] = '10'
os.environ['FEWSHOT_SEED'] = '1'
os.environ['FEWSHOT_DIR'] = 'data/OWOD/voccocosplit'

CONFIG = 'configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2_wapr.py'
CKPT = 'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task1/best_owod_Both_epoch_20.pth'

def main():
    from mmengine.config import Config
    from mmengine.runner import load_state_dict
    
    # Register all custom modules
    import yolo_world  # noqa
    import mmyolo  # noqa
    import mmdet  # noqa
    
    cfg = Config.fromfile(CONFIG)
    cfg.work_dir = '/tmp/diagnose_wapr'
    
    # Build model
    from mmyolo.registry import MODELS
    model = MODELS.build(cfg.model)
    
    # Load T1 checkpoint
    ckpt = torch.load(CKPT, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k: v for k, v in state_dict.items() if 'text_model' not in k}
    if "embeddings" in state_dict:
        state_dict['embeddings'] = model.update_embeddings(state_dict['embeddings'])
    
    load_state_dict(model, state_dict, strict=False)
    model.eval()
    model.cuda()
    
    # Get txt_feats
    txt_feats = model.embeddings[None].cuda()  # (1, K, 512)
    print(f"\n{'='*60}")
    print(f"WAPR DIAGNOSTIC — EMBEDDING SPACE ANALYSIS")
    print(f"{'='*60}")
    print(f"\nText embeddings shape: {txt_feats.shape}")
    norms = txt_feats[0].norm(dim=-1)
    print(f"Text embed norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    print(f"  First 5 known: {norms[:5].tolist()}")
    print(f"  T_unk (idx -2): {norms[-2]:.4f}")
    print(f"  Anchor (idx -1): {norms[-1]:.4f}")
    
    # Build val dataloader
    from mmengine.registry import DATASETS
    val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
    
    head = model.bbox_head.head_module
    NUM_IMGS = 10
    print(f"\nProcessing {NUM_IMGS} val images...")
    
    all_raw_cos = []
    all_bn_cos = []
    all_logit_max_prob = []
    all_anchor_prob = []
    all_max_known_prob = []
    raw_embed_norms = []
    bn_embed_norms = []
    
    with torch.no_grad():
        for i in range(min(NUM_IMGS, len(val_dataset))):
            data = val_dataset[i]
            from mmengine.dataset import pseudo_collate
            batch = pseudo_collate([data])
            
            batch_inputs = batch['inputs']
            if isinstance(batch_inputs, list):
                batch_inputs = torch.stack(batch_inputs)
            batch_inputs = batch_inputs.float().cuda()
            
            # Forward through backbone + neck
            img_feats = model.backbone(batch_inputs)
            if model.with_neck:
                img_feats = model.neck(img_feats)
            
            # Only use one FPN level (the largest) for quick analysis
            for lvl in range(len(img_feats)):
                img_feat = img_feats[lvl]
                
                # Raw cls_embed
                cls_embed = head.one2many_cls_preds(img_feat)  # (B, 512, H, W)
                
                # BN output
                bn_embed = head.one2many_cls_contrasts.norm(cls_embed)  # (B, 512, H, W)
                
                # Full logit
                cls_logit = head.one2many_cls_contrasts(cls_embed, txt_feats)  # (B, K, H, W)
                
                b, c, h, w = cls_embed.shape
                _, k, _, _ = cls_logit.shape
                
                flat_raw = cls_embed.permute(0, 2, 3, 1).reshape(b, h*w, c)
                flat_bn = bn_embed.permute(0, 2, 3, 1).reshape(b, h*w, c)
                flat_logit = cls_logit.permute(0, 2, 3, 1).reshape(b, h*w, k)
                
                known_txt = txt_feats[:, :40, :]
                
                # Norms
                raw_embed_norms.append(flat_raw.norm(dim=-1).flatten())
                bn_embed_norms.append(flat_bn.norm(dim=-1).flatten())
                
                # 1. Raw cosine sim
                raw_n = F.normalize(flat_raw, dim=-1, p=2)
                txt_n = F.normalize(known_txt, dim=-1, p=2)
                raw_cos = torch.bmm(raw_n, txt_n.transpose(1, 2)).max(dim=-1)[0]
                
                # 2. BN cosine sim
                bn_n = F.normalize(flat_bn, dim=-1, p=2)
                bn_cos = torch.bmm(bn_n, txt_n.transpose(1, 2)).max(dim=-1)[0]
                
                # 3. Logit probs
                known_probs = flat_logit[:, :, :40].sigmoid()
                max_known = known_probs.max(dim=-1)[0]
                anchor_prob = flat_logit[:, :, -1].sigmoid()
                
                all_raw_cos.append(raw_cos.flatten())
                all_bn_cos.append(bn_cos.flatten())
                all_logit_max_prob.append(max_known.flatten())
                all_anchor_prob.append(anchor_prob.flatten())
                all_max_known_prob.append(max_known.flatten())
    
    raw_cos = torch.cat(all_raw_cos)
    bn_cos = torch.cat(all_bn_cos)
    logit_probs = torch.cat(all_logit_max_prob)
    anchor_prob = torch.cat(all_anchor_prob)
    max_known = torch.cat(all_max_known_prob)
    raw_norms = torch.cat(raw_embed_norms)
    bn_norms = torch.cat(bn_embed_norms)
    
    print(f"\n{'='*60}")
    print(f"EMBEDDING NORMS")
    print(f"{'='*60}")
    print(f"  Raw cls_embed norm:  mean={raw_norms.mean():.2f}, std={raw_norms.std():.2f}, "
          f"min={raw_norms.min():.2f}, max={raw_norms.max():.2f}")
    print(f"  BN(cls_embed) norm:  mean={bn_norms.mean():.2f}, std={bn_norms.std():.2f}, "
          f"min={bn_norms.min():.2f}, max={bn_norms.max():.2f}")
    print(f"  Text embed norm:     mean={norms[:40].mean():.2f}")
    
    print(f"\n{'='*60}")
    print(f"SPACE 1: Raw cls_embed (pre-BN) vs txt_feats — COSINE SIM")
    print(f"{'='*60}")
    print(f"  ALL anchors — max cos sim:")
    print(f"    mean={raw_cos.mean():.4f}, std={raw_cos.std():.4f}")
    print(f"    p25={raw_cos.quantile(0.25):.4f}, p50={raw_cos.quantile(0.5):.4f}, "
          f"p75={raw_cos.quantile(0.75):.4f}, p95={raw_cos.quantile(0.95):.4f}, p99={raw_cos.quantile(0.99):.4f}")
    
    print(f"\n{'='*60}")
    print(f"SPACE 2: BN(cls_embed) vs L2norm(txt_feats) — COSINE SIM")
    print(f"{'='*60}")
    print(f"  ALL anchors — max cos sim:")
    print(f"    mean={bn_cos.mean():.4f}, std={bn_cos.std():.4f}")
    print(f"    p25={bn_cos.quantile(0.25):.4f}, p50={bn_cos.quantile(0.5):.4f}, "
          f"p75={bn_cos.quantile(0.75):.4f}, p95={bn_cos.quantile(0.95):.4f}, p99={bn_cos.quantile(0.99):.4f}")
    
    print(f"\n{'='*60}")
    print(f"SPACE 3: BNContrastiveHead logit → sigmoid (model's own)")
    print(f"{'='*60}")
    print(f"  ALL anchors — max known prob (sigmoid):")
    print(f"    mean={logit_probs.mean():.4f}, std={logit_probs.std():.4f}")
    print(f"    p25={logit_probs.quantile(0.25):.4f}, p50={logit_probs.quantile(0.5):.4f}, "
          f"p75={logit_probs.quantile(0.75):.4f}, p95={logit_probs.quantile(0.95):.4f}, p99={logit_probs.quantile(0.99):.4f}")
    
    print(f"\n  Anchor (objectness) prob:")
    print(f"    mean={anchor_prob.mean():.4f}, std={anchor_prob.std():.4f}")
    print(f"    p95={anchor_prob.quantile(0.95):.4f}, p99={anchor_prob.quantile(0.99):.4f}")
    
    # Gatekeeper analysis
    gk_mask = (anchor_prob > 0.01) & (anchor_prob > max_known)
    n_gk = gk_mask.sum().item()
    n_total = len(anchor_prob)
    
    print(f"\n{'='*60}")
    print(f"GATEKEEPER-PASSING ANCHORS")
    print(f"{'='*60}")
    print(f"  Passing: {n_gk} / {n_total} ({100*n_gk/n_total:.3f}%)")
    
    if n_gk > 0:
        gk_raw = raw_cos[gk_mask]
        gk_bn = bn_cos[gk_mask]
        gk_logit = logit_probs[gk_mask]
        gk_anchor = anchor_prob[gk_mask]
        gk_known = max_known[gk_mask]
        
        print(f"\n  Among gatekeeper-passing anchors:")
        print(f"  ─── Raw cosine sim ───")
        print(f"    mean={gk_raw.mean():.4f}, std={gk_raw.std():.4f}")
        print(f"    p25={gk_raw.quantile(0.25):.4f}, p50={gk_raw.quantile(0.5):.4f}, "
              f"p75={gk_raw.quantile(0.75):.4f}, p95={gk_raw.quantile(0.95):.4f}")
        
        print(f"  ─── BN cosine sim ───")
        print(f"    mean={gk_bn.mean():.4f}, std={gk_bn.std():.4f}")
        print(f"    p25={gk_bn.quantile(0.25):.4f}, p50={gk_bn.quantile(0.5):.4f}, "
              f"p75={gk_bn.quantile(0.75):.4f}, p95={gk_bn.quantile(0.95):.4f}")
        
        print(f"  ─── Logit sigmoid prob ───")
        print(f"    max known:  mean={gk_logit.mean():.4f}, std={gk_logit.std():.4f}")
        print(f"    anchor:     mean={gk_anchor.mean():.4f}, std={gk_anchor.std():.4f}")
        
        # Ratio-based w_r
        ratio = gk_known / gk_anchor.clamp(min=1e-6)
        w_r_ratio = (1.0 - ratio).clamp(0, 1)
        print(f"  ─── Ratio w_r = 1 - max_known/anchor ───")
        print(f"    mean={w_r_ratio.mean():.4f}, std={w_r_ratio.std():.4f}")
        print(f"    p25={w_r_ratio.quantile(0.25):.4f}, p50={w_r_ratio.quantile(0.5):.4f}, "
              f"p75={w_r_ratio.quantile(0.75):.4f}, p95={w_r_ratio.quantile(0.95):.4f}")
        
        # BN cosine w_r
        w_r_bn = (1.0 - gk_bn).clamp(0, 1)
        print(f"  ─── BN cosine w_r = 1 - max_bn_cos ───")
        print(f"    mean={w_r_bn.mean():.4f}, std={w_r_bn.std():.4f}")
        print(f"    p25={w_r_bn.quantile(0.25):.4f}, p50={w_r_bn.quantile(0.5):.4f}, "
              f"p75={w_r_bn.quantile(0.75):.4f}, p95={w_r_bn.quantile(0.95):.4f}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
