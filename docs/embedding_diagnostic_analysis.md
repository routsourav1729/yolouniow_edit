# YOLO-UniOW IDD T2 — Embedding Space Diagnostic Analysis

## Setup

- **Model**: YOLO-UniOW-L, IDD dataset, Task 2 (8 base + 6 novel + unknown + anchor = 16 prompts)
- **Checkpoint**: `best_owod_Both_epoch_40.pth` from T2 10-shot fine-tuning
- **Eval results**: PK (base mAP) = 39.47, CK (novel mAP) = 25.39, U-Recall = 42.21%, A-OSE = 1903
- **Analysis**: 319,855 vision features extracted at GT locations across all 21,084 test images

## Architecture Recap

Detection scores are computed as:
```
score = sigmoid( BN(cls_embed) · L2norm(prompt) × exp(logit_scale) + bias )
```
where `cls_embed` is a 512-dim vector from FPN cls_pred convs at each grid cell, and `prompt` is the learned 512-dim class embedding. The dot product between BN-normalized vision features (norm ~41) and L2-normalized prompts determines which class wins at each location.

## Key Finding 1: Novel Prompts Barely Trained

| Class | Type | Init Norm | Trained Norm | cos(init, trained) | Angular Shift |
|-------|------|-----------|--------------|-------------------|---------------|
| car | base | 1.00 | 4.52 | 0.352 | 69.4° |
| motorcycle | base | 1.00 | 4.31 | 0.298 | 72.7° |
| person | base | 1.00 | 4.36 | 0.277 | 73.9° |
| bus | **novel** | 1.00 | **1.01** | **0.996** | **5.2°** |
| truck | **novel** | 1.00 | **1.00** | **0.997** | **4.7°** |
| tanker_vehicle | **novel** | 1.00 | **1.00** | **0.994** | **6.3°** |
| crane_truck | **novel** | 1.00 | **1.01** | **0.992** | **7.3°** |
| street_cart | **novel** | 1.00 | **1.00** | **0.993** | **7.0°** |
| excavator | **novel** | 1.00 | **1.00** | **0.996** | **5.3°** |

**Base class prompts** rotated 70°+ from CLIP initialization and grew to norm ~4.4 during T1 full training.
**Novel class prompts** rotated only 5–7° from CLIP initialization and stayed at unit norm after T2 10-shot fine-tuning (40 epochs).

### Why does CK mAP still improve from 17→25 despite tiny rotation?

The scoring math amplifies small angular changes:
- BN features have norm ~41
- logit_scale ≈ exp(0.1–0.5) ≈ 1.1–1.7
- bias ≈ -11 to -13

A 5° rotation on a unit-norm prompt changes the L2-normalized direction by ~0.09, producing a dot-product shift of `41 × 0.09 × 1.5 ≈ 5.5` logit units against a bias of -12. This is enough to push sigmoid scores from near-zero to detectable. But the prompts remain fundamentally under-separated — they all point in nearly the same direction (their CLIP initialization direction).

### Verification: Correct Weights Are Loaded

Comparing T1 checkpoint (10 embeddings) with T2 checkpoint (16 embeddings):

| Row | T1 Identity | T2 Identity | cos(T1, T2) | Status |
|-----|------------|------------|-------------|--------|
| 0–7 | base classes | base classes | 1.000 | Frozen correctly (mask=0) |
| 8 | unknown | bus | 0.377 | Replaced with CLIP init |
| 9 | anchor | truck | 0.605 | Replaced with CLIP init |
| 10–13 | — | tanker..excavator | — | New CLIP init |
| 14 | — | unknown | — | New CLIP init |
| 15 | — | anchor | — | Copied from T1 anchor |

**These are the genuine T2-finetuned novel embeddings.** The T2 training initialized novel slots from CLIP text and trained them for 40 epochs on 10 shots. They differ from CLIP init (L2=0.08–0.13, cos=0.992–0.997) — they did train, just not far. The embedding_mask `[0,0,0,0,0,0,0,0, 1,1,1,1,1,1, 1, 0]` marks novel+unknown as trainable and base+anchor as frozen.

## Key Finding 2: Novel Vision Feature Clouds Are Badly Separated

From 319k features across the full test set:

| Class | Type | N samples | cos(prompt, centroid) | Intra-class cos | Nearest centroid |
|-------|------|-----------|----------------------|-----------------|-----------------|
| car | base | 55,363 | 0.189 | 0.322 | bus (0.574) |
| person | base | 46,498 | 0.399 | 0.345 | rider (0.616) |
| bus | novel | 7,900 | **-0.003** | 0.442 | **truck (0.874)** |
| truck | novel | 15,378 | **0.007** | 0.413 | **bus (0.874)** |
| tanker_vehicle | novel | 355 | **-0.081** | 0.502 | **excavator (0.886)** |
| crane_truck | novel | 182 | **-0.023** | 0.564 | **excavator (0.901)** |
| excavator | novel | 217 | **-0.091** | 0.468 | **crane_truck (0.901)** |
| unknown | unk | 4,281 | **-0.044** | 0.442 | traffic sign (0.686) |

**Critical observations:**
1. **Novel prompt-to-centroid cosine similarity is near zero or negative** — the prompts point in essentially random directions relative to their actual vision feature distributions
2. **Novel class centroids are nearly identical** — bus↔truck (0.87), tanker↔excavator (0.89), crane↔excavator (0.90). The vision features for these classes are barely distinguishable
3. **Base prompt-to-centroid similarity is moderate (0.19–0.40)** — much better alignment, explaining the higher base mAP
4. **Intra-class cosine for novel classes is high (0.41–0.56)** — the feature clouds are tight but the prompts don't point toward them

## Key Finding 3: A-OSE Breakdown (Full Test Set)

Total A-OSE = 1903 unknowns misclassified as known. From GT-location logit analysis on all 4,281 unknown GT boxes:

### Which known class absorbs unknowns (at GT locations)?

| Absorbing Class | Count | % |
|----------------|-------|---|
| crane_truck (novel) | 1,079 | 25.2% |
| traffic sign (base) | 821 | 19.2% |
| person (base) | 767 | 17.9% |
| motorcycle (base) | 385 | 9.0% |
| car (base) | 275 | 6.4% |
| traffic light (base) | 208 | 4.9% |
| truck (novel) | 181 | 4.2% |

**Crane_truck alone absorbs 25% of unknown misclassifications** despite being a novel class with only 182 test samples. This is because its prompt (barely moved from CLIP init) happens to point in a direction that weakly matches diverse unknown objects.

### Per unknown class absorption pattern

| Unknown Class | N | known>unk % | Top Absorber |
|--------------|---|-------------|-------------|
| tractor | 447 | 87.5% | motorcycle |
| concrete_mixer | 110 | 80.0% | traffic sign |
| road_roller | 47 | 80.9% | crane_truck |
| pull_cart | 67 | 76.1% | street_cart |
| animal | 1,598 | 68.0% | person |
| pole | 2,012 | 26.3% | crane_truck |

**Mean best-known score at unknown GT: 0.017, Mean unknown score: 0.006**
Both scores are very low (post-sigmoid), but known > unknown in **51%** of cases. The model is making decisions in the noise floor.

## Key Finding 4: The Prompt Similarity Heatmap Reveals the Problem

**Left heatmap** (Prompt ↔ Prompt cosine similarity): Novel class prompts have inter-similarity of 0.3–0.4 with each other AND with base prompts. They haven't differentiated. The unknown prompt sits at similar similarity to everything.

**Right heatmap** (Vision centroid ↔ centroid): Shows the actual feature structure. Bus↔truck (0.87), tanker↔crane↔excavator (0.75–0.90) form a tight vehicle supercluster. But in prompt space, these classes haven't separated to match this structure.

## Key Finding 5: UMAP Visualization

The UMAP overview (cosine metric, 320k points) shows:
- **Base class features** form a large continuous mass with some class-specific regions. Base prompts (stars) sit within or near their class regions
- **Novel class features** form small isolated clusters at the periphery (especially heavy vehicles: bus, truck, crane_truck, excavator cluster together at top-right)
- **All 6 novel prompts collapse to nearly the same UMAP location** — confirming they barely separated from CLIP init
- **Unknown features** scatter broadly, overlapping with multiple known class regions
- **Anchor embedding** (norm=29.3) projects to an extreme outlier position

## Root Cause Summary

The A-OSE problem stems from a chain of three issues:

1. **Under-trained novel prompts**: 10-shot × 40 epochs produces only 5–7° rotation from CLIP text initialization. The prompts haven't learned where their vision features actually live in the BN-normalized space. Their cosine similarity to their own class centroids is ~0 (essentially orthogonal).

2. **Novel class feature overlap**: Bus/truck and tanker/crane/excavator form super-clusters with centroid cosine similarity >0.87. Even well-trained prompts would struggle to separate them. The 10-shot training data is insufficient to learn the fine-grained boundaries.

3. **Weak unknown prompt**: The unknown prompt is slightly less aligned than the weakest known prompt at most unknown GT locations (known > unknown 51% of the time), but both scores are in the noise floor (0.006–0.017 post-sigmoid). The decision is essentially random at these low confidence levels, and the NMS/score-threshold pipeline amplifies this into A-OSE.

## Implications for Design

### What the data tells us we need

1. **Prompt initialization matters enormously** — the 5–7° rotation suggests the optimization landscape is nearly flat for novel prompts. Better initialization (e.g., attribute-enriched text, nearest-base-class interpolation) could start prompts closer to the correct vision feature direction.

2. **Novel prompt norms should grow** — base prompts trained to norm ~4.4 during T1. Novel prompts staying at norm ~1.0 means the L2-normalization step compresses their small angular differences further. Allowing or encouraging norm growth could amplify discriminative directions.

3. **The bus/truck and tanker/crane/excavator confusion is a feature-space problem, not just a prompt problem** — centroid similarity >0.87 means the BN features themselves don't separate these classes. Prompt tuning alone can't fix this; the visual backbone would need adaptation (LoRA was off during T2).

4. **Unknown detection operates in the noise floor** — mean scores of 0.006–0.017 are far below any reasonable operating threshold. The unknown prompt needs to be pushed much further toward the actual unknown feature distribution, or the scoring mechanism needs structural changes (e.g., energy-based scoring, OOD-specific head).

5. **Crane_truck is the worst A-OSE offender despite being novel** — its CLIP text direction happens to weakly match diverse unknown objects. This is a pathological case of under-trained prompts: the CLIP "crane truck" text embedding direction accidentally absorbs poles, road rollers, and other unknowns.
