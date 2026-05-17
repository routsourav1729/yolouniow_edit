# Web Search Context: Fast A-OSE Reduction for YOLO-UniOW

This document is a context packet for a web-enabled ChatGPT/research agent.
Its only job is to give enough local evidence and design constraints to search
for a better architecture or calibration method for YOLO-UniOW.

The target problem is not general accuracy. The target problem is:

> Reduce absolute open-set error (A-OSE), especially unknown objects predicted as
> known fine-grained classes, while keeping YOLO-UniOW inference almost as fast as
> the current design.

The method being searched for should do the same kind of job as "MSCAL" if that
is the relevant external method name, but it must be faster and cheaper than an
OVOW-style dense multi-scale, per-class visual branch. There is no local
implementation or mention of `MSCAL`/`mscal` in this repository, so a web agent
should first verify what MSCAL exactly means in the open-world/open-set object
detection literature before recommending designs.

## Summary For The Web Agent

We need a lightweight way to use visual information to stop unknown objects from
being claimed by known classes.

The current YOLO-UniOW design is efficient because it avoids OVOW's visual
multi-modal neck and per-class, per-scale projection heads. YOLO-UniOW mostly
uses frozen visual features plus learned prompt embeddings. This is fast, but in
few-shot stages it is structurally weak for fine-grained open-set rejection:
novel classes have too few positive examples, the learned prompt vectors barely
move, and post-BN visual embeddings overlap heavily between similar known,
novel, and unknown vehicle-like classes.

The observed failure is concentrated, not random. Unknown objects are mostly
misclassified as a small set of fine-grained vehicle/construction classes:
`truck`, `tanker_vehicle`, `excavator`, and `autorickshaw`.

The desired design should:

- Use visual features only as a cheap rejection or calibration signal.
- Avoid dense per-class per-scale expansion.
- Prefer candidate-only computation after top-k selection or NMS.
- Preserve the frozen-backbone/frozen-head prompt-tuning nature of YOLO-UniOW.
- Reduce A-OSE with minimal drop in known AP and unknown recall.
- Be compatible with few-shot open-world detection where unknown labels are not
  abundant or are only pseudo-mined.

Good search directions are likely:

- open-set object detection calibration
- unknown-as-known error reduction
- visual-language open-set detection rejection
- energy/margin/entropy calibration for detection
- class-conditional thresholds
- OpenMax-like rejection in detector embedding space
- Mahalanobis/prototype/density rejection in penultimate features
- selective classification for object detection
- lightweight OOD heads for YOLO detectors
- unknown-aware NMS or bidirectional NMS
- multi-scale calibration alternatives to MSCAL
- candidate-level rather than anchor-level visual calibration

## Current Best YOLO-UniOW Result

Best local run:

```text
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/idd/fedbce/idd_t2_fedbce_hnunk20_tunkvanilla_freeze20_40e_val10_118448.out
```

Final open-world metrics from that run:

| Metric | Value |
|---|---:|
| U-Recall | 47.9327 |
| WI | 0.0033 |
| A-OSE | 1902 |
| PK | 39.4743 |
| CK | 32.3067 |
| Both | 36.4025 |
| Unknown AP50 | 1.4895 |
| Unknown precision | 0.1084 |
| Unknown recall | 47.9327 |

AP50 / AR50 class table from the best run:

| Class | AP50 / AR50 |
|---|---:|
| car | 63.740 |
| motorcycle | 51.968 |
| rider | 40.811 |
| person | 36.206 |
| autorickshaw | 56.245 |
| bicycle | 28.924 |
| traffic sign | 19.420 |
| traffic light | 18.479 |
| bus | 53.298 |
| truck | 31.812 |
| tanker_vehicle | 14.107 |
| crane_truck | 30.214 |
| street_cart | 42.740 |
| excavator | 21.670 |
| unknown AR50 | 47.933 |

The known/novel mAP is acceptable for the current stage, but unknown precision
is extremely low and A-OSE remains high. This means the detector can sometimes
localize unknown objects, but the decision boundary still allows many unknowns
to be assigned to known classes.

## A-OSE Breakdown

A-OSE means unknown ground-truth objects counted as known predictions. This is
the exact error we want to reduce.

From the best run:

| Predicted known class | A-OSE count | Share of total A-OSE |
|---|---:|---:|
| truck | 446 | 23.45% |
| tanker_vehicle | 428 | 22.50% |
| excavator | 349 | 18.35% |
| autorickshaw | 308 | 16.19% |
| person | 101 | 5.31% |
| motorcycle | 98 | 5.15% |
| street_cart | 62 | 3.26% |
| rider | 42 | 2.21% |
| crane_truck | 35 | 1.84% |
| car | 16 | 0.84% |
| traffic sign | 7 | 0.37% |
| bus | 7 | 0.37% |
| bicycle | 3 | 0.16% |
| traffic light | 0 | 0.00% |
| total | 1902 | 100.00% |

The top four classes account for:

```text
truck + tanker_vehicle + excavator + autorickshaw
= 446 + 428 + 349 + 308
= 1531 / 1902
= 80.49% of all A-OSE
```

This suggests a targeted design may work better than a global threshold. We do
not need to perturb all classes equally. We mostly need to defend against
vehicle/construction-like unknowns being absorbed by similar known classes.

## Unknown Recall Breakdown

Unknown ground-truth classes in the IDD T2 evaluation are not simply one kind of
object. They include both easy and fine-grained unknowns:

| Unknown GT class | Recalled / Total | Recall |
|---|---:|---:|
| pole | 905 / 2012 | 44.98% |
| animal | 624 / 1598 | 39.05% |
| tractor | 360 / 447 | 80.54% |
| concrete_mixer | 71 / 110 | 64.55% |
| pull_cart | 60 / 67 | 89.55% |
| road_roller | 31 / 47 | 65.96% |
| total | 2051 / 4281 | 47.91% |

Important interpretation:

- Some unknown classes are already localized or recalled reasonably well.
- A-OSE is still high because unknown-vs-known classification is fragile.
- Heavy vehicle/construction unknowns are visually near known classes such as
  `truck`, `tanker_vehicle`, and `excavator`.

## A-OSE Score Diagnostics

For A-OSE predictions in the best run:

| Quantity | Value |
|---|---:|
| count | 1902 |
| mean predicted known score | 0.1850 |
| median predicted known score | 0.1280 |
| p90 predicted known score | 0.4070 |
| max predicted known score | 0.9479 |

Interpretation:

- Many A-OSE cases are not extremely confident.
- A candidate-level rejection or relabeling rule may catch many of them without
  destroying known AP.
- Some cases are high-confidence, so score-only thresholding will not be enough.
  The method probably needs an extra signal: visual prototype distance, margin,
  density, energy, uncertainty, or overlap with an unknown candidate.

## IDD T2 Class Setup

The local YOLO-UniOW T2 setup uses:

| Group | Class IDs / Names |
|---|---|
| Base known | 0 car, 1 motorcycle, 2 rider, 3 person, 4 autorickshaw, 5 bicycle, 6 traffic sign, 7 traffic light |
| Novel known | 8 bus, 9 truck, 10 tanker_vehicle, 11 crane_truck, 12 street_cart, 13 excavator |
| Unknown prompt | 14 unknown |
| Anchor prompt | 15 anchor |

Unknown evaluation GT classes include:

```text
pole, animal, tractor, concrete_mixer, pull_cart, road_roller
```

The most difficult confusion is not generic background. It is fine-grained
object-level confusion between semantically nearby traffic/vehicle/construction
categories.

## Current YOLO-UniOW Architecture

Relevant local files:

```text
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/yolo_world/models/detectors/yolo_world_owod.py
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/third_party/mmyolo/mmyolo/models/dense_heads/yolov10_head.py
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/yolo_world/models/dense_heads/yolov10_world_head.py
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/yolo_world/models/dense_heads/visual_cache.py
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py
```

High-level design:

- YOLO-UniOW is based on YOLO-World/YOLOv10 style detection.
- It does not keep a text encoder active at inference for this stage.
- It uses trainable prompt embeddings for base/novel/unknown/anchor concepts.
- It freezes most visual detection components in the fine-tuning stage.
- It uses `mm_neck=False` in the relevant open-world fine-tuning config.
- The detector head scores visual features against prompt embeddings.
- Unknown behavior comes mainly from `T_unk` and pseudo-unknown mining via
  `T_anchor`.

The scoring math from local diagnostics is:

```text
score = sigmoid(BN(cls_embed) dot L2norm(prompt) * exp(logit_scale) + bias)
```

That means the post-BN visual embedding space is critical. If similar known,
novel, and unknown objects overlap in this post-BN space, then the prompt
classifier has very little signal to reject unknown objects.

## YOLO-UniOW Pseudo-Unknown Mining

The patched YOLOv10 head mines pseudo unknowns using anchor-like detections:

```text
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/third_party/mmyolo/mmyolo/models/dense_heads/yolov10_head.py
```

Relevant behavior:

- It forms an unknown mask from anchors that are low-IoU to known GT and not
  assigned foreground.
- It requires anchor score above a threshold.
- It can optionally require anchor score to exceed max known score after a
  given epoch.

This is useful, but it is still weak in few-shot settings:

- pseudo unknown positives are noisy;
- real unknown categories are diverse;
- fine-grained unknown vehicle classes can be near known vehicle classes;
- the unknown prompt can remain under-trained or too generic.

## Current YOLO-UniOW Inference Behavior

In inference:

- the anchor channel is zeroed out;
- scores are computed for known classes plus `T_unk`;
- the label is chosen mostly by argmax;
- unknown post-processing suppresses unknown boxes that overlap other boxes at a
  high IoU threshold;
- the reverse does not happen by default: weak known boxes overlapping unknown
  boxes are not aggressively relabeled or suppressed as unknown.

This matters for A-OSE:

- if a true unknown object has a slightly higher known score than unknown score,
  the known label wins;
- if both known and unknown boxes exist for the same object, current
  post-processing is more likely to remove the unknown duplicate than to remove
  or relabel the known duplicate;
- this favors known predictions and contributes directly to unknown-as-known
  error.

## Few-Shot Prompt Limitation

Local diagnostic document:

```text
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/docs/embedding_diagnostic_analysis.md
```

Important findings from that diagnostic:

- In T2 few-shot training, novel prompts barely move.
- Base prompts trained with richer data rotate about 70 degrees and grow to
  norms around 4.4.
- Novel prompts rotate only about 5 to 7 degrees and remain near norm 1.0.
- Unknown scores at GT locations can sit near the noise floor.

Interpretation:

YOLO-UniOW's current few-shot design is stunted by the lack of positive
examples. The prompt vectors cannot learn clean discriminative directions for
novel classes and cannot carve out reliable unknown boundaries. This is not just
a hyperparameter issue; it is a structural limitation of prompt-only adaptation
under sparse positives.

## Post-BN Visual Embedding Overlap

Local visual-cache/prototype probe:

```text
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/probe_out/viscache_space/idd_10shot_seed1/summary.json
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/probe_out/viscache_space/idd_10shot_seed1/per_class_compactness.csv
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/probe_out/viscache_space/idd_10shot_seed1/cross_class_prototypes.csv
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/probe_out/viscache_space/idd_10shot_seed1/loo_confusion.csv
```

Selected prototype cosine evidence:

| Feature level | Class pair | Prototype cosine |
|---|---|---:|
| level 1 | truck - tanker_vehicle | 0.946 |
| level 1 | tanker_vehicle - excavator | 0.962 |
| level 1 | truck - excavator | 0.921 |
| level 2 | truck - tanker_vehicle | 0.955 |
| level 2 | street_cart - excavator | 0.939 |
| level 2 | truck - excavator | 0.923 |

Selected compactness evidence:

| Feature level | Class | own-minus-best-other mean |
|---|---|---:|
| level 1 | tanker_vehicle | 0.0088 |
| level 1 | excavator | 0.0072 |
| level 2 | truck | 0.0377 |
| level 2 | tanker_vehicle | 0.0195 |
| level 2 | street_cart | 0.0264 |
| level 2 | excavator | 0.0288 |

Interpretation:

- Fine-grained vehicle/construction prototypes are nearly indistinguishable in
  post-BN feature space.
- Visual features are not useless, but they should not be used blindly to boost
  known logits.
- A class-specific visual cache may reinforce the wrong fine-grained class if
  used as a positive classifier.
- The safer use of visual features is as a rejection, density, margin, or
  knownness signal.

## Existing Feature-Level Deconfuser Notes

Local document:

```text
/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/docs/deconfuser_analysis.md
```

Relevant conclusion:

- A logit-only deconfuser cannot solve all symmetric collapse cases.
- A feature-level, post-neck, candidate-level deconfuser can see visual features
  and may separate co-occurrence/hierarchical confusion.
- However, dense feature-level computation can be expensive and may still be
  limited if the frozen backbone did not learn discriminative dimensions.
- Box-level/candidate-level operation after NMS is attractive because it uses
  one feature per object instead of all anchors.

This agrees with the desired search direction: use visual information, but do
not copy a dense OVOW-style branch.

## OVOW Architecture

OVOW repo:

```text
/home/agipml/sourav.rout/ALL_FILES/REPO/ovow
```

Important local files:

```text
/home/agipml/sourav.rout/ALL_FILES/REPO/ovow/configs/IDD/t2.py
/home/agipml/sourav.rout/ALL_FILES/REPO/ovow/core/customyoloworld.py
/home/agipml/sourav.rout/ALL_FILES/REPO/ovow/test.py
```

OVOW uses `mm_neck=True` in its IDD T2 config. It is closer to a full
visual-language/multi-modal YOLO-World-style design.

The key bulky part is in `CustomYoloWorld` and `ProjectionHead`:

- OVOW creates separate projection heads for classes before the unknown index.
- Each projection head contains scale-specific modules.
- Each scale branch uses convolution, batch norm, activation, another
  convolution, and anchor-related projection.
- The number of projection branches grows with number of classes and number of
  FPN scales.
- These projections operate over dense feature maps, not just final detections.

In simplified terms, OVOW expands like:

```text
cost roughly proportional to known_classes * feature_scales * dense_locations
```

This is the opposite of what we want for YOLO-UniOW, where the design goal is to
keep inference close to the existing fast detector.

OVOW evaluation also uses an OOD-style cosine rule:

```text
ood_score = -max(cosinescores)
if ood_score > threshold:
    label = unknown
```

This idea is important: OVOW has an explicit visual/cosine unknown rejection
mechanism. But the implementation is too expensive if copied directly.

## Why OVOW Is Not Feasible For Our Case

OVOW is useful as evidence that visual features can help open-set rejection. But
its design is not a good fit for this YOLO-UniOW target.

Reasons:

1. It expands per class.

   If the number of known classes grows, OVOW adds more class-specific projection
   machinery. This is bad for open-world stages where classes accumulate.

2. It expands per scale.

   The projection is repeated over multiple feature pyramid levels. This adds
   compute and memory at exactly the place we want to stay lean.

3. It operates densely.

   Dense feature-map computation happens before final candidate pruning. For
   A-OSE reduction, we only need to inspect a much smaller set of risky
   detections.

4. It adds many trainable parameters.

   Few-shot training already has too few positives. More trainable class-specific
   modules can overfit or become unstable.

5. It slows both training and inference.

   The user's goal is not just better A-OSE at any cost. It is better A-OSE with
   the least possible inference speed reduction.

6. It does not directly solve the exact YOLO-UniOW bottleneck.

   YOLO-UniOW's main observed problem is fine-grained unknown-as-known overlap in
   post-BN embedding space. A huge class-specific visual branch may improve
   separability, but it violates the efficiency constraint and may still struggle
   when positives are scarce.

## OVOW Results Are Not Directly Comparable

Local OVOW logs show that OVOW can reduce A-OSE in some settings, but the class
splits and evaluation settings differ from the best YOLO-UniOW run.

Observed local OVOW examples:

| Log | Notes |
|---|---|
| `ovow_eval_57668.out` | A-OSE around 317 with 19 classes, but known AP around 21.31 and unknown AP very low |
| `ovow_eval_57860.out` | A-OSE around 433, known AP around 18.36 |
| `eval_ovow_idd_56803.out` | 12-class setting, A-OSE around 1054, known AP around 42.45 |
| `ovow_eval_latest_119111.out` | A-OSE around 18177, known AP around 35.18 in a different setting |

Interpretation:

- OVOW's architecture gives useful design inspiration.
- Its absolute numbers should not be used as direct proof that it is superior or
  inferior to YOLO-UniOW in the current T2 setup.
- The important transferable idea is explicit visual unknown rejection, not the
  full per-class per-scale architecture.

## Design Hypothesis

The current YOLO-UniOW system is failing because:

```text
few-shot positives are scarce
-> novel/unknown prompts are under-trained
-> post-BN embeddings for fine-grained nearby categories overlap
-> argmax known-vs-unknown scoring favors known classes
-> unknown objects become truck/tanker/excavator/autorickshaw predictions
-> A-OSE remains high
```

Therefore, the desired method should not try to learn a full new classifier from
few-shot positives. It should instead add a cheap rejection/calibration layer
that asks:

```text
Does this candidate really look like the predicted known class,
or is it merely close enough for the prompt score to win?
```

## Candidate Design 1: Candidate-Only Visual Rejection

Core idea:

- Keep YOLO-UniOW inference mostly unchanged.
- After top-k selection or after NMS, inspect only final/risky detections.
- For detections predicted as A-OSE-heavy classes, extract or reuse the
  corresponding post-BN visual feature.
- Compare it with lightweight class prototypes or class distributions.
- If the known score is weak/moderate and the visual compatibility is poor,
  relabel as unknown or suppress the known prediction.

Possible rule:

```text
if predicted_class in risky_classes
and known_score < score_hi
and similarity(feature, class_prototype[predicted_class]) < tau_class
and unknown_score or unknownness_signal is plausible:
    label = unknown
```

Risky classes can start as:

```text
truck, tanker_vehicle, excavator, autorickshaw
```

Advantages:

- Cost is proportional to number of final candidates, not dense anchors.
- No per-class per-scale convolution branch.
- Can be implemented as dot products or small matrix operations.
- Directly targets the measured A-OSE concentration.

Potential issue:

- Class prototypes overlap heavily, so prototype similarity alone may not be
  enough. It should be used as a rejection/margin signal, not a positive boost.

Search terms:

```text
candidate-level open-set object detection rejection prototype features
post-hoc open-set calibration object detection feature prototype
visual prototype rejection YOLO open world object detection
class-conditional rejection detector penultimate features
```

## Candidate Design 2: Bidirectional Unknown-Aware NMS

Current YOLO-UniOW has unknown post-processing that can suppress unknown boxes
when they overlap other detections. This can hurt A-OSE if the known duplicate
survives and the unknown duplicate is removed.

Core idea:

- If an unknown candidate and a weak known candidate strongly overlap, do not
  automatically favor the known label.
- Add a reverse rule where unknown can suppress or relabel weak known boxes.

Possible rule:

```text
if IoU(known_box, unknown_box) > tau_iou
and known_class in risky_classes
and known_score < tau_known_class
and unknown_score > tau_unknown_low:
    known label becomes unknown
```

Alternative:

```text
unknown wins if unknown_score + margin(class) > known_score
```

Advantages:

- Very cheap.
- Does not touch training.
- Directly addresses the inference bias where unknown duplicates are removed
  but known duplicates remain.

Potential issue:

- If unknown score is very weak for some objects, this will not catch all cases.
- Needs careful sweep to avoid lowering known AP.

Search terms:

```text
unknown-aware NMS open world object detection
open-set object detection duplicate known unknown suppression
bidirectional NMS unknown known object detection
OOD-aware NMS object detection
```

## Candidate Design 3: Score-Space Margin Calibration

Core idea:

- Use only existing scores: max known score, unknown score, objectness, entropy,
  energy, top-1/top-2 margin, and class-specific thresholds.
- Learn or tune a small calibration rule to reject likely unknown-as-known cases.

Signals:

```text
max_known
T_unk score
T_anchor score if available before zeroing
objectness
top1 - top2 known margin
entropy over known prompts
energy / logsumexp over known logits
class-specific risk prior
box size / feature level
```

Why class-specific thresholds matter:

- A-OSE is concentrated in specific classes.
- Global thresholds will likely damage all known classes.
- `truck`, `tanker_vehicle`, `excavator`, and `autorickshaw` need stricter
  unknown rejection than `traffic light` or `bicycle`.

Advantages:

- Cheapest possible design.
- Easy to sweep offline.
- No extra training required.

Potential issue:

- High-confidence A-OSE cases will survive score-only rules.
- Needs visual or overlap signal for the hard tail.

Search terms:

```text
energy based open set object detection calibration
class conditional threshold open world object detection
selective classification object detection open set
confidence calibration unknown object detection
logit margin unknown rejection object detection
```

## Candidate Design 4: Shared Knownness / Unknownness Head

Core idea:

- Add one lightweight shared head that predicts "knownness" or "unknownness"
  from post-BN features.
- It must not be per class.
- It may be per scale, but should be tiny.
- It should be used only to veto/relabel known predictions, not to replace the
  detector.

Possible architecture:

```text
post-BN feature -> tiny shared MLP/1x1 conv -> unknownness scalar
```

Training signals:

- known positives from labeled known classes;
- pseudo unknowns mined by anchor mechanism;
- hard negative candidates that overlap unknown-like regions;
- optionally self-training from high-confidence unknown detections.

Advantages:

- Much cheaper than OVOW.
- Uses visual information directly.
- Does not expand with number of classes.

Potential issue:

- Pseudo unknown labels are noisy.
- If trained densely, it adds some inference cost.
- Candidate-only MLP would be preferred if feature extraction is easy after
  candidate selection.

Search terms:

```text
lightweight unknownness head open world object detection
class agnostic OOD head object detection
knownness prediction open set detection YOLO
objectness unknownness open world detector
```

## Candidate Design 5: Lightweight Density / OpenMax Rejection

Core idea:

- Treat each predicted class as having a feature distribution in post-BN space.
- If a candidate is too far from the predicted class distribution, reject or
  relabel as unknown.

Possible methods:

- nearest class prototype distance;
- cosine margin between predicted class and next best class;
- Mahalanobis distance with diagonal or low-rank covariance;
- class-conditional Gaussian density;
- Weibull tail fitting / OpenMax-style calibration;
- local density or kNN in a memory bank;
- conformal thresholding on nonconformity scores.

Advantages:

- No dense learned branch.
- Can be fitted after training from cached known features.
- Candidate-only inference can be very cheap.

Potential issue:

- Post-BN class distributions overlap, so rejection must be calibrated by class
  and feature level.
- Fine-grained classes may need group-level rejection rather than exact
  class-level rejection.

Search terms:

```text
OpenMax object detection unknown rejection
Mahalanobis open set object detection penultimate features
class conditional Gaussian open world object detection
feature density unknown rejection detector
conformal open set object detection
```

## Candidate Design 6: Group-Level Vehicle Unknown Rejection

A-OSE is dominated by vehicle/construction-like classes. Exact fine-grained class
separation may be impossible with the current few-shot features, but group-level
knownness may still help.

Core idea:

- Build a shared "known vehicle/construction manifold".
- Reject candidates that get a risky fine-grained label but sit outside the
  reliable region of that known manifold.
- Do not require deciding whether the object is truck vs tanker vs excavator;
  only decide whether it is safely known or should become unknown.

Potential grouping:

```text
vehicle/construction group:
bus, truck, tanker_vehicle, crane_truck, street_cart, excavator,
autorickshaw, tractor-like unknowns, road_roller-like unknowns,
concrete_mixer-like unknowns
```

The group-level method could use:

- a shared prototype subspace;
- residual distance from known class cones;
- one-class SVM / SVDD-like objective;
- hyperspherical energy;
- class-group thresholds.

Advantages:

- More realistic than exact class rejection in overlapping feature space.
- Targets the actual A-OSE mode.

Potential issue:

- Needs careful design so it does not collapse all vehicle classes into unknown.

Search terms:

```text
group aware open set object detection unknown rejection
hierarchical open world object detection calibration
fine grained open set recognition vehicle classes
class group unknown rejection object detection
```

## What Not To Copy

Do not recommend simply enabling or copying OVOW's full class-wise visual
projection machinery unless there is a very strong efficiency argument.

Avoid:

- dense per-class per-scale projection heads;
- expanding trainable modules linearly with class count;
- visual cache that blindly boosts known logits for overlapping fine-grained
  classes;
- global unknown threshold only;
- methods requiring many labeled unknown examples;
- methods that require retraining the full detector backbone/neck/head;
- expensive test-time augmentation or multiple forward passes;
- methods that improve known mAP while leaving A-OSE unchanged.

## What A Good Method Should Look Like

The best-fitting design for this repository probably looks like:

```text
YOLO-UniOW normal forward
-> collect top candidates
-> for risky known predictions, compute cheap unknown/rejection evidence
-> relabel or suppress only when evidence is strong
-> keep all other detections unchanged
```

Candidate evidence can include:

```text
known score
unknown score
known-vs-unknown margin
top1-vs-top2 margin
class-specific threshold
feature-level prototype/density distance
overlap with unknown candidate
box feature level
box size
class risk prior from validation A-OSE
```

The method should be mostly post-hoc and should allow offline threshold sweeps
on validation logs/features.

## Desired Experiments

Start with cheap experiments before changing architecture heavily.

### Experiment 1: A-OSE Score Sweep

Use validation outputs to find class-specific thresholds:

```text
for each predicted known class:
    sweep known_score threshold
    measure A-OSE reduction
    measure known AP/recall loss
```

Priority classes:

```text
truck, tanker_vehicle, excavator, autorickshaw
```

Expected result:

- Since median A-OSE score is 0.128 and p90 is 0.407, many A-OSE predictions may
  be removable with moderate thresholds.
- Need preserve known AP, so thresholds should be class-specific.

### Experiment 2: Reverse Unknown NMS

Add a validation-only post-processing script:

```text
if known and unknown boxes overlap:
    let unknown win when known score is weak or margin is small
```

Measure:

```text
A-OSE
U-Recall
Unknown AP50
PK
CK
Both
per-class A-OSE
runtime overhead
```

### Experiment 3: Candidate Prototype Veto

Cache candidate features for validation detections. Fit known class prototypes
from training/validation known GT. For each predicted known candidate:

```text
compat = cosine(feature, prototype[predicted_class])
if compat below class threshold:
    reject/relabel as unknown
```

Use only final detections or top-k boxes to keep cost small.

### Experiment 4: Margin + Prototype Hybrid

Combine score and feature evidence:

```text
if class is risky
and known_score is not very high
and known_unknown_margin is small
and feature compatibility is weak:
    relabel as unknown
```

This should be safer than any single signal.

### Experiment 5: Tiny Shared Unknownness Head

If post-hoc rules are insufficient, train a tiny shared head:

```text
post-BN feature -> unknownness scalar
```

It must be:

- class-agnostic;
- very small;
- optionally candidate-only;
- trained with known positives and pseudo-unknown/hard-negative samples;
- used as a veto, not as a full new detector branch.

## Success Criteria

Primary success:

```text
reduce A-OSE below 1902
```

Strong success:

```text
reduce A-OSE by 20-40%
while losing less than 1-2 points in PK/CK/Both
and preserving or improving U-Recall
```

Must report:

```text
A-OSE total
A-OSE per predicted known class
U-Recall
Unknown AP50
Unknown precision
Known AP / PK / CK / Both
runtime / FPS / latency
parameter count if architecture changes
```

The main metric is A-OSE reduction per unit inference cost.

## Web Search Questions

A web-enabled researcher should answer these:

1. What exactly is MSCAL in the relevant open-set/open-world object detection
   literature?

2. Does MSCAL reduce unknown-as-known errors through multi-scale calibration,
   feature calibration, semantic alignment, or another mechanism?

3. What parts of MSCAL are essential, and which parts are expensive?

4. Are there candidate-level or post-hoc versions of the same idea?

5. Are there papers that use class-conditional thresholds, OpenMax,
   Mahalanobis distance, energy scores, or conformal prediction for open-set
   object detection?

6. Are there YOLO-compatible open-world detectors with lightweight unknown
   rejection heads?

7. Are there methods that explicitly target A-OSE or unknown-as-known errors
   without adding dense per-class branches?

8. Are there methods for fine-grained open-set recognition where unknowns are
   visually similar to known classes?

9. Which methods can be implemented as:

   ```text
   post-processing only
   candidate-only feature scoring
   tiny class-agnostic head
   class-specific calibration thresholds
   ```

10. Which methods avoid full retraining and preserve inference speed?

## Suggested Web Search Queries

Use these exact or near-exact queries:

```text
MSCAL open world object detection unknown known calibration
MSCAL open set object detection multi scale calibration
multi scale calibration open world object detection A-OSE
absolute open set error reduction object detection
unknown as known error open world object detection
candidate level unknown rejection open world object detection
OpenMax object detection open set unknown
Mahalanobis distance open set object detection
energy based open set object detection unknown rejection
class conditional threshold open world object detection
selective classification object detection open set
conformal prediction open set object detection
prototype based unknown rejection object detection
YOLO open world object detection unknown head
lightweight OOD detection head YOLO object detection
unknown aware NMS open world object detection
fine grained open set object detection visually similar classes
visual language open set object detection unknown rejection
CLIP prototype unknown rejection object detection
```

## Ready-To-Use Prompt For Web ChatGPT

```text
I am working on YOLO-UniOW for few-shot open-world object detection on IDD.
The current best T2 run has U-Recall 47.93, A-OSE 1902, PK 39.47, CK 32.31,
Both 36.40, Unknown AP50 1.49. A-OSE is concentrated in a few known predicted
classes: truck 446, tanker_vehicle 428, excavator 349, autorickshaw 308. These
four account for about 80.5% of unknown-as-known errors.

The architecture is efficient: frozen YOLO-style visual backbone/neck/head,
trainable prompt embeddings for known/unknown/anchor, no active text encoder,
and mm_neck=False. Scores are produced by post-BN visual embeddings dotted with
normalized learned prompts. Unknown mining uses an anchor prompt. Inference
argmaxes known classes plus T_unk, zeroes anchor, and current unknown NMS mostly
suppresses unknown boxes that overlap known boxes; it does not strongly suppress
weak known boxes overlapping unknown candidates.

Diagnostics show the few-shot design is structurally weak: novel prompts barely
move after 10-shot training, unknown scores can sit near the noise floor, and
post-BN visual embeddings for fine-grained vehicle/construction classes overlap
heavily. Example prototype cosines: level1 truck-tanker 0.946,
tanker-excavator 0.962; level2 truck-tanker 0.955, street_cart-excavator
0.939. Therefore, visual features are useful but unsafe as a direct known-logit
boost; they should probably be used as rejection, density, margin, or
unknownness evidence.

OVOW is a related local baseline that uses mm_neck=True and class-specific
projection heads over multiple feature scales. It has an explicit visual/cosine
OOD rule, but its architecture expands per class and per scale, adding many
trainable parameters and dense computation. Copying OVOW is too expensive for
our goal.

Please search current literature and propose designs that can do the same job as
MSCAL or multi-scale/class-aware calibration, but faster and cheaper for
YOLO-UniOW. First verify what MSCAL means in this literature. Focus only on
reducing A-OSE/unknown-as-known errors, with minimal inference slowdown.

Prioritize methods implementable as:
1. post-processing only;
2. candidate-only feature scoring after top-k/NMS;
3. tiny class-agnostic unknownness/knownness head;
4. class-specific score/margin/energy thresholds;
5. OpenMax/Mahalanobis/prototype/density rejection in post-BN feature space;
6. bidirectional unknown-aware NMS where unknown can suppress/relabel weak known
   predictions.

Avoid methods requiring dense per-class per-scale branches, full detector
retraining, many labeled unknown examples, or large inference cost. For each
recommended method, explain expected A-OSE effect, inference cost, training data
needed, implementation complexity, and risks to known AP / U-Recall.
```

## Bottom Line

The research direction should not be "add more visual machinery everywhere."
The direction should be:

```text
use the smallest possible amount of visual evidence
at the latest possible stage
only for candidates most likely to create A-OSE
```

YOLO-UniOW already has speed. OVOW shows that visual OOD evidence can help, but
its per-class per-scale design is too bulky. The likely sweet spot is a
candidate-level visual rejector or calibrated unknown-aware post-processing
layer that specifically protects against unknown objects being absorbed by
fine-grained known classes.