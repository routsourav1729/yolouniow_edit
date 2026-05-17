Architecture Analysis: YOLO-UniOW vs CED-FOOD

## YOLO-UniOW: Training

### Class Embedding Structure

The model maintains K+2 learnable embeddings:
- `[0 … K-1]` → known class tokens (T_1, …, T_K)
- `[-2]` → T_unk (unknown token)
- `[-1]` → T_anchor (objectness/anchor token)

For each anchor the head outputs K+2 cosine-similarity logits, one per embedding.
YOLO-UniOW: Training

Class Embedding Structure

The model maintains K+2 learnable embeddings:

- `[0 … K-1]` → known class tokens (T_1, …, T_K)
- `[-2]` → T_unk (unknown token)
- `[-1]` → T_anchor (objectness/anchor token)

For each anchor the head outputs K+2 cosine-similarity logits, one per embedding.


Known-class training (every batch)

TAL (Task-Aligned Assignment) runs in `loss_by_feat` every batch. For each anchor, it computes:

$$
	ext{alignment} = \text{cls\_score}_k^{\alpha} \cdot \text{IoU}_k^{\beta}
$$

The top-k anchors per GT box win and are positives. Their `assigned_scores` channel for the matched class gets:

$$
s_k = \frac{\text{alignment} \cdot \text{IoU}}{\text{alignment}_{\max} + \varepsilon} \in (0,1]
$$

All other K+2 channels for that anchor get target = 0. The classification loss is then soft BCE over the full B×N×(K+2) tensor:

$$
\mathcal{L}_{cls} = \frac{1}{S} \sum_{b,n,k} \text{BCE}(\hat{p}_{b,n,k},\ s_{b,n,k})
$$

where

$$
S = \sum_{b,n,k} s_{b,n,k}
$$

This is not per-class sequential — all K+2 channels for all anchors are trained in one joint BCE. But the TAL metric means each anchor is only positively supervised for its best-matching class. Unmatched class channels get target=0 and are therefore penalised down.


Pseudo-Unknown Mining (the gatekeeper Φ — every batch)

This runs every training step inside `loss_by_feat` on the current (detached) predictions. Conceptually:

$$
\begin{aligned}
	ext{anchor\_scores} &= \sigma(\text{logit}_{T_{anchor}}) \\
	ext{max\_known} &= \max_k \sigma(\text{logit}_k) \\
	ext{unknown\_mask} &= (\max\mathrm{IoU\_to\_GT} < \tau_{iou}) \\
&\quad\wedge\ (\text{not TAL-assigned}) \\
&\quad\wedge\ (\text{anchor\_scores} > \tau_{score}) \\
&\quad\wedge\ (\text{anchor\_scores} > \text{max\_known})
\end{aligned}
$$

Anchors passing all four conditions become pseudo-unknown candidates. Their T_unk channel (`assigned_scores[:, :, -2]`) is filled with the anchor objectness score as the soft target:

$$
	ext{target}_{T_{unk}} = \sigma(\text{logit}_{T_{anchor}}) \in (0,1)
$$

The classification loss for these anchors pulls T_unk's output up in proportion to how "object-like" the anchor looks (via `T_anchor`), while pressing all K known-class channels down (they stay at target=0 in `assigned_scores`).

This runs every batch — the candidate set changes each step as the model evolves. It is NOT a one-time precomputation.

YOLO-UniOW: Inference

At test time:

- `flatten_cls_scores[:, :, -1] = 0` — the `T_anchor` channel is zeroed completely (it was only for training supervision).
- Remaining: K known channels + 1 `T_unk` channel, all passed through sigmoid.
- Standard `max(1)` or `filter_scores_and_topk` → each anchor's winning class is the argmax across all K+1 active channels.

So a single anchor gets ONE label. If `T_unk` wins → that anchor is an "unknown" detection. If class `k` wins → it is a known detection.

Then `_unknown_post_process` applies a second NMS that removes unknown predictions that are nearly identical to known boxes (high IoU threshold, e.g., 0.99).


Your Core Question: Can the Same Anchor Count for Both Known and Unknown Recall?

Strictly per anchor: no. Each anchor has one winning class (argmax over K+1). The same anchor object cannot simultaneously be unknown AND known.

But across nearby anchors: yes, this is a real problem. YOLO has dense overlapping anchors at multiple strides. Two anchors at slightly different spatial positions can each partially overlap the same GT box:

- Anchor A (stride-8, position p) → `T_unk` wins → unknown detection
- Anchor B (stride-16, position p±δ) → class `k` wins → known detection

Both Anchor A and Anchor B can overlap the same GT, so the same physical object contributes to both unknown recall and known recall. The `_unknown_post_process` only suppresses unknowns that overlap knowns at very high IoU (e.g., ≥ 0.99), which may not catch the stride-8 vs stride-16 case.

Is this fundamentally wrong? Yes, in terms of evaluation integrity:

- If the object is a known class but the model's known-class embedding didn't fire strongly enough (`T_unk` won on one anchor), that counts as a spurious unknown recall.
- If the object is a true unknown but a known class fires on a nearby anchor, you get known recall for something you never annotated — which inflates known AP.
- Standard OWOD evaluation measures recall separately for known and unknown and expects no anchor to fire for both. The double-counting inflates both known and unknown recall for the same object.


CED-FOOD: How Unknown Works (Inference)

CED-FOOD is a two-stage Detectron2 model (RPN → ROI features → classification head). The architecture is fundamentally different:

Score vector layout:

$$
\begin{aligned}
	ext{align\_scores} &= \text{normalized\_x} \cdot F\_\text{normalize}(\text{text\_emb}, p=2, dim=1)^T \\
	ext{un\_scores} &= \text{Linear}(\text{normalized\_x}) \\
	ext{align\_bg\_score} &= \text{Linear\_bg}(\text{normalized\_x}) \\
	ext{logits} &= [\text{align\_scores},\ \text{un\_scores},\ \text{align\_bg\_score}] \\
	ext{probs} &= \mathrm{softmax}(\text{logits})
\end{aligned}
$$

Unknown is a separate learned linear head (`un_score`), not a text embedding prototype. It is one additional class channel in the softmax, parallel to all K known classes.

Training the unknown head (UpLoss, every batch):

- For FG proposals (IoU > threshold with a GT): `target[T_unk] = 1 - objectness_score` → push unknown score DOWN in proportion to RPN confidence. `target[GT_class] = objectness_score`.
- For BG proposals: `target[T_unk] = objectness_score` → push unknown score UP for object-like background proposals.

This is a soft evidential/Dirichlet-style cross-entropy, run every batch. The unknown head learns to fire high on "things that look object-like but have no GT annotation."

Inference:

1. `scores[:, :-1]` — drop the background column, keeping K+1 classes (K known + 1 unknown).
2. `batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)` — NMS is per class-id. Each proposal has one class (argmax of K+1), so the same proposal cannot be both known and unknown.
3. `unknown_aware_nms` (optional, with IoU threshold ≈ 0.9): if an unknown prediction overlaps a known prediction, the lower-scoring one is suppressed.

Key structural difference: Because CED-FOOD is two-stage, each RPN proposal is a single region. Within that region, the softmax produces one label (the argmax). Two different RPN proposals can overlap the same GT — one labeled known, one labeled unknown — but the per-class NMS and unknown-aware NMS are designed to resolve this.


Direct Comparison (no tables)

- **Unknown representation:** YOLO-UniOW uses a learned text-like embedding `T_unk` that competes with known tokens via cosine similarity. CED-FOOD uses a learned linear `un_score` scalar channel in a softmax.
- **Training target for unknown:** YOLO-UniOW uses the `T_anchor` sigmoid (anchor objectness) as the soft target for anchors passing gatekeeper Φ — mined every batch. CED-FOOD uses UpLoss: BG proposals get `target[unk]=objectness`, FG proposals get `target[unk]=1-objectness` (soft evidential targets).
- **Same object → known + unknown recall?:** YOLO-UniOW: possible across overlapping anchors at different strides; the `_unknown_post_process` at IoU≥0.99 is the only guard. CED-FOOD: less likely — two-stage RPN proposals, per-class NMS and `unknown_aware_nms` at IoU≈0.9 reduce contradictions.
- **Unknown vs known score competition:** Both use an argmax: YOLO-UniOW has unknown compete directly with known classes via cosine logits; CED-FOOD uses softmax. `un_score` gradients are independent of CLIP known embeddings in CED-FOOD.
- **Anchor/objectness proxy:** YOLO-UniOW relies on `T_anchor` as a proxy for "objectness" during mining and soft-targeting. CED-FOOD uses RPN objectness as a soft target factor inside `UpLoss`.
- **Post-inference unknown suppression:** YOLO-UniOW uses an `_unknown_post_process` (very high IoU threshold, e.g., 0.99). CED-FOOD uses `unknown_aware_nms` (more aggressive, e.g., IoU≈0.9).

The core conceptual issue you identified — the same anchor counting for both known and unknown recall — is structurally present in YOLO-UniOW due to the dense overlapping multi-stride anchors. The conservative unknown-NMS at 0.99 IoU is deliberately conservative to avoid suppressing genuine unknowns, but this means neighboring anchors (e.g., stride-8 vs stride-16) from the same object region can each produce detections of different class types, causing double recall. CED-FOOD is slightly better here because RPN already suppresses heavily overlapping proposals via its own NMS before classification.

Direct Comparison

| Aspect | YOLO-UniOW | CED-FOOD |
| --- | --- | --- |
| Unknown representation | Learned text-like embedding `T_unk` (one token competing with known tokens via cosine similarity) | Learned linear layer `un_score` (separate scalar output) |
| Training target for unknown | `T_anchor` sigmoid score (objectness) as soft target for anchors passing gatekeeper Φ — every batch | Soft EDL target: BG proposals get `target[unk]=objectness`, FG proposals get `target[unk]=1-objectness` |
| Same object → known + unknown recall? | Yes, possible across overlapping anchors at different strides; unknown-NMS at IoU≥0.99 is the only guard | Less likely — two-stage means each proposal is a distinct region, per-class NMS + `unknown_aware_nms` at IoU=0.9 cleans it up |
| Unknown vs. known score competition | Unknown competes directly in argmax with K known classes → if any known class fires stronger, unknown loses | Same: softmax over K+1, argmax wins — but `un_score` head is independent of known class embeddings, so its gradient doesn't interfere with known class heads |
| Anchor score as unknown proxy | Yes — `T_anchor` acts as a "foreground but no class" proxy, used both for mining and as soft target | No explicit objectness proxy; RPN objectness enters only as soft target weight in `UpLoss` |
| Post-inference unknown suppression | `_unknown_post_process` at IoU≥0.99 | `unknown_aware_nms` at IoU=0.9 (more aggressive) |
