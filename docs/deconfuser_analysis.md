# Deconfuser feasibility & confusion-pattern analysis

Source runs:
- `slurm_logs/analysis/probe/probe_114637.out` — IDD test (multi-prompt v2)
- `slurm_logs/analysis/probe/probe_114638.out` — nuImages test (multi-prompt v2)
- `slurm_logs/analysis/probe/probe_114789.out` — IDD test (compare zs vs T2)
- `slurm_logs/analysis/probe/probe_114796.out` — nuImages test (compare)
- `slurm_logs/analysis/probe/probe_114822.out` — IDD few-shot **train** (multi-prompt v2)
- `slurm_logs/analysis/probe/probe_114823.out` — nuImages few-shot **train** (failed: missing image files in active `JPEGImages/nuOWODB/`; files live in `nuOWODB_symlink_backup_*`)

Probe code now pre-filters missing files, so the next nuImages-train run will succeed; for this report nuImages-train is omitted and only IDD-train is compared against test.

---

## 1. Train vs. test — is the confusion structural or random?

Per-class cross-class winner rate from the multi-prompt probe (sum-aggregated; see §2 caveat):

| GT class       | IDD train | IDD test | Δ (train−test) | Note |
|----------------|----------:|---------:|---------------:|------|
| **car**            | 0.613 | 0.380 | +0.23 | T1 base |
| **motorcycle**     | 0.588 | 0.784 | −0.20 | T1 base |
| **rider**          | 0.260 | 0.326 | −0.07 | chronic loser |
| **person**         | 0.625 | 0.593 | +0.03 | T1 base |
| **autorickshaw**   | 0.062 | 0.135 | −0.07 | chronic loser |
| **bicycle**        | 0.667 | 0.539 | +0.13 | small-N train |
| **traffic_sign**   | 0.630 | 0.767 | −0.14 | T1 base |
| **traffic_light**  | 1.000 | 0.877 | +0.12 | small-N train |
| **bus**            | 0.382 | 0.584 | −0.20 | T2 novel |
| **truck**          | 0.401 | 0.703 | −0.30 | T2 novel |
| **tanker**         | 0.141 | 0.249 | −0.11 | chronic loser |
| **crane_truck**    | 0.884 | 0.574 | +0.31 | small-N (138) |
| **street_cart**    | 0.927 | 0.691 | +0.24 | small-N (206) |
| **excavator**      | 0.025 | 0.092 | −0.07 | chronic loser |

Train sample is the 53-image few-shot pool (post 10-shot cap), so absolute numbers per class are noisy when N<300 (bicycle, crane_truck, street_cart). What matters is the **ranking** of who consistently wins vs. loses.

**The confusion is structural, not random.** Three signals:

1. **Same four classes are bottom-ranked in both splits**: rider, autorickshaw, tanker, excavator. These never break out of the 0.0–0.30 winner band in either split. If the head's failures were noise we would expect the bottom of the train list to differ from test.
2. **Stealer identities transfer**: in IDD-train top-3 stealers, `street_cart` steals from autorickshaw / motorcycle / person / car / bus, and `crane_truck` steals from tanker / excavator / truck. The v2 IDD-test stealing matrix names the same two prompts as the dominant cross-class winners. The model learnt to over-fire those two prompt embeddings on training data and the bias survives at test time.
3. **Direction of the tuning effect is preserved**: compare-mode shows tuning lifts T1 winner rate by +0.23 on IDD test; on IDD train tuning achieves *higher* per-class scores for the same set (car, motorcycle, person are the clear top-3 in both). The classes the BCE head fails to discriminate at training time are the same classes it fails to discriminate at test time.

**Conclusion**: the confusion pattern is a property of the trained head, not a sampling artefact of the test set. A targeted intervention (loss-side or post-hoc) is in principle reachable.

---

## 2. Probe consistency fix already applied

The multi-prompt mode previously aggregated per-class score as `sum(sigmoid(prompt))`. A 3-prompt class could reach 3.0 while a 1-prompt class capped at 1.0, biasing winner-rate. Now changed to `.amax(dim=2)` (see [tools/probe_text_visual.py:653-660](tools/probe_text_visual.py#L653-L660)). Re-run the multi-prompt probes to refresh the absolute numbers; the ranking is unlikely to change because in IDD only `traffic_sign` had 1 prompt while every confusable class had 3. Compare-mode is unaffected.

---

## 3. Common confusion pattern (cross-dataset, cross-split)

Three reproducible failure modes:

**A. Hierarchical sub-type collapse.** Sub-class is a strict visual superset of its parent: child / wheelchair / stroller / police_officer / construction_worker → adult; bus.bendy → bus.rigid; rider → motorcycle. In compare mode (T2-tuned) on nuImages, the *steal_rate* for child/wheelchair/stroller/police/worker stays at 93–100 % even after tuning; the visual feature for these anchors is genuinely contained in the adult feature, so the BCE head has no separating direction to learn.

**B. Visual co-occurrence binding.** TAL assigns soft positives to anchors that lie inside / next to multiple objects at the same pyramid level — rider co-fires with motorcycle, trailer with truck, stroller with adult. The head receives "both classes are 0.5–1.0 positive" gradient on the same visual feature, which directly trains it to fire both classes for that feature at inference. This is the single biggest source of the "both compare modes still steal heavily on novel" finding.

**C. Rare-class starvation.** For classes with N<700 in nuImages-test (ambulance 58, wheelchair 15, child 549, police_car 712, stroller 264, personal_mobility 931), tuning has no signal to work with: the embedding row of `model.embeddings` stays close to its random init direction, the per-class steal rate is 95–100 %, and `d_steal` from compare-mode is ≈ 0 for all of them. Tuning is *unable* to fix these classes because there is nothing to fix from.

These three patterns collectively explain the empirical compare-mode result that base classes gain +0.13 / +0.23 winner rate and lose 11–15 % stealing, while novel classes are flat.

---

## 4. Why a logit→logit deconfuser cannot help on its own

Setup: a deconfuser is a small module `g(z) → z'` that takes the BCE head's pre-sigmoid logits `z ∈ R^{N_C}` per anchor and returns corrected logits.

The constraint that breaks it is information loss before `z`. The head produces `z = BN(F_vis) · E_text · s + b` where the only per-anchor variable is the post-neck visual feature `F_vis`. For two anchors that differ in *which* class is correct (say one is a child, one is an adult, both at the same pyramid level), the head's training signal under TAL was "lift z_adult and lift z_child for both anchors" because the TAL soft target is multi-positive on the overlapping anchor band. The trained head therefore produces `z_adult ≈ z_child` for both anchors. No monotonic, anchor-independent mapping `g(z)` can flip the argmax: whatever `g` does to make child win on the child anchor will also make child win on the adult anchor.

Empirically this is exactly what compare-mode shows for nuImages T2: tuning has already optimised the embedding rows against a fixed visual backbone, and the residual confusion that remains is anchor-specific information that the logit vector simply does not contain.

A deconfuser can correct *systematic, asymmetric* leakage (e.g. truck consistently beats bus by +0.05 → subtract 0.05 from truck's logit) — this is what an empirical confusion-matrix subtraction does. It cannot correct *symmetric collapse* (adult vs child), which is the dominant residual on nuImages T2.

---

## 5. Does feature-level (post-neck, candidate-only) caching change the picture?

The user's variant: cache `F_vis` from the neck for anchors that pass an objectness/proposal gate, and run the deconfuser as `g(F_vis, z) → z'`. This is materially different from the logit-only case.

**What this gains.** Now the deconfuser sees the actual visual feature, not a 14-dim summary of it. For child vs. adult, the two anchors *do* have different `F_vis` (the child anchor sits on a smaller box at higher pyramid level, with different texture / pose statistics). A learned `g` over `F_vis` can in principle separate them — this is no longer a degenerate problem.

**What this still cannot fix without retraining the backbone.**
- The neck features are themselves the output of a backbone trained against `model.embeddings` whose adult/child rows ended up close in cosine space (compare-mode evidence: child d_steal=0.005, child winner=0.32 even with the tuned T2 row). The visual backbone was never asked during training to separate child from adult — it was asked to produce a feature that fires the adult-text-direction. So while child/adult anchors *do* differ in `F_vis`, the dimensions along which they differ are *not the dimensions the backbone made discriminative*. A 1-layer post-hoc head on `F_vis` will recover only what the backbone preserved by accident.
- Filtering to "candidates with localisation proposals" reduces compute but does not change the above — the same features are still being looked at.
- Box-level vs. anchor-level: caching at proposal level (after NMS-ish filter) helps because you now have one feature per object instead of one per anchor, killing TAL's multi-positive ambiguity. This is a real positive — the supervision for the deconfuser becomes a clean one-class-per-box problem instead of an anchor-soft-target problem.

**Net assessment.** Feature-level + candidate-only is the strongest deconfuser variant and is genuinely worth attempting on (i) hierarchical sub-type pairs where the backbone is likely to have preserved some shape/scale cue (child smaller than adult, stroller has wheels, wheelchair has wheels at hip height) and (ii) co-occurrence pairs (rider vs. motorcycle vs. person) where bounding-box geometry alone is informative. It will *not* help on the rare-class starvation set, because there is no training data for the deconfuser to learn from either.

Realistic upper bound: a feature-level post-hoc head can probably recover 30–50 % of the symmetric-collapse confusions on classes with N ≥ 1000, and ≤ 10 % on classes with N < 500. Stealing-by-co-occurrence is the easiest target; sub-type collapse on rare classes is the hardest.

---

## 6. Pattern summary — what is fixable, what is not

| Pattern | Where it shows up | Fixable post-hoc? | Why |
|---|---|---|---|
| Asymmetric leakage between trained classes | truck→bus, motorcycle→street_cart on IDD test | **Yes (logit-level)** | Confusion matrix is sparse and direction-stable across train/test; subtraction works |
| Co-occurrence binding | rider↔motorcycle↔person; trailer↔truck; stroller↔adult | **Maybe (feature-level only)** | Box geometry differs; needs `F_vis` not just `z` |
| Hierarchical sub-type collapse, rare class | child / wheelchair / stroller / personal_mobility | **No** | <1000 anchors and the backbone never had a separation signal — feature-level head can't learn from nothing |
| Hierarchical sub-type collapse, common class | adult-pedestrian under multi-prompt; bus.rigid vs. bus.bendy | **Partial (feature-level)** | Enough data; geometry/scale cues plausibly survive in `F_vis` |
| Rare-class starvation in general | ambulance, wheelchair, etc. on nuImages test | **No** | Both backbone and deconfuser would be data-limited |
| Multi-prompt over-broadening (e.g. "scooter") | nuImages personal_mobility prompt set | **Yes (prompt-level)** | Drop or tighten the prompt — done at config, no model change |

---

## 7. Directions worth exploring (no proposed solutions, just where the leverage seems to be)

- The dominant residual after T2 tuning is concentrated in (a) co-occurrence pairs and (b) hierarchical sub-types where the parent class has 100× more samples. These two failure modes have *different* root causes — one is TAL multi-positive, the other is class-imbalance + embedding-row collapse. They likely need different interventions; bundling them into one "deconfuser" may underperform either.
- Compare-mode `d_steal` is essentially zero for every nuImages T2 class. This is informative on its own: it means the T2 training run did not move stealing on those classes at all. The 10-shot signal is being absorbed by the embedding rows of *base* classes (which is why T1 improves) rather than by novel rows. This is a finetuning-recipe property, not a head-architecture property.
- The few-shot training-set probe is the natural place to look for "does the head ever solve this confusion, even on the data it saw". For IDD, classes that lose at test time also lose on train (rider 0.26 train / 0.33 test; tanker 0.14 / 0.25; excavator 0.03 / 0.09). The head never learns to discriminate them at train time, so post-hoc correction at test time is bounded by what the train-time features already encode.
- Per-class steal direction is not symmetric. crane_truck and street_cart are *aggressors* (high `off_wins`, low net), tanker and excavator are *victims* (high steal received, no aggression). A single deconfuser parameterisation may need to treat aggressor-suppression and victim-rescue as distinct training objectives.
- The probe consistency fix (sum→max) should be re-run before any new modelling decision is made — the absolute winner-rate numbers in the v2 multi-prompt report are slightly biased toward 3-prompt classes, though the ranking-based observations above are unaffected.
