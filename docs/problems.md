# YOLO-UniOW OWOD Problems and Observations

This note is meant to be copied into an LLM as context. It summarizes the observed problems from the IDD and nuScenes/nuOWODB Task-2 experiments, comparing zero-shot behavior against the few-shot tuned T2 model, then connecting those results to model-behavior analysis from the confusion/anchor diagnostics.

## Source Logs

IDD zero-shot vs tuned:

- Zero-shot T2 eval: `/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/zeroshot_uniow_zs_117109.out`
- Few-shot tuned T2 eval: `/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/idd_t2_allfs_116614.out`
- Behavior analysis: `/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/analysis/idd_conf1.out`

nuScenes/nuOWODB zero-shot vs tuned:

- Zero-shot T2 eval: `/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/zeroshot_uniow_zs_111993.out`
- Few-shot tuned T2 eval: `/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/nuscenes/owod_l_t2_111862.out`
- Behavior analysis: `/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/analysis/nuscneesconf1.out`

## IDD: Zero-Shot vs Few-Shot Tuned

The IDD T2 setup has 8 previous/base classes and 6 current/novel classes.

Base classes:

- `car`, `motorcycle`, `rider`, `person`, `autorickshaw`, `bicycle`, `traffic sign`, `traffic light`

Current/novel T2 classes:

- `bus`, `truck`, `tanker_vehicle`, `crane_truck`, `street_cart`, `excavator`

Unknown classes:

- `pole`, `animal`, `tractor`, `concrete_mixer`, `pull_cart`, `road_roller`

### Aggregate Metrics

| Metric | Zero-shot | Few-shot tuned | Change |
| --- | ---: | ---: | ---: |
| Prev/base AP50 | 29.14 | 39.48 | +10.34 |
| Current/novel AP50 | 23.81 | 27.29 | +3.48 |
| Known AP50 | 26.85 | 34.25 | +7.40 |
| Unknown AP50 | 0.032 | 0.684 | +0.652 |
| Unknown Recall @50 | 8.76 | 39.64 | +30.88 |
| A-OSE | 3139 | 1709 | -1430 |

High-level observation: tuning clearly improves aggregate known AP, unknown recall, and A-OSE on IDD. However, this aggregate hides a serious class-level regression: the large vehicle novel classes `bus` and `truck` get worse after few-shot tuning, while smaller/specialized novel classes improve.

### IDD Novel-Class AP50

| Novel class | Zero-shot AP50 | Few-shot tuned AP50 | Observation |
| --- | ---: | ---: | --- |
| `bus` | 50.790 | 42.744 | regresses by -8.046 |
| `truck` | 37.804 | 23.167 | regresses by -14.637 |
| `tanker_vehicle` | 9.937 | 12.657 | improves slightly |
| `crane_truck` | 9.141 | 27.800 | large improvement |
| `street_cart` | 19.310 | 39.361 | large improvement |
| `excavator` | 15.873 | 18.015 | small improvement |

This suggests the few-shot update is not simply "learning T2 better". It is selectively helping classes whose visual/semantic anchors were weak in zero-shot, but it also perturbs previously strong zero-shot alignments for large vehicle classes.

### IDD Specific Failure Pattern

After tuning, `bus` and `truck` have much higher recall-like activity but poorer precision/alignment:

- Tuned `bus`: 7900 GT objects. Predictions/confusion: `unknown:2777`, `missed:2737`, `bus:1579`, `car:506`, `autorickshaw:167`.
- Tuned `truck`: 15378 GT objects. Predictions/confusion: `missed:6294`, `unknown:4020`, `car:1732`, `truck:1037`, `autorickshaw:929`.

The tuned model often does not commit to the correct novel label for these classes. It either misses them, maps them to `unknown`, or maps them to nearby base traffic classes such as `car` and `autorickshaw`.

This points to a misalignment between:

- the text/semantic embedding for the novel class,
- the visual anchor selected by the detector,
- and the few-shot examples available for that class.

For `bus` and `truck`, zero-shot already had a strong enough semantic/visual prior. Few-shot tuning appears to disturb that prior while trying to optimize the task-level objective.

### IDD Unknowns and A-OSE

Unknown recall improves strongly from 8.76 to 39.64, and A-OSE drops from 3139 to 1709. This means the tuned model is better at producing unknown detections overall.

But the remaining A-OSE is concentrated in semantically nearby current/vehicle classes:

- `tanker_vehicle`: 525 unknowns misclassified as this class, 30.7 percent of A-OSE.
- `excavator`: 327, 19.1 percent.
- `autorickshaw`: 308, 18.0 percent.
- `truck`: 144, 8.4 percent.

The unknown confusion is not random. Unknown construction/vehicle-like objects are pulled into newly learned or visually similar vehicle classes.

Example tuned IDD unknown confusion:

- `concrete_mixer`: often predicted as `excavator` or `tanker_vehicle`.
- `pull_cart`: often predicted as `street_cart`.
- `road_roller`: often predicted as `tanker_vehicle` or `excavator`.
- `tractor`: often predicted as `autorickshaw`, `tanker_vehicle`, or `excavator`.

This is a key open-world failure: newly added current classes become attractive false-positive sinks for true unknown objects.

## nuScenes/nuOWODB: Zero-Shot vs Few-Shot Tuned

nuOWODB T2 has 10 previous/base classes and 7 current/novel pedestrian-subtype classes.

Base classes:

- `vehicle.bicycle`, `vehicle.motorcycle`, `vehicle.car`, `vehicle.bus.bendy`, `vehicle.bus.rigid`, `vehicle.truck`, `vehicle.emergency.ambulance`, `vehicle.emergency.police`, `vehicle.construction`, `vehicle.trailer`

Current/novel classes:

- `human.pedestrian.adult`, `human.pedestrian.child`, `human.pedestrian.wheelchair`, `human.pedestrian.stroller`, `human.pedestrian.personal_mobility`, `human.pedestrian.police_officer`, `human.pedestrian.construction_worker`

Unknown classes:

- `movable_object.trafficcone`, `movable_object.barrier`, `movable_object.debris`, `movable_object.pushable_pullable`, `static_object.bicycle_rack`, `animal`, `vehicle.ego`

### Aggregate Metrics

| Metric | Zero-shot | Few-shot tuned | Change |
| --- | ---: | ---: | ---: |
| Prev/base AP50 | 19.97 | 27.34 | +7.37 |
| Current/novel AP50 | 6.43 | 6.47 | +0.04 |
| Known AP50 | 14.40 | 18.74 | +4.35 |
| Unknown AP50 | 2.19 | 0.715 | -1.48 |
| Unknown Recall @50 | 27.37 | 31.86 | +4.49 |
| A-OSE | 1263 | 810 | -453 |

High-level observation: the tuned nuScenes T2 model improves base performance and lowers A-OSE, but it barely learns the new/current classes. Current AP50 changes from 6.43 to only 6.47, effectively flat.

This is the clearest cold-start problem in these experiments: the T2 few-shot stage does not meaningfully convert the novel pedestrian-subtype classes into reliable known detections.

### nuScenes Novel-Class AP50

| Novel class | Zero-shot AP50 | Few-shot tuned AP50 | Observation |
| --- | ---: | ---: | --- |
| `human.pedestrian.adult` | 2.223 | 2.226 | unchanged |
| `human.pedestrian.child` | 6.229 | 6.310 | unchanged/slight |
| `human.pedestrian.wheelchair` | 0.000 | 0.000 | no learning |
| `human.pedestrian.stroller` | 25.034 | 25.234 | unchanged/slight |
| `human.pedestrian.personal_mobility` | 0.030 | 0.031 | unchanged |
| `human.pedestrian.police_officer` | 0.896 | 0.919 | unchanged |
| `human.pedestrian.construction_worker` | 10.589 | 10.535 | unchanged/slight down |

The tuned model is almost identical to the zero-shot model on the current/novel pedestrian subclasses.

### nuScenes Cold-Start Pattern

The few-shot tuned model improves previous/base vehicle classes:

- `vehicle.bicycle`: 42.573 -> 48.710.
- `vehicle.motorcycle`: 46.469 -> 54.868.
- `vehicle.car`: 41.068 -> 61.518.
- `vehicle.truck`: 32.230 -> 42.026.

But it does not learn the T2 pedestrian subtypes. The confusion analysis shows why:

- `human.pedestrian.adult`: N=28529, `missed:23861`, `unknown:1874`, only `human.pedestrian.adult:335`.
- `human.pedestrian.construction_worker`: N=3089, `missed:2397`, only `human.pedestrian.construction_worker:450`.
- `human.pedestrian.personal_mobility`: N=453, `missed:398`, only `human.pedestrian.personal_mobility:8`.
- `human.pedestrian.child`: N=251, `missed:207`, only `human.pedestrian.child:28`.

Most novel objects are not being turned into correct known-class predictions. They are missed or absorbed by unknown/nearby pedestrian classes.

This likely reflects a cold-start mismatch:

- The novel classes are fine-grained pedestrian subtypes.
- The detector/objectness branch does not get enough useful few-shot gradient to separate them.
- The semantic/text anchors are too close or ambiguous.
- The visual examples are sparse relative to the diversity of pedestrians in the test set.

### nuScenes Unknown Behavior

Tuned unknown recall improves slightly, 27.37 -> 31.86, and A-OSE drops, 1263 -> 810. However unknown AP falls, 2.19 -> 0.715, and the tuned model produces far more unknown predictions:

- Zero-shot unknown predictions: 233,066.
- Tuned unknown predictions: 1,038,913.

So the tuned model is more willing to fire the unknown head, which improves recall but hurts precision/AP.

Unknown class recall after tuning is uneven:

- `movable_object.trafficcone`: 39.53 percent.
- `movable_object.barrier`: 22.95 percent.
- `movable_object.debris`: 37.89 percent.
- `movable_object.pushable_pullable`: 46.32 percent.
- `static_object.bicycle_rack`: 47.92 percent.
- `animal`: 16.25 percent.

The analysis run reports similar overall U-Recall, 12428/38808 = 32.02 percent, with most unknown GT still missed rather than cleanly detected as unknown.

## Model-Behavior Diagnostics

### Known Scores Are Much Stronger For Base Classes Than For Novel Classes

In nuScenes, the role-correct score table shows a sharp base-vs-novel gap:

- Strong base examples:
  - `vehicle.car`: correct score mean 0.2198, median rank 1.
  - `vehicle.motorcycle`: 0.1337, median rank 1.
  - `vehicle.bicycle`: 0.1252, median rank 1.
  - `vehicle.truck`: 0.1018, median rank 2.
- Weak novel examples:
  - `human.pedestrian.adult`: 0.0053, median rank 6.
  - `human.pedestrian.personal_mobility`: 0.0052, median rank 8.
  - `human.pedestrian.wheelchair`: 0.0001, median rank 12.
  - `human.pedestrian.construction_worker`: 0.0345, median rank 4.

The exception is `human.pedestrian.stroller` with score mean 0.1823, but it has very small N=53 in the role-correct table, so it should not be treated as proof that the whole novel branch is healthy.

The main point: base classes often have the correct class near rank 1, while many novel classes have their correct class buried around rank 4-8. This matches the low current AP.

### IDD Anchor/Max-Known Scores Show Novel Confusion

In IDD, base classes generally have much stronger max-known scores than novel/unknown classes:

- Base `car`: maxK mean 0.252, Tunk mean 0.004.
- Base `rider`: maxK mean 0.233, Tunk mean 0.002.
- Base `person`: maxK mean 0.211, Tunk mean 0.004.
- Base `motorcycle`: maxK mean 0.195, Tunk mean 0.001.

Novel classes have weaker and less stable alignment:

- `bus`: anchor mean 0.050, maxK mean 0.047, Tunk mean 0.028, ratio mean 1.910.
- `truck`: anchor mean 0.056, maxK mean 0.048, Tunk mean 0.025, ratio mean 2.239.
- `tanker_vehicle`: anchor mean 0.074, maxK mean 0.031, Tunk mean 0.033, ratio mean 0.264.
- `crane_truck`: anchor mean 0.017, maxK mean 0.016, Tunk mean 0.007.
- `street_cart`: anchor mean 0.018, maxK mean 0.040, Tunk mean 0.004.

This explains the class-level tradeoff. The detector has enough signal to find many novel objects, but the class evidence is weak and unstable. For some classes this helps compared to zero-shot, but for already-strong classes like `bus` and `truck`, tuning can push detections into `unknown` or nearby known classes.

### Unknown/Objectness Scores Are Too Low For Many Unknown Objects

Both datasets show very low Tobj/Tunk values for unknown GT objects, especially at stride 16 and stride 32, with many medians equal to zero.

nuScenes aggregate unknown score distribution:

- stride 8: Tobj mean 0.0112, Tunk mean 0.0099, Tobj median 0.0020, Tunk median 0.0002.
- stride 16: Tobj mean 0.0071, Tunk mean 0.0060, both medians 0.0000.
- stride 32: Tobj mean 0.0079, Tunk mean 0.0063, both medians 0.0000.

IDD unknown examples are similar:

- `pole`: Tobj/Tunk medians near zero across strides.
- `concrete_mixer`: high confusion with known construction-like classes, but weak unknown/objectness medians.
- `tractor`: Tobj/Tunk medians near zero for most strides.

Interpretation: the model often does not produce a strong objectness/unknown response for true unknown objects. The unknown detector is not just suffering from classification confusion; many unknowns have low objectness-like activation before classification even matters.

This may be caused by the training signal:

- Few-shot T2 images provide gradients for a small set of current classes, not for the true unknown distribution.
- Unknown objects are not directly supervised as a coherent set.
- Anchor/objectness selection remains biased toward classes that were already well represented in previous training.
- Sparse few-shot images do not provide enough coverage to teach robust unknownness at different scales/strides.

### Anchors and Max-Known Scores Get Confused For Novel and Unknown Classes

The diagnostics suggest that anchor selection and max-known scoring are entangled:

- For base classes, max-known scores are relatively high and the correct class rank is often near the top.
- For novel classes, correct scores are low and ranks are poor, even when the object is matched by IoU.
- For unknown classes, the model sometimes assigns high semantic proximity to nearby known/current classes instead of `unknown`.

Concrete examples:

- IDD `pull_cart` frequently becomes `street_cart`.
- IDD `concrete_mixer` frequently becomes `excavator` or `tanker_vehicle`.
- IDD `tractor` frequently becomes `autorickshaw`, `tanker_vehicle`, or `excavator`.
- nuScenes `static_object.bicycle_rack` often becomes `vehicle.bicycle`.
- nuScenes pedestrian subclasses often collapse into missed/unknown/neighbor pedestrian categories.

So the system has two related problems:

1. Open-world unknown separation is weak.
2. Fine-grained current-class alignment is weak, especially when classes are semantically close or visually overlapping.

## Working Hypotheses

1. Few-shot tuning is improving the detector as a general known/unknown detector, but not consistently improving class-specific alignment.

2. Some classes benefit from few-shot tuning because zero-shot was weak and any visual adaptation helps, for example IDD `crane_truck` and `street_cart`.

3. Some classes regress because zero-shot already had a good text-visual prior, and the few-shot update disturbs that prior, for example IDD `bus` and `truck`.

4. nuScenes T2 has a stronger cold-start problem than IDD because the novel classes are fine-grained pedestrian subtypes. The model needs to learn subtle distinctions from very sparse data, and the current tuning does not provide enough discriminative signal.

5. Unknown AP can get worse even when unknown recall improves. This happens when the model fires unknown too broadly, as in tuned nuScenes, where unknown predictions increase from 233k to over 1.0M.

6. A-OSE after tuning is concentrated in semantically nearby classes, not uniformly distributed. Newly learned current classes can become false-positive attractors for true unknown objects.

7. Low Tobj/Tunk medians for unknown GT suggest the problem is partly upstream of classification: the model often does not produce a strong objectness/unknown response for unknown objects.

## Problem Statement For Future Work

The current YOLO-UniOW T2 few-shot tuning improves aggregate metrics but does not reliably solve open-world class expansion. It has two failure modes:

1. On IDD, tuning improves overall performance but damages already-strong zero-shot novel classes such as `bus` and `truck`, while improving weaker specialized classes. This indicates unstable class alignment after few-shot adaptation.

2. On nuScenes, tuning improves base classes and slightly improves unknown recall, but the current novel pedestrian subclasses remain nearly unchanged from zero-shot. This indicates a cold-start failure: the T2 few-shot stage is not learning discriminative novel-class boundaries.

Across both datasets, unknown detection remains fragile. Unknown objects often have very low objectness/unknown scores, and when they are detected they are frequently absorbed into semantically nearby known/current classes. The next method should explicitly address both problems: preserve useful zero-shot alignment while adapting to few-shot classes, and provide stronger unknown/objectness supervision or calibration so true unknowns do not collapse into nearby known labels.
