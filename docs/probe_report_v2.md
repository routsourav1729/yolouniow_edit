# Text-Visual Alignment Probe — Report v2

**Date:** 2026-05-06  
**Jobs:** probe_114637 (IDD), probe_114638 (nuImages)  
**Probe version:** IDD=v2, nuImages=v1  
**Code:** `tools/probe_text_visual.py`  
**Outputs:** `results/probe/v2/` (IDD), `results/probe/v1/` (nuImages)

---

## 1. Setup Summary

| | IDD | nuImages |
|---|---|---|
| Images | 21,084 | 14,747 |
| Test classes | 14 | 23 |
| Unique prompts | 40 | 69 |
| Checkpoint | `best_owod_Both_epoch_40.pth` (T2) | `best_owod_Both_epoch_10.pth` (T2) |
| Model config | `yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2` | `yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_nuowodb_t2` |

Scores are `sigmoid(BN(Fvis) · E_prompt)` where `E_prompt` is encoded by the finetuned CLIP text encoder (LoRA + pretrain weights loaded from `pretrained/yolo_uniow_l_lora_bn_5e-4_...pth`). Absolute score values are low (~0.01–0.12) because the visual backbone was aligned to finetuned `self.embeddings`, not raw CLIP. Relative metrics (winner rate, specificity, net_useful) are the meaningful numbers.

---

## 2. Cross-Class Winner Rates

Winner rate = fraction of GT-matched anchors where the correct class's summed sigmoid score is highest across all test classes.

### IDD
| Class | N anchors | Winner rate | Margin |
|---|---|---|---|
| traffic light | 5,677 | **0.877** | +0.106 |
| motorcycle | 391,209 | **0.784** | +0.093 |
| traffic sign | 47,539 | **0.767** | +0.038 |
| truck | 175,583 | 0.703 | +0.035 |
| street cart | 23,512 | 0.691 | +0.039 |
| person | 132,819 | 0.593 | +0.109 |
| bus | 133,670 | 0.584 | +0.047 |
| crane truck | 3,569 | 0.574 | −0.012 |
| bicycle | 7,950 | 0.539 | +0.025 |
| car | 470,195 | 0.380 | +0.030 |
| rider | 225,778 | 0.326 | −0.041 |
| tanker | 9,256 | 0.249 | −0.021 |
| autorickshaw | 165,451 | 0.135 | −0.012 |
| excavator | 4,254 | **0.092** | +0.002 |

Classes with `n/a`: animal, concrete_mixer, pole, pull_cart, road_roller, tractor — GT classes present in dataset but no matching test_class entry in config (correct: these are intentionally left as unknown classes).

### nuImages
| Class | N anchors | Winner rate | Margin |
|---|---|---|---|
| movable_object.trafficcone | 44,944 | **0.787** | +0.051 |
| vehicle.emergency.ambulance | 58 | 0.552 | +0.017 |
| human.pedestrian.stroller | 264 | 0.489 | +0.060 |
| movable_object.barrier | 73,297 | 0.494 | +0.003 |
| human.pedestrian.adult | 70,123 | 0.401 | +0.014 |
| static_object.bicycle_rack | 5,932 | 0.428 | −0.014 |
| human.pedestrian.child | 549 | 0.388 | +0.019 |
| vehicle.car | 275,770 | 0.318 | +0.007 |
| vehicle.emergency.police | 712 | 0.306 | −0.002 |
| human.pedestrian.construction_worker | 8,001 | 0.303 | +0.017 |
| vehicle.bus.rigid | 17,381 | 0.306 | +0.006 |
| vehicle.motorcycle | 26,271 | 0.296 | +0.020 |
| vehicle.truck | 64,305 | 0.266 | −0.000 |
| vehicle.bicycle | 21,977 | 0.258 | +0.056 |
| animal | 134 | 0.246 | −0.013 |
| vehicle.construction | 10,708 | 0.104 | −0.002 |
| human.pedestrian.police_officer | 406 | 0.121 | −0.044 |
| movable_object.pushable_pullable | 3,317 | 0.198 | −0.002 |
| vehicle.bus.bendy | 450 | 0.073 | −0.014 |
| movable_object.debris | 3,075 | 0.052 | −0.017 |
| human.pedestrian.personal_mobility | 931 | **0.040** | −0.032 |

**Pattern:** Novel classes (T2/T3) consistently underperform base classes. IDD novel classes (excavator 9.2%, autorickshaw 13.5%) and nuImages novel classes (personal_mobility 4%, bus_bendy 7.3%, debris 5.2%) are the weakest. The negative margins on several novel classes indicate they are being beaten by a competing class's prompts.

---

## 3. Within-Class Winner Analysis

Which sub-prompt wins argmax most often within GT-matched anchors for each class.

### IDD — notable findings

| Class | Best prompt | Win% | Notes |
|---|---|---|---|
| motorcycle | "two wheeler" | 54.6% | Beats "motorcycle" (35%) — but "two wheeler" is a massive stealer (see §5) |
| person | "person" | 79.7% | Dominant; "pedestrian" 17.8%, "human" 2.6% |
| autorickshaw | "three wheeler" | 69.7% | Stealer — "three wheeler" bleeds to motorcycle, car |
| bus | "passenger bus" | 83.1% | Best bus prompt |
| truck | "cargo truck" | 85.6% | Dominant — but also the worst stealer overall |
| excavator | "digging machine" | 67.2% | Despite low class winner rate (9.2%) — anchor-level win doesn't help when cross-class contest is lost |
| rider | "motorcyclist" (42.5%) / "driver" (40.7%) | split | "driver" is a heavy stealer (net=−24K) |
| street cart | "street vendor cart" (57.7%) / "hand cart" (36%) | split | Both are net-negative; "food cart" least harmful |

### nuImages — notable findings

| Class | Best prompt | Win% | Notes |
|---|---|---|---|
| car | "car" | 58.6% | Clean winner |
| motorcycle | "two wheeler" | 58.7% | Same stealer issue as IDD |
| truck | "cargo truck" | 84.0% | Dominant within-class, heavy off-class stealer |
| bus_rigid | "passenger bus" | 73.5% | Good |
| ambulance | "medical van" | 60.3% | But net=−29K (steals from car/truck/bus) |
| stroller | "pushchair" | 64.4% | But net=−2896 |
| personal_mobility | "personal mobility device" | 64.6% | Near-pure stealer (net=−2758, 99.7% steal) |
| animal | "dog" | 83.6% | Specific; "dog" alone has spec=986 but net=−116 |
| police_car | "patrol car" | 58.3% | Net=−29K — catastrophic stealer from car |

**Key insight:** A prompt winning within-class does not mean it is safe — it may win on its GT class *and* dominate other GT classes too. Net_useful captures this; within-class winner rate alone is misleading for prompts with large absolute score magnitude (like "cargo truck", "two wheeler", "person").

---

## 4. Mean Score Matrix Highlights

The full matrix is in `score_matrix.csv`. Key confusions from the printed output:

### IDD confusions
- **excavator** GT: highest mean is `excavator` class (0.038) but `crane_truck` (0.018), `truck` (0.024), `tanker` (0.020) all score comparably → excavator loses cross-class contest
- **autorickshaw** GT: `truck` class (0.040) ≈ `autorickshaw` class (0.042) → very tight margin, explains 13.5% winner rate
- **rider** GT: `person` class (0.103) nearly equals `rider` class (0.070) → person class outscores rider on rider anchors
- **tanker** GT: `truck` (0.050) > `tanker` (0.035) → truck class dominates tanker anchors

### nuImages confusions
- **vehicle.construction** GT: `construction_worker` class (0.063) > `construction_vehicle` class (0.028) — pedestrian-like word activates on vehicle anchors
- **vehicle.trailer** GT: `truck` (0.029) ≈ `trailer` (0.030) — near tie
- **human.pedestrian.personal_mobility** GT: scattered across bicycle, motorcycle, wheelchair, stroller classes — no dominant anchor
- **vehicle.emergency.police** GT: `car` class (0.029) and `police_car` class (0.034) nearly equal; police car prompts (patrol car, police car) steal massively from car anchors

---

## 5. Prompt Leakage Diagnostics

`net_useful = on_target_wins − off_target_wins`. Positive = net useful to OWOD embedding initialization; negative = net harmful.

### IDD — Top 10 most useful prompts
| Prompt | Parent class | Specificity | Within% | On-wins | Steal% | Net |
|---|---|---|---|---|---|---|
| car | car | 18.1 | — | 194,140 | 13.6% | **+163,626** |
| motorcycle | motorcycle | 31.4 | 35.0% | 95,731 | 12.0% | **+82,684** |
| passenger bus | bus | 16.0 | 83.1% | 56,297 | 24.5% | **+38,030** |
| motorbike | motorcycle | 36.3 | 10.4% | 21,139 | 16.4% | **+16,978** |
| traffic sign | traffic_sign | 304.3 | 36.8% | 14,877 | 13.4% | **+12,577** |
| road traffic sign | traffic_sign | 237.4 | 40.6% | 13,454 | 15.3% | **+11,026** |
| city bus | bus | 18.2 | 8.0% | 7,424 | 24.7% | **+4,995** |
| motorcyclist | rider | 20.4 | 42.5% | 8,695 | 40.6% | +2,745 |
| pedestrian | person | 19.9 | 17.8% | 4,316 | 36.1% | +1,874 |
| tuk tuk | autorickshaw | 17.5 | 1.4% | 702 | 28.8% | +418 |

### IDD — Top 10 worst stealers (drop candidates)
| Prompt | Parent class | Steal% | Net |
|---|---|---|---|
| cargo truck | truck | 67.7% | **−128,495** |
| person | person | 62.1% | **−47,746** |
| three wheeler | autorickshaw | 77.1% | **−42,655** |
| hand cart | street_cart | 86.6% | **−31,975** |
| driver | rider | 60.5% | **−24,866** |
| water tanker | tanker | 96.0% | **−18,942** |
| street vendor cart | street_cart | 71.0% | **−11,809** |
| pedal bicycle | bicycle | 83.9% | **−9,082** |
| two wheeler | motorcycle | 51.3% | **−7,985** |
| digging machine | excavator | 98.0% | **−6,861** |

### nuImages — Top 10 most useful prompts
| Prompt | Parent class | Specificity | Within% | On-wins | Steal% | Net |
|---|---|---|---|---|---|---|
| car | car | 29.2 | 58.6% | 64,131 | 7.7% | **+58,760** |
| passenger car | car | 7.5 | 36.4% | 24,724 | 13.6% | **+20,848** |
| concrete barrier | barrier | 55.1 | 30.0% | 16,012 | 7.7% | **+14,674** |
| traffic cone | traffic_cone | 665.7 | 34.3% | 12,724 | 5.9% | **+11,927** |
| road cone | traffic_cone | 444.4 | 57.6% | 17,857 | 28.8% | **+10,630** |
| adult pedestrian | adult_pedestrian | 65.1 | 32.5% | 8,387 | 11.0% | **+7,350** |
| walking person | adult_pedestrian | 49.5 | 37.7% | 8,315 | 11.0% | **+7,283** |
| sedan | car | 41.7 | 5.0% | 5,281 | 3.6% | **+5,083** |
| orange cone | traffic_cone | 521.1 | 8.2% | 2,766 | 12.8% | +2,359 |
| motorcycle | motorcycle | 77.4 | 21.6% | 2,269 | 23.9% | +1,556 |

### nuImages — Top 10 worst stealers (drop candidates)
| Prompt | Parent class | Steal% | Net |
|---|---|---|---|
| medical van | ambulance | 100.0% | **−29,662** |
| patrol car | police_car | 99.5% | **−29,110** |
| emergency ambulance | ambulance | 99.9% | **−25,289** |
| bike stand | bicycle_rack | 98.1% | **−25,154** |
| police car | police_car | 99.9% | **−20,795** |
| trailer | trailer | 94.5% | **−23,717** |
| wheelchair | wheelchair_user | 100.0% | **−12,787** |
| ambulance | ambulance | 100.0% | **−13,165** |
| young pedestrian | child_pedestrian | 99.3% | **−15,564** |
| bicycle rack | bicycle_rack | 94.6% | **−16,038** |

---

## 6. Top-Stealing Cross-Class Patterns

### IDD — who steals from whom

| Victim GT class | #1 stealer | #2 stealer | #3 stealer |
|---|---|---|---|
| car (470K) | cargo truck (96K) | two wheeler (83K) | driver (40K) |
| rider (225K) | **person (89K!)** | two wheeler (42K) | motorcycle (9K) |
| motorcycle (391K) | three wheeler (45K) | person (19K) | hand cart (10K) |
| autorickshaw (165K) | **cargo truck (75K!)** | two wheeler (15K) | car (10K) |
| bus (133K) | **cargo truck (40K!)** | two wheeler (4K) | driver (3K) |
| truck (175K) | passenger bus (5K) | two wheeler (4K) | car (4K) |
| pole (9K) | road signal lamp (4K) | lifting crane (1K) | traffic sign (1K) |

### nuImages — who steals from whom

| Victim GT class | #1 stealer | #2 stealer | #3 stealer |
|---|---|---|---|
| vehicle.car (275K) | **patrol car (28K)** | medical van (23K) | police car (20K) |
| vehicle.truck (64K) | emergency ambulance (8K) | trailer (6K) | medical van (4K) |
| movable_object.barrier (73K) | trailer (13K) | cargo truck (4K) | bicycle rack (4K) |
| human.pedestrian.adult (70K) | young pedestrian (12K) | person in wheelchair (4K) | bike stand (4K) |
| vehicle.bicycle (21K) | **bike stand (6K)** | bicycle rack (1K) | cycle parking rack (1K) |
| movable_object.trafficcone (44K) | traffic barrier (3K) | bike stand (2K) | pushable object (2K) |
| vehicle.motorcycle (26K) | wheelchair (2K) | electric scooter (2K) | bike stand (1K) |
| static_object.bicycle_rack (5K) | two wheeler (1K) | road barrier (690) | bicycle (353) |

---

## 7. Cross-Dataset Patterns

### Consistent patterns across IDD and nuImages

1. **"cargo truck" / "cargo trailer" are universal vehicle stealers.** In IDD, "cargo truck" steals 128K anchors net negative, robbing car, autorickshaw, bus, tractor. In nuImages, "cargo truck" steals from construction, trailer, barrier. The CLIP text encoder strongly activates generic heavy-vehicle language broadly — drop from truck classes in favor of more specific terms.

2. **"two wheeler" is a cross-class toxin.** IDD: net=−7,985, steals from motorcycle/bicycle/autorickshaw. nuImages: net=−9,052. Both datasets show it winning within-class (54% IDD, 58% nuImages) while harming other classes. Despite being a high within-class winner, it destroys cross-class discrimination.

3. **Passenger / city bus are clean while generic "bus" is ambiguous.** "passenger bus" is the top within-class winner in both datasets (83% IDD, 73% nuImages) and has positive net_useful. "bus" alone has much lower specificity and is borderline.

4. **Highly descriptive multi-word phrases backfire for rare/novel classes.** nuImages: "medical van" (ambulance, 100% steal), "emergency ambulance" (100% steal), "patrol car" (99.5% steal), "personal mobility device" (99.7% steal), "young pedestrian" (99.3% steal). IDD: "water tanker" (96% steal), "digging machine" (98% steal), "construction excavator" (97.9% steal). These activate on visually dominant classes (car, truck, person) rather than the rare target.

5. **Single-word canonical terms for dominant classes are the safest prompts.** "car" (IDD net=+163K, nuImages net=+59K), "motorcycle" (IDD net=+82K, nuImages net=+1.5K), "sedan" (nuImages net=+5K, only 3.6% steal). Simple, unambiguous nouns matching the class's visual prototype work best.

6. **Traffic infrastructure prompts are extremely clean.** "traffic cone" (nuImages spec=665, steal=5.9%), "traffic sign" (IDD spec=304, steal=13.4%), "road traffic sign" (IDD spec=237) all have very high specificity and positive net. These classes have visually distinctive features that CLIP encodes well.

7. **Pedestrian sub-type confusion is systematic.** In both datasets, pedestrian-related classes steal from each other. IDD: "person" (net=−47K) steals rider anchors heavily (89K stolen). nuImages: "adult" (net=−832), "young pedestrian" (−15K), "walking person" steals from construction_worker. Multi-word descriptive pedestrian phrases tend to activate on the larger adult_pedestrian class.

8. **Novel class prompts need anchor count consideration.** Low-frequency novel classes (IDD excavator=4K, nuImages bus_bendy=450, personal_mobility=931) have almost no on-target anchors to win, so even small off-target activation produces large negative net. For rare classes, only use prompts with high specificity (spec > 20).

---

## 8. Recommended Prompt Actions

### IDD — per class

| Class | Action | Drop | Keep / Replace with |
|---|---|---|---|
| car | Minor fix | — | "car" (clean) |
| motorcycle | Drop generic | "two wheeler" | "motorcycle", "motorbike" |
| rider | Drop generic | "driver" | "motorcyclist", "rider" |
| person | Drop generic | "person" | "pedestrian", "human" |
| autorickshaw | Drop generic | "three wheeler" | "auto rickshaw", "tuk tuk" + add "autorickshaw" |
| bicycle | Drop | "pedal bicycle" | "bicycle" only |
| bus | Keep specific | — | "passenger bus", "city bus" (drop "bus" if margin allows) |
| truck | Fix anchor | "cargo truck", "lorry" | "heavy truck", "delivery truck" |
| tanker | Drop | "water tanker", "fuel tanker" | "tanker truck" only |
| crane truck | Drop | "lifting crane", "mobile crane" | "crane truck" only |
| street cart | Drop | "hand cart", "street vendor cart" | "food cart" only (least harmful) |
| excavator | Drop all | "digging machine", "construction excavator", "yellow excavator" | "excavator" + try "yellow digger", "backhoe" |
| traffic sign | Keep | — | All 3 prompts OK |
| traffic light | Keep 1 | "traffic signal light", "road signal lamp" | "traffic light" only |

### nuImages — per class

| Class | Action | Drop | Keep / Replace with |
|---|---|---|---|
| car | Keep | — | "car", "sedan", "passenger car" (all good) |
| motorcycle | Drop generic | "two wheeler" | "motorcycle", "motorbike" |
| bus_rigid | Keep specific | "bus" (borderline) | "passenger bus", "city bus" |
| bus_bendy | Full rethink | "accordion bus", "bendy bus" | Try "articulated bus", "double-jointed bus" |
| truck | Fix | "cargo truck", "lorry" | "truck" only; add "box truck" |
| ambulance | Full rethink | all 3 current prompts (all net-negative) | "ambulance vehicle", "rescue ambulance" |
| police_car | Full rethink | all 3 (patrol car=−29K) | "police SUV", "marked patrol vehicle" |
| construction_vehicle | Fix | "construction vehicle", "bulldozer" | "excavator" alone (top within-class, positive net) |
| trailer | Drop generic | "trailer", "cargo trailer" | "semi trailer" (least bad) |
| adult_pedestrian | Keep | — | "adult pedestrian", "walking person" |
| child_pedestrian | Fix | "young pedestrian" (−15K), "kid walking" (−2K) | "child pedestrian", "small child walking" |
| wheelchair_user | Fix | "wheelchair" (−12K), "person in wheelchair" (−7K) | "wheelchair user" only |
| stroller | Fix | "pram" (−1K), "pushchair" (−2K) | "baby stroller" (least bad despite small net) |
| personal_mobility | Full rethink | all 3 | Try "kick scooter", "electric skateboard" |
| police_officer | Keep cautious | "law enforcement officer" (−1K) | "police officer", "uniformed officer" |
| construction_worker | Fix | "hard hat worker" (−2K), "site worker" (−4K) | "construction worker" only |
| barrier | Keep 1–2 | "traffic barrier" (−1K) | "road barrier", "concrete barrier" |
| traffic_cone | Keep all | — | All 3 clean; "traffic cone" and "road cone" best |
| pushable_object | Fix | "trolley" (−1K), "pushable object" (−6K) | "wheeled cart" only |
| debris | Fix | all 3 (all net-negative) | Try "road debris", "litter on road" — raw CLIP may not distinguish |
| bicycle_rack | Full rethink | all 3 (all net-negative) | Try "metal bike rack", "outdoor cycle stand" |
| animal | Fix | all 3 (all net-negative for rare class) | Accept low performance — too rare (134 anchors) |

---

## 9. Summary of Key Numbers

| Metric | IDD | nuImages |
|---|---|---|
| Prompts with positive net_useful | 16 / 40 (40%) | 20 / 69 (29%) |
| Prompts with steal% > 80% | 9 | 29 |
| Best single prompt | "car" (+163K) | "car" (+59K) |
| Worst single prompt | "cargo truck" (−128K) | "medical van" (−30K) |
| Highest cross-class winner rate | traffic light 87.7% | traffic cone 78.7% |
| Lowest cross-class winner rate | excavator 9.2% | personal mobility 4.0% |
| Mean absolute score range | 0.001–0.122 | 0.001–0.173 |

---

## 10. Next Steps

1. **Immediate:** Re-run probe after dropping worst-stealing prompts (esp. "cargo truck", "two wheeler", "person", "three wheeler" in IDD; all ambulance/police-car variants plus "bike stand" in nuImages). Expect noticeable jump in winner rates for currently-confused classes.

2. **Embedding initialization:** The net_useful-positive prompts per class are the safest candidates for embedding initialization for novel classes in OWOD T2/T3. Use only prompts where `net > 0` and `steal% < 30%`.

3. **Rare classes:** For classes with < 1K anchors and all net-negative prompts (personal_mobility, bus_bendy, debris, bicycle_rack, animal), the issue is insufficient GT representation for the probe + high activation from dominant nearby classes. Consider using pretrained `self.embeddings` directly for initialization rather than prompt-derived embeddings.

4. **Score gap:** Visual backbone trained against finetuned embeddings; fresh CLIP-encoded prompts produce lower absolute scores but preserved relative order. If absolute calibration matters, consider calibrating probe with known-good classes (car, motorcycle) as anchors.
