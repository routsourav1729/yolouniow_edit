_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2.py']

# nuOWODB T2 with KUME k-vs-k ONLY (no WAPR, no k-vs-unk).
# Mirror of the IDD kume_kvk config. nuOWODB: 10 base + 7 novel = 17 known.
import os

_margin = float(os.environ.get('KUME_KVK_MARGIN', '0.5'))

model = dict(
    kume=dict(
        num_known_classes=17,
        unk_idx=17,
        margin=_margin,
        weight=0.5,
    ),
    # NOTE: wapr is intentionally absent. KUME runs alone.
)
