_base_ = ['yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py']

# Override embedding paths to use the IDD-specific embed directory.
# Known class embeddings (idd_t1.npy / idd_t2.npy) and the
# object/object_tuned wildcards are all stored under embeddings/uniow-idd/.
model = dict(
    embedding_path=f'embeddings/uniow-idd/{_base_.owod_dataset.lower()}_t{_base_.owod_task}.npy',
    unknown_embedding_path='embeddings/uniow-idd/object.npy',
    anchor_embedding_path='embeddings/uniow-idd/object_tuned.npy',
)
