# partly taken from  https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools

import os
import copy
import random
from collections import defaultdict
from mmdet.utils import ConfigType
from mmdet.datasets import BaseDetDataset
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS

import xml.etree.ElementTree as ET
from mmengine.logging import MMLogger

from .owodb_const import *

@DATASETS.register_module()
class OWODDataset(BatchShapePolicyDataset, BaseDetDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    METAINFO = {
        'classes': (),
        'palette': None,
    }

    def __init__(self,
                 data_root: str,
                 dataset: str = 'MOWODB',
                 image_set: str='train',
                 owod_cfg: ConfigType = None,
                 training_strategy: int = 0,
                 fewshot_dir: str = '',
                 fewshot_k: int = 0,
                 fewshot_seed: int = 1,
                 **kwargs):

        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set_fns = []

        self.image_set = image_set
        self.dataset=dataset
        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset]
        self.task_num = owod_cfg.task_num
        self.owod_cfg = owod_cfg

        # Few-shot per-class filtering (CED-FOOD style)
        # When set, only annotations matching the per-class k-shot file
        # are retained for each image.
        self._fewshot_dir = fewshot_dir
        self._fewshot_k = fewshot_k
        self._fewshot_seed = fewshot_seed
        self._fewshot_allowed = {}  # img_id -> set of allowed class names
        
        self._logger = MMLogger.get_current_instance()

        # training strategy
        self.training_strategy = training_strategy
        if "test" not in image_set:
            if training_strategy == 0:
                self._logger.info(f"Training strategy: OWOD")
            elif training_strategy == 1:
                self._logger.info(f"Training strategy: ORACLE")
            else:
                raise ValueError(f"Invalid training strategy: {training_strategy}")

        OWODDataset.METAINFO['classes'] = self.CLASS_NAMES
        
        self.data_root=str(data_root)
        annotation_dir = os.path.join(self.data_root, 'Annotations', dataset)
        image_dir = os.path.join(self.data_root, 'JPEGImages', dataset)

        file_names = self.extract_fns()
        self.image_set_fns.extend(file_names)
        self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
        self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])
        self.imgids.extend(x for x in file_names)            
        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        assert (len(self.images) == len(self.annotations) == len(self.imgids))

        super().__init__(**kwargs)

    def extract_fns(self):
        # If few-shot dir is specified, use per-class file loading
        if self._fewshot_dir and self._fewshot_k > 0:
            return self._extract_fns_fewshot()

        splits_dir = os.path.join(self.data_root, 'ImageSets')
        splits_dir = os.path.join(splits_dir, self.dataset)
        image_sets = []
        file_names = []

        if 'test' in self.image_set: # for test
            image_sets.append(self.image_set)
        else: # owod or oracle
            image_sets.append(f"t{self.task_num}_{self.image_set}")

        self.image_set_list = image_sets
        for image_set in image_sets:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
            with open(os.path.join(split_f), "r") as f:
                file_names.extend([x.strip() for x in f.readlines()])
        return file_names

    def _extract_fns_fewshot(self):
        """Read per-class k-shot files (CED-FOOD / TFA style).

        For each novel class, reads ``box_{k}shot_{cls}_train.txt`` from the
        few-shot split directory. Each line is an image path. Builds:
          - A deduplicated image-ID list (returned)
          - ``self._fewshot_allowed``: { img_id -> set of class names }
            so that ``parse_data_info`` can filter annotations per-image.
        """
        prev_intro = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro = self.owod_cfg.CUR_INTRODUCED_CLS
        novel_classes = list(self.CLASS_NAMES[prev_intro:prev_intro + curr_intro])

        seed_dir = os.path.join(self._fewshot_dir,
                                f'seed{self._fewshot_seed}')

        all_ids = []
        for cls in novel_classes:
            fname = f'box_{self._fewshot_k}shot_{cls}_train.txt'
            fpath = os.path.join(seed_dir, fname)
            if not os.path.exists(fpath):
                self._logger.warning(
                    f'Few-shot file not found: {fpath}  — skipping {cls}')
                continue
            with open(fpath, 'r') as f:
                fileids = [
                    line.strip().split('/')[-1].replace('.jpg', '')
                    for line in f if line.strip()
                ]
            for fid in fileids:
                self._fewshot_allowed.setdefault(fid, set()).add(cls)
            all_ids.extend(fileids)
            self._logger.info(
                f'Few-shot: {cls} — {len(fileids)} images from {fname}')

        # Deduplicate, preserve order
        seen = set()
        unique_ids = []
        for fid in all_ids:
            if fid not in seen:
                seen.add(fid)
                unique_ids.append(fid)

        self.image_set_list = [
            f'fewshot_{self._fewshot_k}shot_seed{self._fewshot_seed}'
        ]
        self._logger.info(
            f'Few-shot: {len(unique_ids)} unique images, '
            f'{len(novel_classes)} classes, k={self._fewshot_k}')
        return unique_ids

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["bbox_label"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["bbox_label"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        total_num_class = self.owod_cfg.num_classes
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
        # for annotation in entry:
            if annotation["bbox_label"] not in known_classes:
                annotation["bbox_label"] = total_num_class - 1
        return entry

    def load_data_list(self):
        data_list = []
        self._logger.info(f"Loading {self.dataset} from {self.image_set_list}...")
        for i, img_id in enumerate(self.imgids):
            raw_data_info = dict(
                img_path=self.images[i],
                img_id=img_id,
            )
            parsed_data_info = self.parse_data_info(raw_data_info)
            data_list.append(parsed_data_info)

        self._logger.info(f"{self.dataset} Loaded, {len(data_list)} images in total")

        # Few-shot k-instance cap per class (like CED-FOOD's np.random.choice)
        if self._fewshot_k > 0 and self._fewshot_allowed:
            data_list = self._cap_fewshot_instances(data_list)

        return data_list

    def _cap_fewshot_instances(self, data_list):
        """Cap at exactly k bbox instances per novel class.

        CED-FOOD returns one dict per bbox and caps with random.choice.
        We keep the image-centric format but drop excess bboxes per class.
        """
        prev_intro = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro = self.owod_cfg.CUR_INTRODUCED_CLS
        k = self._fewshot_k

        # Count instances per novel class across all images
        class_instance_locs = defaultdict(list)
        # (img_idx, inst_idx) for each novel class
        for img_idx, data_info in enumerate(data_list):
            for inst_idx, inst in enumerate(data_info.get('instances', [])):
                lbl = inst['bbox_label']
                if prev_intro <= lbl < prev_intro + curr_intro:
                    class_instance_locs[lbl].append((img_idx, inst_idx))

        # Determine which (img_idx, inst_idx) to keep
        keep_set = set()
        for lbl, locs in class_instance_locs.items():
            cls_name = self.CLASS_NAMES[lbl]
            if len(locs) > k:
                random.seed(self._fewshot_seed)
                sampled = random.sample(locs, k)
                self._logger.info(
                    f'Few-shot cap: {cls_name} {len(locs)} -> {k} instances')
            else:
                sampled = locs
                self._logger.info(
                    f'Few-shot cap: {cls_name} {len(locs)} instances (<= {k}, kept all)')
            keep_set.update(sampled)

        # Rebuild data_list, keeping only selected instances
        new_data_list = []
        for img_idx, data_info in enumerate(data_list):
            new_instances = []
            for inst_idx, inst in enumerate(data_info.get('instances', [])):
                lbl = inst['bbox_label']
                if prev_intro <= lbl < prev_intro + curr_intro:
                    # Novel class — check if selected
                    if (img_idx, inst_idx) in keep_set:
                        new_instances.append(inst)
                else:
                    # Non-novel (shouldn't exist after OWOD filtering, but keep if present)
                    new_instances.append(inst)
            if new_instances:
                data_info_copy = copy.copy(data_info)
                data_info_copy['instances'] = new_instances
                new_data_list.append(data_info_copy)

        self._logger.info(
            f'Few-shot cap: {len(data_list)} -> {len(new_data_list)} images '
            f'with instances')
        return new_data_list
    
    def parse_data_info(self, raw_data_info):
        data_info = copy.copy(raw_data_info)
        img_id = data_info["img_id"]
        tree = ET.parse(self.imgid2annotations[img_id])

        # Per-image allowed classes for few-shot filtering
        allowed_classes = self._fewshot_allowed.get(img_id, None)

        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text

            # Few-shot per-class filter: only keep annotations whose
            # class was the one that "selected" this image.
            # E.g. if this image was listed in box_10shot_bus_train.txt,
            # only bus annotations are kept — truck/car/etc. are skipped.
            if allowed_classes is not None and cls not in allowed_classes:
                continue

            # Only apply VOC cocofied mapping if cls is not already a known class
            # (e.g. IDD uses 'motorcycle' directly, don't remap to 'motorbike')
            if cls not in self.CLASS_NAMES and cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            if cls in self.CLASS_NAMES:
                bbox_label = self.CLASS_NAMES.index(cls)
            else:
                # Class not in dataset's class list — treat as unknown
                # (e.g. pole, tractor, animal in IDD → will be labelled UNK)
                bbox_label = len(self.CLASS_NAMES) - 1
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                bbox_label=bbox_label,
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                ignore_flag=0,
            )
            instances.append(instance)

        if 'train' in self.image_set:
            if self.training_strategy == 1: # oracle
                instances = self.label_known_class_and_unknown(instances)
            else: # owod
                instances = self.remove_prev_class_and_unk_instances(instances)
        elif 'test' in self.image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in self.image_set:
            instances = self.remove_unknown_instances(instances)
            
        data_info.update(
            height=int(tree.findall("./size/height")[0].text),
            width=int(tree.findall("./size/width")[0].text),
            instances=instances,
        )

        return data_info

    def filter_data(self):
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos