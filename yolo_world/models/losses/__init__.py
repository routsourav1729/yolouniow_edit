# Copyright (c) Tencent Inc. All rights reserved.
from .dynamic_loss import CoVMSELoss
from .hardneg_contrastive import HardNegativeContrastiveLoss

__all__ = ['CoVMSELoss', 'HardNegativeContrastiveLoss']
