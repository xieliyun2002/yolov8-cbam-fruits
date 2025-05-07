import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        super().__init__()
        spc = torch.tensor(samples_per_cls, dtype=torch.float)
        eff = 1 - beta**spc
        w   = (1 - beta) / eff
        w   = w / w.mean()
        self.register_buffer('cw', w)
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce   = self.bce(logits, targets)
        p     = torch.sigmoid(logits)
        pt    = p*targets + (1-p)*(1-targets)
        focal = (1-pt).pow(self.gamma)
        cw    = self.cw.view(1, -1, 1)
        return (cw * focal * bce).mean()

class CBFLossWrapper:
    def __init__(self, base_loss, cbfl, cls_w):
        self.base_loss = base_loss
        self.cbfl = cbfl
        self.cls_w = cls_w

    def __call__(self, preds, batch):
        # 原 YOLO 损失
        yolo_loss = self.base_loss(preds, batch)

        # CBFL 损失
        logits = preds[0]
        cls_ids = batch["cls"].long()
        one_hot = torch.nn.functional.one_hot(cls_ids, logits.shape[1]) \
                    .permute(0, 2, 1).float().to(logits.device)
        cbfl_loss = self.cbfl(logits, one_hot)

        return yolo_loss + self.cls_w * cbfl_loss

