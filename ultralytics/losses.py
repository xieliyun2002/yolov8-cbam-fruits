import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss


def extract_cls_preds(preds, nc, reg_max):
    feats = preds[1] if isinstance(preds, tuple) else preds  # Detect 输出
    no = nc + reg_max * 4
    b = feats[0].shape[0]

    cls_list = []
    for x in feats:  # 每层 [B, no, H, W]
        x = x.view(b, no, -1)         # [B, no, HW]
        cls_pred = x[:, reg_max*4:, :]  # [B, nc, HW]
        cls_list.append(cls_pred)

    cls_preds = torch.cat(cls_list, dim=2)  # [B, nc, N]
    return cls_preds.permute(0, 2, 1).contiguous()  # [B, N, nc]


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        super().__init__()
        spc = torch.tensor(samples_per_cls, dtype=torch.float)
        eff = 1 - beta ** spc
        w   = (1 - beta) / eff
        w   = w / w.mean()
        self.register_buffer('cw', w)
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce   = self.bce(logits, targets)
        p     = torch.sigmoid(logits)
        pt    = p * targets + (1 - p) * (1 - targets)
        focal = (1 - pt).pow(self.gamma)
        cw    = self.cw.view(1, 1, -1)          # 修正
        return (cw * focal * bce).mean()


class CBFLossWrapper:
    def __init__(self, base_loss, cbfl, cls_w, nc, reg_max):
        self.base_loss, self.cbfl, self.cls_w = base_loss, cbfl, cls_w
        self.nc, self.reg_max = nc, reg_max

    def __call__(self, preds, batch):
        # 原 YOLO loss
        yolo_loss = self.base_loss(preds, batch)

        # 分类 logits 提取
        logits = extract_cls_preds(preds, self.nc, self.reg_max)  # (B, N, C)
        cls_ids = batch['cls'].long().squeeze(-1)                 # (B, N)
        one_hot = F.one_hot(cls_ids, self.nc).float().to(logits.device)

        cbfl_loss = self.cbfl(logits, one_hot)
        return yolo_loss + self.cls_w * cbfl_loss
