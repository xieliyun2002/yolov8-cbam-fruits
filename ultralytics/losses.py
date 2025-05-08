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
        eff = 1 - beta**spc
        w   = (1 - beta) / eff
        w   = w / w.mean()
        self.register_buffer('cw', w)
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (1 - pt).pow(self.gamma)

        # ✅ 修复：将 cw 移到 logits 同一设备上
        cw = self.cw.view(1, 1, -1).to(logits.device)

        return (cw * focal * bce).mean()

class CBFLossWrapper(torch.nn.Module):
    def __init__(self, base_loss, cbfl, cls_weight):
        super().__init__()
        self.base_loss = base_loss          # 原 YOLO 损失（box+dfl+…）
        self.cbfl      = cbfl               # ClassBalancedFocalLoss
        self.cls_w     = cls_weight         # 分类损失权重

    def forward(self, preds, batch):
        # 1) 原始 YOLO 损失
        det_loss = self.base_loss(preds, batch)

        # 2) 取出分类 logits  [B, N, C]
        feats        = preds[1] if isinstance(preds, tuple) else preds
        b            = feats[0].shape[0]
        nc           = self.cbfl.cw.numel()
        reg_max      = self.base_loss.reg_max
        _, cls_pred  = torch.cat([x.view(b, nc + reg_max*4, -1) for x in feats], 2)\
                         .split((reg_max*4, nc), 1)
        cls_pred     = cls_pred.permute(0, 2, 1).contiguous()   # [B,N,C]

        # 3) 生成 one‑hot GT（忽略无效 anchor）
        cls_id   = batch['cls'].long().squeeze(-1)              # [B,N]
        pos_mask = (cls_id != -1)                               # -1 表示背景
        targets  = torch.zeros_like(cls_pred)
        targets.scatter_(2, cls_id.unsqueeze(-1), 1.0)
        targets  = targets * pos_mask.unsqueeze(-1)

        # 4) CBFL 分类损失
        loss_cls = self.cbfl(cls_pred, targets)

        return det_loss + self.cls_w * loss_cls
