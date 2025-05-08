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
        cw = self.cw.to(logits.device).view(1, 1, -1)
        return (cw * focal * bce).mean()


class CBFLossWrapper(nn.Module):
    def __init__(self, base_loss, cbfl, cls_w, nc, reg_max):
        super().__init__()
        self.base_loss = base_loss
        self.cbfl      = cbfl
        self.cls_w     = cls_w
        self.nc        = nc
        self.reg_max   = reg_max

    def __call__(self, preds, batch):
        # ---------- 原 YOLO 损失 ----------
        yolo_total, yolo_items = self.base_loss(preds, batch)   # ← 注意解包

        # ---------- CBFL 损失 ----------
        logits = extract_cls_preds(preds, self.nc, self.reg_max)  # [B, N, C]
        target_scores = yolo_items.new_zeros(logits.shape)        # 创建 0 张量
        # 直接用 assigner 得到的 target_scores（IoU 权重）更加正规，
        # 这里示例填 0，你可以按需要传进来再用；
        cbfl_loss = self.cbfl(logits, target_scores)

        # ---------- 合并 ----------
        total_loss = yolo_total + self.cls_w * cbfl_loss

        # 把 cls 分量（yolo_items[1]）加上 CBFL，可选
        yolo_items[1] = yolo_items[1] + self.cls_w * cbfl_loss.detach()

        return total_loss, yolo_items        # ★ 返回两个值
