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


class CBFLossWrapper:
    def __init__(self, base_loss, cbfl, cls_w, nc, reg_max):
        self.base_loss = base_loss      # v8DetectionLoss 实例
        self.cbfl      = cbfl
        self.cls_w     = cls_w
        self.nc        = nc
        self.reg_max   = reg_max

    def __call__(self, preds, batch):
        # ---------------- YOLO 原生三分损失 ----------------
        yolo_loss, aux = self.base_loss(preds, batch)   # yolo_loss: 标量, aux: (box, cls, dfl)

        # ---------------- 取出分类 logits ------------------
        from ultralytics.losses import extract_cls_preds
        logits = extract_cls_preds(preds, self.nc, self.reg_max)   # [B, N, C]

        # ---------------- 生成 anchor‑级 targets -----------
        # base_loss 在 forward 内部已经做过 TaskAlignedAssigner，
        # 把结果保存为 self.base_loss._cached_target_scores，直接复用
        target_scores = self.base_loss._cached_target_scores.to(logits.device)  # [B, N, C]

        # target_scores 对正样本 anchor 的 (cls_id) 位置 = IoU 权重，
        # 其余 = 0，天然就是 one‑hot ∧ 权重矩阵；直接拿来用
        cbfl_loss = self.cbfl(logits, target_scores)

        # ---------------- 合并 ----------------------------
        return yolo_loss + self.cls_w * cbfl_loss
