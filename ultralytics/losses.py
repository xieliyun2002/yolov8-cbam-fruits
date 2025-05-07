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
        bce   = self.bce(logits, targets)
        p     = torch.sigmoid(logits)
        pt    = p*targets + (1-p)*(1-targets)
        focal = (1-pt).pow(self.gamma)
        cw    = self.cw.view(1, -1, 1)
        return (cw * focal * bce).mean()

class CBFLossWrapper(nn.Module):
    def __init__(self, base_loss, cbfl, cls_w):
        super().__init__()
        self.base_loss = base_loss
        self.cbfl      = cbfl
        self.cls_w     = cls_w

        # <<<<<< 修 正 处 >>>>>>
        self.nc       = cbfl.cw.numel()          # ← 代替 cbfl.num_classes
        self.reg_max  = getattr(base_loss, 'reg_max', 16)

    def forward(self, preds, batch):
        # 1. 基础 YOLOv8 损失
        yolo_loss = self.base_loss(preds, batch)

        # 2. 提取分类 logits：preds[1] 是 Detect 的输出（List[Tensors]）
        logits = extract_cls_preds(preds, self.nc, self.reg_max)  # => [B, N, C]

        # 3. 获取目标标签
        gt_labels = batch['cls']  # shape: [B, N, 1]
        target_scores = batch['batch_idx'].unsqueeze(-1).float() * 0 + 1.0  # 所有为1，等效于 valid mask

        # 4. 构建 one-hot 标签
        cls_ids = gt_labels.squeeze(-1).long()               # [B, N]
        one_hot = F.one_hot(cls_ids, num_classes=self.nc)    # [B, N, C]
        one_hot = one_hot * target_scores.unsqueeze(-1)      # 掩码无效项
        one_hot = one_hot.to(dtype=logits.dtype)             # 与 logits 对齐

        # 5. 计算分类损失
        cbfl_loss = self.cbfl(logits, one_hot)               # logits: [B, N, C], one_hot: [B, N, C]

        # 6. 加权合并
        return yolo_loss + self.cls_w * cbfl_loss
