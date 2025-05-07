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
    def __init__(self, yolo_loss, cbfl, cls_w=1.0):
        super().__init__()
        self.yolo_loss = yolo_loss
        self.cbfl = cbfl
        self.cls_w = cls_w

    def forward(self, preds, batch):
        # 原 YOLOv8 损失
        yolo_loss, loss_items = self.yolo_loss(preds, batch)

        # 获取分类 logits
        pred_cls = preds[1]  # YOLOv8 的 preds 是 (pred_box, pred_cls, pred_dfl)
        gt_labels = batch["cls"]
        target_scores = batch["batch_idx"].unsqueeze(1).float()  # 或根据你数据定义设置 mask

        # One-hot 标签构造
        cls_ids = gt_labels.squeeze(-1).long()                # (B, N)
        one_hot = F.one_hot(cls_ids, num_classes=pred_cls.shape[1]).float()  # (B, N, C)
        one_hot = one_hot.permute(0, 2, 1).contiguous()        # (B, C, N)

        cbfl_loss = self.cbfl(pred_cls, one_hot.to(pred_cls.dtype))

        # 返回总损失 + 各项
        total_loss = yolo_loss + self.cls_w * cbfl_loss
        return total_loss, loss_items
