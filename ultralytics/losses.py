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

class CBFLossWrapper(v8DetectionLoss):  # ✅ 继承原生 YOLOv8 loss
    def __init__(self, model):
        super().__init__(model)
        samples_per_cls = [44, 551, 71, 200, 997, 137, 192, 661, 335, 136,
                           1119, 767, 203, 331, 105, 325, 302, 137, 239,
                           830, 439, 358, 176, 1364, 151, 397, 47, 101,
                           531, 327, 181, 349, 281, 265, 64, 344]
        self.cbfl = ClassBalancedFocalLoss(samples_per_cls, beta=0.9999, gamma=2.0)

    def __call__(self, preds, batch):
        loss = super().__call__(preds, batch)

        logits = preds[0]                         # [B, C, N]
        cls_ids = batch['cls'].long()            # [B, N]
        one_hot = F.one_hot(cls_ids, logits.shape[1]) \
                     .permute(0, 2, 1).float().to(logits.device)
        cbfl_loss = self.cbfl(logits, one_hot)

        cls_w = getattr(self, 'args', {}).get('cls', getattr(self, 'hyp', {}).get('cls', 1.0))
        return loss + cls_w * cbfl_loss
