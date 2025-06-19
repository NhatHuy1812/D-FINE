import torch.nn.functional as F
from criterion.boxes.ddf import DDFLoss


class DfineX2LKDLoss:
    """Knowledge‑distillation head to transfer D‑FINE‑X (teacher) → D‑FINE‑L (student).

    Works at the *same* spatial resolution (1280×1280), so no up/down‑sampling is
    required.  It re‑uses the repo's fine‑grained distribution loss (DDFLoss).
    """

    def __init__(self, T: float = 2.0, w_cls: float = 2.0, w_box: float = 4.0,
                 w_feat: float = 1.0, w_aux: float = 1.0):
        self.T = T
        self.w_cls = w_cls
        self.w_box = w_box
        self.w_feat = w_feat
        self.w_aux = w_aux
        self.ddf = DDFLoss()

    def __call__(self, s_out, t_out, s_feats, t_feats):
        """Return (total_loss, dict_of_individual_terms)."""
        # 1) temperature‑scaled KL on class logits
        loss_cls = F.kl_div(
            F.log_softmax(s_out["pred_logits"] / self.T, dim=-1),
            F.softmax(t_out["pred_logits"] / self.T, dim=-1).detach(),
            reduction="batchmean",
        ) * (self.T ** 2)

        # 2) fine‑grained distribution regression KD
        loss_box = self.ddf(s_out["pred_distri"], t_out["pred_distri"].detach())

        # 3) last‑decoder feature matching (MSE)
        loss_feat = F.mse_loss(s_feats[-1], t_feats[-1].detach())

        # 4) auxiliary GO‑LSD heads
        loss_aux = 0.0
        for s_aux, t_aux in zip(s_out["aux_outputs"], t_out["aux_outputs"]):
            loss_aux += self.ddf(s_aux["pred_distri"], t_aux["pred_distri"].detach())
        loss_aux /= len(s_out["aux_outputs"])

        total = (
            self.w_cls * loss_cls
            + self.w_box * loss_box
            + self.w_feat * loss_feat
            + self.w_aux * loss_aux
        )
        return total, {
            "loss_kd_cls": loss_cls,
            "loss_kd_box": loss_box,
            "loss_kd_feat": loss_feat,
            "loss_kd_aux": loss_aux,
        }
