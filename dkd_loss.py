import torch
import torch.nn as nn
import torch.nn.functional as F

def dkd_loss(logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=4.0):
    """
    Decoupled Knowledge Distillation (DKD) Loss function.
    """
    # Create masks to separate the target class from non-target classes
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # Softmax with temperature
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    # TCKD: Target Class Knowledge Distillation
    # We aggregate the prob of non-target classes into a single value
    pred_student_cat = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_cat = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student_cat = torch.log(pred_student_cat + 1e-8)
    
    tckd_loss = (
        F.kl_div(log_pred_student_cat, pred_teacher_cat, reduction='batchmean')
        * (temperature**2)
    )
    
    # NCKD: Non-Target Class Knowledge Distillation
    # We remove the target class and re-normalize the rest
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
        * (temperature**2)
    )
    
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=8.0, temperature=4.0, warmup_epochs=20):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs

    def forward(self, logits_student, logits_teacher, target, epoch):
        # OPTIONAL: WARMUP STRATEGY
        # If we are in early epochs, the "Teacher" (peers) is dumb.
        # We might want to return 0 loss or low weight.
        if epoch < self.warmup_epochs:
            # You can return 0, or just a standard KD loss here
            return torch.tensor(0.0).to(logits_student.device)
            
        return dkd_loss(logits_student, logits_teacher, target, self.alpha, self.beta, self.temperature)