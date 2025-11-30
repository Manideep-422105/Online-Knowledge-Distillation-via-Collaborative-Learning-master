# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import MultiStepLR
# import torch.nn.functional as F
# from data import get_dataloader
# from models import model_dict
# import os
# from utils import AverageMeter, accuracy
# import numpy as np
# from datetime import datetime
# from tqdm import tqdm

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# # ==========================================
# # DKD LOSS IMPLEMENTATION (Embedded)
# # ==========================================
# def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
#     gt_mask = _get_gt_mask(logits_student, target)
#     other_mask = _get_other_mask(logits_student, target)
    
#     pred_student = F.softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
#     pred_student_cat = cat_mask(pred_student, gt_mask, other_mask)
#     pred_teacher_cat = cat_mask(pred_teacher, gt_mask, other_mask)
#     log_pred_student_cat = torch.log(pred_student_cat + 1e-8)
    
#     tckd_loss = (
#         F.kl_div(log_pred_student_cat, pred_teacher_cat, reduction='batchmean')
#         * (temperature**2)
#     )
    
#     pred_teacher_part2 = F.softmax(
#         logits_teacher / temperature - 1000.0 * gt_mask, dim=1
#     )
#     log_pred_student_part2 = F.log_softmax(
#         logits_student / temperature - 1000.0 * gt_mask, dim=1
#     )
    
#     nckd_loss = (
#         F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
#         * (temperature**2)
#     )
    
#     return alpha * tckd_loss + beta * nckd_loss

# def _get_gt_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
#     return mask

# def _get_other_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
#     return mask

# def cat_mask(t, mask1, mask2):
#     t1 = (t * mask1).sum(dim=1, keepdims=True)
#     t2 = (t * mask2).sum(1, keepdims=True)
#     rt = torch.cat([t1, t2], dim=1)
#     return rt

# class DKDLoss(nn.Module):
#     def __init__(self, alpha, beta, temperature, warmup_epochs):
#         super(DKDLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.temperature = temperature
#         self.warmup_epochs = warmup_epochs

#     def forward(self, logits_student, logits_teacher, target, epoch):
#         # Warmup Strategy:
#         # If early epoch, return 0.0 loss (only learn from GT).
#         # Otherwise, the students will memorize the random noise of their peers.
#         if epoch < self.warmup_epochs:
#             return torch.tensor(0.0).to(logits_student.device)
            
#         return dkd_loss(logits_student, logits_teacher, target, self.alpha, self.beta, self.temperature)

# # ==========================================
# # MAIN TRAINING SCRIPT
# # ==========================================

# parser = argparse.ArgumentParser()
# parser.add_argument('--T', type=float, default=4.0)  # temperature
# parser.add_argument('--model_names', type=str, nargs='+', default=['resnet20', 'resnet20'])

# # Replaced original alpha with DKD specific params
# parser.add_argument('--dkd_alpha', type=float, default=1.0, help='Weight for TCKD (Target Class)')
# parser.add_argument('--dkd_beta', type=float, default=8.0, help='Weight for NCKD (Non-Target Class)')
# parser.add_argument('--warmup', type=int, default=20, help='Epochs to wait before enabling DKD')

# parser.add_argument('--root', type=str, default='dataset')
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--num_workers', type=int, default=1)
# parser.add_argument('--epoch', type=int, default=240)

# parser.add_argument('--lr', type=float, default=0.05)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--weight-decay', type=float, default=5e-4)
# parser.add_argument('--gamma', type=float, default=0.1)
# parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])

# parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--gpu-id', type=int, default=0)
# parser.add_argument('--print_freq', type=int, default=100)

# args = parser.parse_args()
# args.num_branch = len(args.model_names)

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# exp_name = '_'.join(args.model_names) + '_DKD'
# exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
# os.makedirs(exp_path, exist_ok=True)
# print(f"Experiment Path: {exp_path}")

# # Initialize DKD Loss Criterion
# criterion_dkd = DKDLoss(alpha=args.dkd_alpha, beta=args.dkd_beta, temperature=args.T, warmup_epochs=args.warmup)

# def train_one_epoch(models, optimizers, train_loader, epoch):
#     acc_recorder_list = []
#     loss_recorder_list = []
#     for model in models:
#         model.train()
#         acc_recorder_list.append(AverageMeter())
#         loss_recorder_list.append(AverageMeter())

#     for i, (imgs, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")):
#         out_list = []
        
#         # 1. Forward pass for all models
#         if torch.cuda.is_available():
#             imgs = imgs.cuda()
#             label = label.cuda()

#         for model_idx, model in enumerate(models):
#             out = model.forward(imgs[:, model_idx, ...])
#             out_list.append(out)

#         # 2. Calculate Loss (Peer-to-Peer)
#         for student_idx, student_model in enumerate(models):
            
#             # Identify the Teacher (The peer)
#             # If 2 models: 0 learns from 1, 1 learns from 0
#             teacher_idx = (student_idx + 1) % len(models)
            
#             logits_student = out_list[student_idx]
#             logits_teacher = out_list[teacher_idx].detach() # Detach teacher!

#             # 1. CE Loss
#             ce_loss = F.cross_entropy(logits_student, label)
            
#             # 2. DKD Loss (One-on-One)
#             dkd_loss_val = criterion_dkd(logits_student, logits_teacher, label, epoch)

#             loss = ce_loss + dkd_loss_val

#             optimizers[student_idx].zero_grad()
            
#             if student_idx < len(models) - 1:
#                 loss.backward(retain_graph=True)
#             else:
#                 loss.backward()

#             optimizers[student_idx].step()

#             loss_recorder_list[student_idx].update(loss.item(), n=imgs.size(0))
#             acc = accuracy(logits_student, label)[0]
#             acc_recorder_list[student_idx].update(acc.item(), n=imgs.size(0))

#     losses = [recorder.avg for recorder in loss_recorder_list]
#     acces = [recorder.avg for recorder in acc_recorder_list]
#     return losses, acces
    
# def evaluation(models, val_loader):
#     acc_recorder_list = []
#     loss_recorder_list = []
#     for model in models:
#         model.eval()
#         acc_recorder_list.append(AverageMeter())
#         loss_recorder_list.append(AverageMeter())

#     with torch.no_grad():
#         for img, label in val_loader:
#             if torch.cuda.is_available():
#                 img = img.cuda()
#                 label = label.cuda()

#             for model_idx, model in enumerate(models):
#                 out = model(img)
#                 acc = accuracy(out, label)[0]
#                 loss = F.cross_entropy(out, label)
#                 acc_recorder_list[model_idx].update(acc.item(), img.size(0))
#                 loss_recorder_list[model_idx].update(loss.item(), img.size(0))
#     losses = [recorder.avg for recorder in loss_recorder_list]
#     acces = [recorder.avg for recorder in acc_recorder_list]
#     return losses, acces


# def train(model_list, optimizer_list, train_loader, scheduler_list):
#     best_acc = [-1 for _ in range(args.num_branch)]
#     for epoch in range(args.epoch):
#         train_losses, train_acces = train_one_epoch(model_list, optimizer_list, train_loader, epoch)
#         val_losses, val_acces = evaluation(model_list, val_loader)

#         for i in range(len(best_acc)):
#             if val_acces[i] > best_acc[i]:
#                 best_acc[i] = val_acces[i]
#                 state_dict = dict(epoch=epoch + 1, model=model_list[i].state_dict(),
#                                   acc=val_acces[i])
#                 name = os.path.join(exp_path, args.model_names[i], 'ckpt', 'best.pth')
#                 os.makedirs(os.path.dirname(name), exist_ok=True)
#                 torch.save(state_dict, name)

#             scheduler_list[i].step()

#         if (epoch + 1) % args.print_freq == 0 or epoch < 5: # Print early epochs too
#             for j in range(len(best_acc)):
#                 print("Epoch [{}] model:{} train loss:{:.2f} acc:{:.2f}  val loss{:.2f} acc:{:.2f}".format(
#                     epoch+1, args.model_names[j], train_losses[j], train_acces[j], val_losses[j],
#                     val_acces[j]))

#     for k in range(len(best_acc)):
#         print("model:{} best acc:{:.2f}".format(args.model_names[k], best_acc[k]))


# if __name__ == '__main__':
#     train_loader, val_loader = get_dataloader(args)
#     model_list = []
#     optimizer_list = []
#     scheduler_list = []
#     for name in args.model_names:
#         lr = 0.01 if name in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2'] else args.lr
#         model = model_dict[name](num_classes=100)
#         if torch.cuda.is_available(): model = model.cuda()

#         optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
#                               weight_decay=args.weight_decay)
#         scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
#         model_list.append(model)
#         optimizer_list.append(optimizer)
#         scheduler_list.append(scheduler)

#     train(model_list, optimizer_list, train_loader, scheduler_list)

import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from data import get_dataloader
from models import model_dict
import os
from utils import AverageMeter, accuracy
import numpy as np
from datetime import datetime
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=float, default=4.0)  # temperature
parser.add_argument('--model_names', type=str, nargs='+', default=['resnet20', 'resnet20'])
parser.add_argument('--alpha', type=float, default=0.5)  # weight for ce and kl

parser.add_argument('--root', type=str, default='dataset')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--epoch', type=int, default=240)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
args.num_branch = len(args.model_names)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

exp_name = '_'.join(args.model_names)
exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)
print(exp_path)


def train_one_epoch(models, optimizers, train_loader,epoch):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.train()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())
# After
    for i, (imgs, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")):
        # torch.Size([batch, num_model, 3, 32, 32]) torch.Size([batch])
        outputs = torch.zeros(size=(len(models), imgs.size(0), 100), dtype=torch.float).cuda()
        out_list = []
        # forward
        for model_idx, model in enumerate(models):

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()

            out = model.forward(imgs[:, model_idx, ...])
            outputs[model_idx, ...] = out
            out_list.append(out)

        # backward
        stable_out = outputs.mean(dim=0)
        stable_out = stable_out.detach()

        for model_idx, model in enumerate(models):
            ce_loss = F.cross_entropy(out_list[model_idx], label)
            div_loss = F.kl_div(
                F.log_softmax(out_list[model_idx] / args.T, dim=1),
                F.softmax(stable_out / args.T, dim=1),
                reduction='batchmean'
            ) * args.T * args.T

            loss = (1 - args.alpha) * ce_loss + (args.alpha) * div_loss

            optimizers[model_idx].zero_grad()
            if model_idx < len(models) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            optimizers[model_idx].step()

            loss_recorder_list[model_idx].update(loss.item(), n=imgs.size(0))
            acc = accuracy(out_list[model_idx], label)[0]
            acc_recorder_list[model_idx].update(acc.item(), n=imgs.size(0))

    losses = [recorder.avg for recorder in loss_recorder_list]
    acces = [recorder.avg for recorder in acc_recorder_list]
    return losses, acces


def evaluation(models, val_loader):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.eval()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    with torch.no_grad():
        for img, label in val_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            for model_idx, model in enumerate(models):
                out = model(img)
                acc = accuracy(out, label)[0]
                loss = F.cross_entropy(out, label)
                acc_recorder_list[model_idx].update(acc.item(), img.size(0))
                loss_recorder_list[model_idx].update(loss.item(), img.size(0))
    losses = [recorder.avg for recorder in loss_recorder_list]
    acces = [recorder.avg for recorder in acc_recorder_list]
    return losses, acces


def train(model_list, optimizer_list, train_loader, scheduler_list):
    best_acc = [-1 for _ in range(args.num_branch)]
    for epoch in range(args.epoch):
        train_losses, train_acces = train_one_epoch(model_list, optimizer_list, train_loader,epoch)
        val_losses, val_acces = evaluation(model_list, val_loader)

        for i in range(len(best_acc)):
            if val_acces[i] > best_acc[i]:
                best_acc[i] = val_acces[i]
                state_dict = dict(epoch=epoch + 1, model=model_list[i].state_dict(),
                                  acc=val_acces[i])
                name = os.path.join(exp_path, args.model_names[i], 'ckpt', 'best.pth')
                os.makedirs(os.path.dirname(name), exist_ok=True)
                torch.save(state_dict, name)

            scheduler_list[i].step()

        if (epoch + 1) % args.print_freq == 0:
            for j in range(len(best_acc)):
                print("model:{} train loss:{:.2f} acc:{:.2f}  val loss{:.2f} acc:{:.2f}".format(
                    args.model_names[j], train_losses[j], train_acces[j], val_losses[j],
                    val_acces[j]))

    for k in range(len(best_acc)):
        print("model:{} best acc:{:.2f}".format(args.model_names[k], best_acc[k]))


if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(args)
    model_list = []
    optimizer_list = []
    scheduler_list = []
    for name in args.model_names:
        lr = 0.01 if name in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2'] else args.lr
        model = model_dict[name](num_classes=100)
        if torch.cuda.is_available(): model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
        model_list.append(model)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    train(model_list, optimizer_list, train_loader, scheduler_list)