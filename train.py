import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from models import model_dict
import os
from utils import AverageMeter, accuracy
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# ==========================================
# 1. DKD LOSS IMPLEMENTATION
# ==========================================
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student_cat = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_cat = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student_cat = torch.log(pred_student_cat + 1e-8)
    tckd_loss = (
        F.kl_div(log_pred_student_cat, pred_teacher_cat, reduction='batchmean')
        * (temperature**2)
    )
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
    def __init__(self, alpha, beta, temperature, warmup_epochs):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs

    def forward(self, logits_student, logits_teacher, target, epoch):
        if epoch < self.warmup_epochs:
            return torch.tensor(0.0).to(logits_student.device)
        return dkd_loss(logits_student, logits_teacher, target, self.alpha, self.beta, self.temperature)

# ==========================================
# 2. DATA LOADER (Embedded to fix Download Error)
# ==========================================
def get_dataloader(args):
    print(f"==> Preparing Data: {args.dataset}")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if args.dataset == 'cifar100':
        # download=True fixes the FileNotFoundError
        train_set = datasets.CIFAR100(root=args.root, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root=args.root, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        # Fallback to CIFAR10
        train_set = datasets.CIFAR10(root=args.root, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=args.root, train=False, download=True, transform=transform_test)
        num_classes = 10
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return train_loader, test_loader, num_classes

# ==========================================
# 3. ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
parser.add_argument('--mode', type=str, default='dkd', choices=['kdcl', 'dkd'], help='Run standard KDCL or DKD')

parser.add_argument('--T', type=float, default=4.0)
parser.add_argument('--model_names', type=str, nargs='+', default=['resnet32', 'ShuffleV1'])
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)

parser.add_argument('--dkd_alpha', type=float, default=1.0)
parser.add_argument('--dkd_beta', type=float, default=2.0)
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--kdcl_alpha', type=float, default=0.5)

args = parser.parse_args()
args.num_branch = len(args.model_names)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

exp_name = f"{args.dataset}_{args.mode}_{'_'.join(args.model_names)}"
exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)
print(f"Experiment Path: {exp_path}")

# Initialize DKD
criterion_dkd = DKDLoss(alpha=args.dkd_alpha, beta=args.dkd_beta, temperature=args.T, warmup_epochs=args.warmup)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_one_epoch(models, optimizers, train_loader, epoch):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.train()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    for i, (imgs, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            label = label.cuda()

        outputs = torch.zeros(size=(len(models), imgs.size(0), 100), dtype=torch.float).cuda()
        out_list = []
        
        for model_idx, model in enumerate(models):
            out = model.forward(imgs[:, model_idx, ...])
            outputs[model_idx, ...] = out
            out_list.append(out)

        stable_out = outputs.mean(dim=0).detach()

        for model_idx, model in enumerate(models):
            ce_loss = F.cross_entropy(out_list[model_idx], label)
            
            if args.mode == 'dkd':
                dist_loss = criterion_dkd(out_list[model_idx], stable_out, label, epoch)
                loss = ce_loss + dist_loss
            else:
                kl_loss = F.kl_div(
                    F.log_softmax(out_list[model_idx] / args.T, dim=1),
                    F.softmax(stable_out / args.T, dim=1),
                    reduction='batchmean'
                ) * (args.T * args.T)
                loss = (1 - args.kdcl_alpha) * ce_loss + args.kdcl_alpha * kl_loss

            optimizers[model_idx].zero_grad()
            if model_idx < len(models) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizers[model_idx].step()

            loss_recorder_list[model_idx].update(loss.item(), n=imgs.size(0))
            acc = accuracy(out_list[model_idx], label)[0]
            acc_recorder_list[model_idx].update(acc.item(), n=imgs.size(0))

    return [r.avg for r in loss_recorder_list], [r.avg for r in acc_recorder_list]


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
    return [r.avg for r in loss_recorder_list], [r.avg for r in acc_recorder_list]


def train(model_list, optimizer_list, train_loader, scheduler_list):
    best_acc = [-1 for _ in range(args.num_branch)]
    for epoch in range(args.epoch):
        train_losses, train_acces = train_one_epoch(model_list, optimizer_list, train_loader, epoch)
        val_losses, val_acces = evaluation(model_list, val_loader)

        for i in range(len(best_acc)):
            if val_acces[i] > best_acc[i]:
                best_acc[i] = val_acces[i]
                state_dict = dict(epoch=epoch + 1, model=model_list[i].state_dict(), acc=val_acces[i])
                name = os.path.join(exp_path, args.model_names[i], 'ckpt', 'best.pth')
                os.makedirs(os.path.dirname(name), exist_ok=True)
                torch.save(state_dict, name)

            scheduler_list[i].step()

        if (epoch + 1) % args.print_freq == 0:
            for j in range(len(best_acc)):
                print("Model:{} Train Loss:{:.2f} Acc:{:.2f} | Val Loss:{:.2f} Acc:{:.2f}".format(
                    args.model_names[j], train_losses[j], train_acces[j], val_losses[j], val_acces[j]))

    print("\n" + "="*30)
    for k in range(len(best_acc)):
        print("FINAL BEST -> Model:{} Acc:{:.2f}%".format(args.model_names[k], best_acc[k]))
    print("="*30)


if __name__ == '__main__':
    # Use internal get_dataloader instead of importing from data.py
    train_loader, val_loader, num_classes = get_dataloader(args)
    
    model_list = []
    optimizer_list = []
    scheduler_list = []
    
    for name in args.model_names:
        lr = 0.01 if name in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2'] else args.lr
        # Pass num_classes dynamically
        model = model_dict[name](num_classes=num_classes)
        if torch.cuda.is_available(): model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
        model_list.append(model)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    train(model_list, optimizer_list, train_loader, scheduler_list)