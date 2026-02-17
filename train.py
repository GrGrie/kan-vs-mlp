# ==========================================
# Example command to run:
# python train.py --wandb --dataset waterbirds --head kan --kan_reg_weight 0.01
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import time
import math
import argparse
import os
import random
import numpy as np
from datetime import datetime
import sys
from collections import deque
from dotenv import load_dotenv
import wandb
from wilds.get_dataset import get_dataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import get_train_loader, get_eval_loader

# ==========================================
# 1. Efficient KAN Linear Layer Implementation (Official)
# ==========================================
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # Robust least squares solution using normal equations with regularization:
        # (A^T A + reg * I) X = A^T B
        
        # Robust least squares solution using normal equations with regularization:
        # (A^T A + reg * I) X = A^T B
        
        AT = A.transpose(-1, -2)
        ATA = AT @ A
        ATB = AT @ B
        
        # Regularization term to ensure invertibility
        reg = 1e-5 * torch.eye(ATA.shape[-1], device=ATA.device).unsqueeze(0)
        
        solution = torch.linalg.solve(ATA + reg, ATB) # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        if regularization_loss_activation == 0:
            p = l1_fake * 0
        else:
            p = l1_fake / regularization_loss_activation
        
        # Safe entropy computation provided: 0 log 0 = 0
        entropy_term = torch.zeros_like(p)
        mask = p > 0
        entropy_term[mask] = p[mask] * p[mask].log()
        regularization_loss_entropy = -torch.sum(entropy_term)
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

# ==========================================
# 2. Models: ResNet18 Backbone + Heads
# ==========================================

class ResNetBackbone(nn.Module):
    def __init__(self, input_channels=3, image_size=32):
        super().__init__()
        # Using 'weights=None' to avoid warnings, trying ImageNet weights
        base_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Adapt for small images (32x32 or similar)
        if image_size <= 64:
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity()
        else:
            # For larger images, keep standard architecture but adjust input channels
            if input_channels != 3:
                base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = 512

    def forward(self, x):
        x = self.encoder(x)
        return torch.flatten(x, 1)

class ModelWithMLPHead(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=256, input_channels=3, image_size=32):
        super().__init__()
        self.backbone = ResNetBackbone(input_channels=input_channels, image_size=image_size)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

class ModelWithKANHead(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=256, input_channels=3, image_size=32):
        super().__init__()
        self.backbone = ResNetBackbone(input_channels=input_channels, image_size=image_size)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.backbone.feature_dim),
            KANLinear(self.backbone.feature_dim, hidden_dim, grid_size=5, spline_order=3),
            nn.LayerNorm(hidden_dim),
            KANLinear(hidden_dim, num_classes, grid_size=5, spline_order=3)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

# ==========================================
# 3. Training Utilities
# ==========================================

class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()
    
    def print(self, message):
        """Print with explicit newline"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def create_log_filename(args):
    """Create unique log filename based on configuration and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"train_{args.dataset}_{args.head}_h{args.hidden_dim}_e{args.epochs}_{timestamp}.txt"
    log_dir = os.path.join(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, log_name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def update_kan_grids(model, loader, device, max_batches=10):
    """
    Update KAN grids from deterministic train batches.
    """
    if loader is None or not hasattr(model, "head"):
        return

    was_training = model.training
    model.eval()

    head_bn_layers = [m for m in model.head.modules() if isinstance(m, nn.BatchNorm1d)]
    head_bn_was_training = [bn.training for bn in head_bn_layers]
    for bn in head_bn_layers:
        bn.train()

    kan_layer_indices = [i for i, layer in enumerate(model.head) if isinstance(layer, KANLinear)]
    if not kan_layer_indices:
        for bn, mode in zip(head_bn_layers, head_bn_was_training):
            bn.train(mode)
        if was_training:
            model.train()
        return

    layer_inputs = {idx: [] for idx in kan_layer_indices}
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        if len(batch) == 3:
            inputs, _, _ = batch
        else:
            inputs, _ = batch
        inputs = inputs.to(device, non_blocking=True)

        x = model.backbone(inputs)
        for i, layer in enumerate(model.head):
            if i in layer_inputs:
                layer_inputs[i].append(x.detach())
            x = layer(x)

    for i in kan_layer_indices:
        if layer_inputs[i]:
            model.head[i].update_grid(torch.cat(layer_inputs[i], dim=0))

    for bn, mode in zip(head_bn_layers, head_bn_was_training):
        bn.train(mode)
    if was_training:
        model.train()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None, kan_reg_weight=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    desc = f"Epoch {epoch}" if epoch is not None else "Training"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if kan_reg_weight > 0 and hasattr(model, 'head'):
            for layer in model.head:
                if isinstance(layer, KANLinear):
                    loss += kan_reg_weight * layer.regularization_loss(
                        regularize_activation=1.0,
                        regularize_entropy=1.0,
                    )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if math.isnan(loss.item()):
            print("Loss is NaN! Stopping training.")
            sys.exit(1)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    return running_loss / len(loader), 100. * correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloaders(data_dir, batch_size, dataset_name='cifar10'):
    """
    Get dataloaders and evaluation grouper for the specified dataset.

    Returns:
        trainloader, valloader, testloader, gridloader,
        num_classes, input_channels, image_size, eval_grouper
    """
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader, testloader, None, 10, 3, 32, None

    if dataset_name == 'cmnist':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_eval = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = get_dataset(dataset='cmnist', root_dir=data_dir)
        groupby_fields = ['background', 'y']
        eval_grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=groupby_fields)

        train_data = dataset.get_subset('train', transform=transform_train)
        val_data = dataset.get_subset('val', transform=transform_eval)
        test_data = dataset.get_subset('test', transform=transform_eval)
        grid_data = dataset.get_subset('train', transform=transform_eval)

        trainloader = get_train_loader('standard', train_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        valloader = get_eval_loader('standard', val_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        testloader = get_eval_loader('standard', test_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        gridloader = get_eval_loader('standard', grid_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

        return trainloader, valloader, testloader, gridloader, dataset.n_classes, 3, 32, eval_grouper

    if dataset_name == 'metashift':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = get_dataset(dataset='metashift', root_dir=data_dir)
        groupby_fields = ['env', 'y']
        eval_grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=groupby_fields)

        train_data = dataset.get_subset('train', transform=transform_train)
        val_data = dataset.get_subset('val', transform=transform_eval)
        test_data = dataset.get_subset('test', transform=transform_eval)
        grid_data = dataset.get_subset('train', transform=transform_eval)

        trainloader = get_train_loader('standard', train_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        valloader = get_eval_loader('standard', val_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        testloader = get_eval_loader('standard', test_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        gridloader = get_eval_loader('standard', grid_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

        return trainloader, valloader, testloader, gridloader, dataset.n_classes, 3, 224, eval_grouper

    if dataset_name == 'waterbirds':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = get_dataset(dataset='waterbirds', root_dir=data_dir)
        groupby_fields = ['background', 'y']
        eval_grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=groupby_fields)

        train_data = dataset.get_subset('train', transform=transform_train)
        val_data = dataset.get_subset('val', transform=transform_eval)
        test_data = dataset.get_subset('test', transform=transform_eval)
        grid_data = dataset.get_subset('train', transform=transform_eval)

        trainloader = get_train_loader('standard', train_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        valloader = get_eval_loader('standard', val_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        testloader = get_eval_loader('standard', test_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        gridloader = get_eval_loader('standard', grid_data, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

        return trainloader, valloader, testloader, gridloader, dataset.n_classes, 3, 224, eval_grouper

    raise ValueError(f"Unknown dataset: {dataset_name}. Choose from 'cifar10', 'cmnist', 'metashift', 'waterbirds'")

def evaluate(model, loader, criterion, device, grouper=None):
    """
    WILDS-style evaluation with metadata-derived groups.

    Returns:
        loss, overall_acc, worst_group_acc, group_accs
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    n_groups = grouper.n_groups if grouper is not None else None
    group_correct = torch.zeros(n_groups, dtype=torch.float64) if n_groups is not None else None
    group_total = torch.zeros(n_groups, dtype=torch.float64) if n_groups is not None else None

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, targets, metadata = batch
            else:
                inputs, targets = batch
                metadata = None

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            batch_correct = predicted.eq(targets)
            total += targets.size(0)
            correct += batch_correct.sum().item()

            if grouper is not None and metadata is not None:
                groups = grouper.metadata_to_group(metadata, return_counts=False).cpu()
                group_total += torch.bincount(groups, minlength=n_groups).double()
                if batch_correct.any():
                    group_correct += torch.bincount(groups[batch_correct.cpu()], minlength=n_groups).double()

    overall_acc = 100.0 * correct / max(total, 1)

    worst_group_acc = None
    group_accs = None
    if grouper is not None:
        valid = group_total > 0
        group_accs = torch.zeros_like(group_total)
        group_accs[valid] = (group_correct[valid] / group_total[valid]) * 100.0
        worst_group_acc = group_accs[valid].min().item() if valid.any() else 0.0

    return running_loss / len(loader), overall_acc, worst_group_acc, group_accs


def format_wandb_run_name(args):
    run_name = f"{args.head}_seed{args.seed}"
    if args.kan_reg_weight != 0:
        run_name += f"_regWeight{args.kan_reg_weight}"
    run_name += f"_{args.dataset}_lr{args.lr}"
    return run_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=str, required=True, choices=["mlp", "kan"])
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cmnist", "metashift", "waterbirds"],
                    help="Dataset to use: cifar10, cmnist (Colored MNIST), metashift, or waterbirds")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", type=str, default="./best.pt")
    ap.add_argument("--kan_reg_weight", type=float, default=0.0,
                    help="Weight for KAN regularization loss (0 disables it)")
    ap.add_argument("--no_grid_update", action="store_true", help="Disable KAN grid update")
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="Enable Weights & Biases experiment tracking")
    args = ap.parse_args()

    # ---- W&B setup (reads WANDB_API_KEY from .env) ----
    load_dotenv()
    if args.wandb:
        wandb.login()
        run_name = format_wandb_run_name(args)
        wandb.init(project="kan-vs-mlp", name=run_name, config=vars(args))

        # Canonical metric names used in logs.
        wandb.define_metric("epoch")
        wandb.define_metric("train/loss", step_metric="epoch", summary="min")
        wandb.define_metric("train/acc", step_metric="epoch", summary="max")
        wandb.define_metric("val/loss", step_metric="epoch", summary="min")
        wandb.define_metric("val/acc", step_metric="epoch", summary="max")
        wandb.define_metric("val/avg10_acc", step_metric="epoch", summary="max")
        wandb.define_metric("val/best_acc", step_metric="epoch", summary="max")
        wandb.define_metric("test/loss", step_metric="epoch", summary="min")
        wandb.define_metric("test/acc", step_metric="epoch", summary="max")

        # Aliases requested for clean default charts.
        wandb.define_metric("loss", step_metric="epoch", summary="min")
        wandb.define_metric("val_loss", step_metric="epoch", summary="min")
        wandb.define_metric("test_loss", step_metric="epoch", summary="min")
        wandb.define_metric("train_accuracy", step_metric="epoch", summary="max")
        wandb.define_metric("val_accuracy", step_metric="epoch", summary="max")
        wandb.define_metric("test_accuracy", step_metric="epoch", summary="max")
        wandb.define_metric("val_avg10_accuracy", step_metric="epoch", summary="max")
        wandb.define_metric("val_best_accuracy", step_metric="epoch", summary="max")

        wandb.define_metric("val/worst_group_acc", step_metric="epoch", summary="max")
        wandb.define_metric("val/worst_group_avg10", step_metric="epoch", summary="max")
        wandb.define_metric("val/worst_group_best", step_metric="epoch", summary="max")
        wandb.define_metric("test/worst_group_acc", step_metric="epoch", summary="max")

    # Create log file with unique name
    log_file = create_log_filename(args)
    logger = Logger(log_file)
    
    # Redirect stdout to logger
    sys.stdout = logger

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Log training configuration
    print("="*60)
    print(f"Training Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Head: {args.head}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.wd}")
    print(f"KAN reg weight: {args.kan_reg_weight}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Log file: {log_file}")
    print(f"WandB: {'enabled' if args.wandb else 'disabled'}")
    print("="*60)
    print()
    
    # Get dataloaders with dataset-specific configuration
    trainloader, valloader, testloader, gridloader, num_classes, input_channels, image_size, eval_grouper = get_dataloaders(
        args.data_dir, args.batch_size, args.dataset
    )
    
    # Create model with appropriate number of classes and input channels
    if args.head == "mlp":
        model = ModelWithMLPHead(
            num_classes=num_classes, 
            hidden_dim=args.hidden_dim,
            input_channels=input_channels,
            image_size=image_size
        ).to(device)
    else:
        model = ModelWithKANHead(
            num_classes=num_classes, 
            hidden_dim=args.hidden_dim,
            input_channels=input_channels,
            image_size=image_size
        ).to(device)

    print(f"Dataset: {args.dataset.upper()} | Device: {device} | Model: {args.head.upper()} | Params: {count_parameters(model):,}")
    print()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    last_10_val_accs = deque(maxlen=10)

    has_groups = eval_grouper is not None
    if has_groups:
        best_val_worst_group_acc = 0.0
        last_10_val_worst_group_accs = deque(maxlen=10)
        print(f"Group tracking enabled: {eval_grouper.n_groups} groups")
        print()
    
    start_time = time.time()
    for epoch in range(args.epochs):
        # Update KAN grid at the start of each epoch (for KAN models only)
        if args.head == "kan" and not args.no_grid_update:
            update_kan_grids(model, gridloader if gridloader is not None else trainloader, device, max_batches=10)
        
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch=epoch+1, kan_reg_weight=args.kan_reg_weight)
        val_loss, val_acc, val_worst_group_acc, _ = evaluate(model, valloader, criterion, device, grouper=eval_grouper)
        test_loss, test_acc, test_worst_group_acc, _ = evaluate(model, testloader, criterion, device, grouper=eval_grouper)
        last_10_val_accs.append(val_acc)
        avg_10_val_acc = sum(last_10_val_accs) / len(last_10_val_accs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

        # Update best validation worst-group accuracy
        if has_groups and val_worst_group_acc is not None and val_worst_group_acc > best_val_worst_group_acc:
            best_val_worst_group_acc = val_worst_group_acc
        
        # Build log message
        log_msg = (
            f"Ep {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}"
            f" | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Test: {test_acc:.1f}%"
            f" | Val-Avg10: {avg_10_val_acc:.1f}% | Val-Best: {best_val_acc:.1f}%"
        )
        
        # Add worst-group metrics if applicable
        if has_groups and val_worst_group_acc is not None:
            last_10_val_worst_group_accs.append(val_worst_group_acc)
            avg_10_val_worst_group = sum(last_10_val_worst_group_accs) / len(last_10_val_worst_group_accs)
            log_msg += f" | Val-WG: {val_worst_group_acc:.1f}% | Val-WG-Avg10: {avg_10_val_worst_group:.1f}% | Val-WG-Best: {best_val_worst_group_acc:.1f}%"
            if test_worst_group_acc is not None:
                log_msg += f" | Test-WG: {test_worst_group_acc:.1f}%"
            
        print(log_msg)
        
        # ---- W&B logging every 10th epoch ----
        if args.wandb and (epoch + 1) % 10 == 0:
            wb_log = {
                "epoch": epoch + 1,
                "Train/Loss": train_loss,
                "Val/Loss": val_loss,
                "Test/Loss": test_loss,
                "Train/Accuracy": train_acc,
                "Val/Accuracy": val_acc,
                "Test/Accuracy": test_acc,
                "Val/Avg10_accuracy": avg_10_val_acc,
                "Val/Best_accuracy": best_val_acc,
            }
            if has_groups and val_worst_group_acc is not None:
                wb_log["Val/WorstGroupAccuracy"] = val_worst_group_acc
                wb_log["Val/WG_avg10_accuracy"] = avg_10_val_worst_group
                wb_log["Val/WG_best_accuracy"] = best_val_worst_group_acc
            if has_groups and test_worst_group_acc is not None:
                wb_log["Test/WorstGroupAccuracy"] = test_worst_group_acc
            wandb.log(wb_log)
    
    end_time = time.time()
    total_time = end_time - start_time
    print()
    print("="*60)
    print(f"Training completed!")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Average over last 10 epochs val accuracy: {avg_10_val_acc:.2f}%")
    if has_groups:
        print(f"Average over last 10 epochs val worst-group accuracy: {avg_10_val_worst_group:.2f}%")
    print(f"Model saved to: {args.save_path}")
    print(f"Log saved to: {log_file}")
    print("="*60)
    
    # ---- W&B finish ----
    if args.wandb:
        wandb.finish()
    
    # Restore stdout
    sys.stdout = logger.terminal

if __name__ == "__main__":
    main()
