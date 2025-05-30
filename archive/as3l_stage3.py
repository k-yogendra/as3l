import argparse
import os
import random
import time
import shutil
import math
import numpy as np
from collections import Counter, defaultdict
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets, transforms

import wandb # Import wandb

# --- Set random seeds for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Wide ResNet Definitions (copied directly from simsiam_pretrain.py and adapted) ---
# This WideResNet will serve as the backbone for the final classifier.
# Its `fc` layer will now map to `num_classes` for classification.

class WideBasicBlock(nn.Module):
    """
    Wide Residual Network Basic Block.
    """
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equal_channels = (in_planes == out_planes)
        self.convShortcut = (not self.equal_channels) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                    padding=0, bias=False) or None

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = out
        if not self.equal_channels:
            shortcut = self.convShortcut(out)
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(shortcut, out)
        return out


class WideResNet(nn.Module):
    """
    Wide Residual Network (WRN) adapted for classification.
    WRN-28-2 means 28 layers deep and a widen_factor of 2.
    """
    def __init__(self, depth, widen_factor, num_classes, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k] # For WRN-28-2 (k=2): [16, 32, 64, 128]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(WideBasicBlock, nStages[1], n, stride=1, dropRate=dropRate)
        self.layer2 = self._make_layer(WideBasicBlock, nStages[2], n, stride=2, dropRate=dropRate)
        self.layer3 = self._make_layer(WideBasicBlock, nStages[3], n, stride=2, dropRate=dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        
        # This is the final classification head for Stage 3
        self.fc = nn.Linear(nStages[3], num_classes) # Maps features to class logits
        self.nChannels = nStages[3] # Feature dimension before FC layer (useful for other models)

        # Initialize layers (optional, as pre-trained weights will overwrite)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, dropRate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropRate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8) 
        feat = out.view(out.size(0), -1) # Feature vector
        logits = self.fc(feat) # Pass through the classification head
        return {'logits': logits, 'feat': feat}

def WRN_28_2_Classifier(num_classes=10):
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes)


# --- Data Augmentations (copied directly from main.py) ---
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

transform_weak = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_strong = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandAugment(), # Requires torchvision >= 0.9.0
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# --- Modified CIFAR10_SSL_Dataset for AS3L Stage 3 ---
class CIFAR10_SSL_Dataset_AS3L(Dataset):
    def __init__(self, base_dataset, indices, prior_pseudo_labels_all, transforms_w, transforms_s, is_labeled=False):
        self.base_dataset = base_dataset
        self.indices = indices 
        self.prior_pseudo_labels_all = prior_pseudo_labels_all 
        self.transforms_w = transforms_w
        self.transforms_s = transforms_s
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx_in_subset):
        original_idx = self.indices[idx_in_subset] 
        img, label_gt = self.base_dataset[original_idx] 

        if self.is_labeled:
            img_aug = self.transforms_w(img) if self.transforms_w else img
            return img_aug, label_gt, original_idx
        else:
            img_w = self.transforms_w(img) if self.transforms_w else img
            img_s = self.transforms_s(img) if self.transforms_s else img
            y_prior = self.prior_pseudo_labels_all[original_idx] 
            return img_w, img_s, original_idx, y_prior

# --- FlexMatchThresholdingHook (copied and fixed) ---
class FlexMatchThresholdingHook:
    def __init__(self, total_training_samples, num_classes, p_cutoff, thresh_warmup=True):
        self.total_training_samples = total_training_samples 
        self.num_classes = num_classes
        self.p_cutoff = p_cutoff
        self.thresh_warmup = thresh_warmup
        
        self.selected_label = torch.ones((self.total_training_samples,), dtype=torch.long) * -1 
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update_class_accuracy(self):
        selected_label_cpu = self.selected_label.cpu().tolist()
        pseudo_counter = Counter(selected_label_cpu)

        wo_negative_one = deepcopy(pseudo_counter)
        if -1 in wo_negative_one.keys():
            wo_negative_one.pop(-1)

        if len(wo_negative_one) > 0: 
            max_count = max(wo_negative_one.values())
            for i in range(self.num_classes):
                self.classwise_acc[i] = pseudo_counter[i] / max_count
        else: 
            self.classwise_acc.fill_(0.0)

    @torch.no_grad()
    def generate_mask(self, probs_x_ulb_w, idx_ulb_global, device):
        if self.selected_label.device != device:
            self.selected_label = self.selected_label.to(device)
        if self.classwise_acc.device != device:
            self.classwise_acc = self.classwise_acc.to(device)

        max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)

        threshold_per_sample = self.p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx]))
        mask = max_probs.ge(threshold_per_sample).float() 

        select_for_update = max_probs.ge(self.p_cutoff)
        
        if idx_ulb_global[select_for_update == 1].nelement() != 0:
            self.selected_label[idx_ulb_global[select_for_update == 1]] = max_idx[select_for_update == 1]
        
        self.update_class_accuracy()

        return mask

    def state_dict(self):
        return {
            'selected_label': self.selected_label.cpu(),
            'classwise_acc': self.classwise_acc.cpu(),
        }

    def load_state_dict(self, state_dict, device):
        self.selected_label = state_dict['selected_label'].to(device)
        self.classwise_acc = state_dict['classwise_acc'].to(device)


# --- FlexMatch Algorithm (copied and adapted) ---
class FlexMatchAlgorithm:
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 num_classes: int,
                 total_training_samples: int,
                 T: float = 0.5,
                 p_cutoff: float = 0.95,
                 hard_label: bool = True,
                 thresh_warmup: bool = True,
                 lambda_u: float = 1.0, 
                 use_amp: bool = False,
                 switching_epoch: int = 60 
                 ):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        
        self.T = T 
        self.p_cutoff = p_cutoff 
        self.hard_label = hard_label 
        self.thresh_warmup = thresh_warmup 
        self.lambda_u = lambda_u 

        self.switching_epoch = switching_epoch

        self.ce_loss = nn.CrossEntropyLoss(reduction='mean') 

        self.masking_hook = FlexMatchThresholdingHook(
            total_training_samples=total_training_samples,
            num_classes=num_classes,
            p_cutoff=p_cutoff,
            thresh_warmup=thresh_warmup
        )

        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def compute_prob(self, logits):
        return torch.softmax(logits / self.T, dim=-1)

    def consistency_loss(self, logits_s, pseudo_label, mask):
        logits_s_masked = logits_s[mask == 1]
        pseudo_label_masked = pseudo_label[mask == 1]

        if logits_s_masked.nelement() == 0:
            return torch.tensor(0.0, device=self.device)

        if self.hard_label:
            loss = F.cross_entropy(logits_s_masked, pseudo_label_masked, reduction='mean')
        else:
            log_probs_s = F.log_softmax(logits_s_masked, dim=-1)
            loss = F.kl_div(log_probs_s, pseudo_label_masked, reduction='batchmean')
        
        return loss

    def train_step(self, x_lb, y_lb, idx_ulb_global, x_ulb_w, x_ulb_s, current_epoch, y_prior_ulb_batch):
        self.model.train()
        self.optimizer.zero_grad()

        x_lb, y_lb = x_lb.to(self.device), y_lb.to(self.device)
        idx_ulb_global = idx_ulb_global.to(self.device)
        x_ulb_w, x_ulb_s = x_ulb_w.to(self.device), x_ulb_s.to(self.device)
        y_prior_ulb_batch = y_prior_ulb_batch.to(self.device)

        # Ensure y_prior_ulb_batch is long type for F.one_hot
        y_prior_ulb_batch = y_prior_ulb_batch.long()

        context_manager = torch.cuda.amp.autocast() if self.use_amp else torch.no_grad()

        with context_manager:
            # Labeled data
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']

            # Unlabeled strong augmentation
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']

            # Unlabeled weak augmentation (inference mode for pseudo-label generation)
            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
        
        # Calculate supervised loss
        sup_loss = self.ce_loss(logits_x_lb, y_lb)

        # --- Unsupervised Loss Calculation with AS3L PPL integration ---
        model_probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

        if current_epoch < self.switching_epoch: 
            y_prior_one_hot = F.one_hot(y_prior_ulb_batch, num_classes=self.num_classes).float()
            
            combined_logits = y_prior_one_hot + model_probs_x_ulb_w 
            y_post_soft = F.normalize(combined_logits, p=1, dim=-1) # L1-normalize to sum to 1
        else:
            y_post_soft = model_probs_x_ulb_w

        mask = self.masking_hook.generate_mask(model_probs_x_ulb_w, idx_ulb_global, self.device)
        
        if self.hard_label:
            pseudo_label_target = torch.argmax(y_post_soft, dim=-1)
        else:
            pseudo_label_target = y_post_soft

        unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label_target, mask)

        total_loss = sup_loss + self.lambda_u * unsup_loss

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        log_dict = {
            'sup_loss': sup_loss.item(),
            'unsup_loss': unsup_loss.item(),
            'total_loss': total_loss.item(),
            'util_ratio': mask.float().mean().item(),
            'classwise_acc': self.masking_hook.classwise_acc.mean().item()
        }

        return log_dict

    def eval_step(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x.to(self.device))
        return outputs['logits']
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'flexmatch_hook_state': self.masking_hook.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.masking_hook.load_state_dict(checkpoint['flexmatch_hook_state'], self.device)
        print(f"Loaded checkpoint from {path}")
        return checkpoint

# --- Function to prepare AS3L datasets ---
def get_as3l_ssl_datasets(data_dir, selected_labeled_indices, selected_labeled_labels, prior_pseudo_labels, total_training_samples):
    base_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    base_test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    assert total_training_samples == len(base_train_dataset)
    
    total_train_indices_list = list(range(total_training_samples))
    
    selected_labeled_indices_set = set(selected_labeled_indices.tolist())
    unlabeled_indices = np.array([idx for idx in total_train_indices_list if idx not in selected_labeled_indices_set], dtype=np.int64)

    assert len(selected_labeled_indices_set.intersection(unlabeled_indices)) == 0
    assert len(selected_labeled_indices) + len(unlabeled_indices) == len(base_train_dataset)
    
    labeled_dataset = CIFAR10_SSL_Dataset_AS3L(base_train_dataset, selected_labeled_indices, 
                                               prior_pseudo_labels_all=prior_pseudo_labels, 
                                               transforms_w=transform_weak, transforms_s=None, is_labeled=True)
    
    unlabeled_dataset = CIFAR10_SSL_Dataset_AS3L(base_train_dataset, unlabeled_indices, 
                                                 prior_pseudo_labels_all=prior_pseudo_labels, 
                                                 transforms_w=transform_weak, transforms_s=transform_strong, is_labeled=False)
    
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)

    print(f"AS3L Labeled Samples (selected): {len(labeled_dataset)}")
    print(f"AS3L Unlabeled Samples (remaining): {len(unlabeled_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")

    return labeled_dataset, unlabeled_dataset, test_dataset

# --- Main AS3L Stage 3 Training Function ---
def main_as3l_stage3():
    parser = argparse.ArgumentParser(description='AS3L Stage 3: Semi-Supervised Training Guided by Prior Pseudo-labels')
    parser.add_argument('--data_root', default='./data', type=str, help='Path to the root directory containing dataset.')
    parser.add_argument('--encoder_weights_path', default='./output/my_as3l_runs/simsiam_pretrain_cifar10_wrn282/wrn_28_2_encoder_fself.pth', type=str,
                        help='Path to the pre-trained SimSiam ENCODER weights (f_self) from Stage 1.')
    parser.add_argument('--stage2_outputs_dir', default='./output/my_as3l_runs/as3l_stage2_outputs', type=str,
                        help='Directory containing outputs from Stage 2 (selected labels, prior pseudo-labels).')
    parser.add_argument('--output_model_dir', default='./output/my_as3l_runs/as3l_stage3_models', type=str,
                        help='Directory to save the final trained AS3L model.')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in the dataset.')
    parser.add_argument('--total_training_samples', default=50000, type=int, 
                        help='Total number of training samples in the base dataset (e.g., 50000 for CIFAR-10).')
    
    # Model Architecture (changed to WRN_28_2_Classifier to match Stage 1 pre-training backbone)
    parser.add_argument('--model_arch', default='wrn_28_2', type=str, help='Model architecture for Stage 3 (must match Stage 1, e.g., wrn_28_2).')
    
    # FlexMatch Hyperparameters
    parser.add_argument('--flexmatch_T', default=0.7, type=float, help='Temperature for pseudo-label sharpening in FlexMatch.')
    parser.add_argument('--flexmatch_p_cutoff', default=0.95, type=float, help='Base confidence threshold for FlexMatch.')
    parser.add_argument('--flexmatch_lambda_u', default=7.0, type=float, help='Unsupervised loss weight (lambda_u/Mu in paper).')
    parser.add_argument('--flexmatch_hard_label', action='store_true', help='Use hard (one-hot) pseudo-labels for consistency loss.')
    parser.add_argument('--flexmatch_soft_label', dest='flexmatch_hard_label', action='store_false', help='Use soft pseudo-labels for consistency loss.')
    parser.set_defaults(flexmatch_hard_label=True) # Default to hard labels
    parser.add_argument('--flexmatch_thresh_warmup', action='store_true', help='Use threshold warm-up in FlexMatch.')
    parser.add_argument('--no_flexmatch_thresh_warmup', dest='flexmatch_thresh_warmup', action='store_false', help='Do NOT use threshold warm-up.')
    parser.set_defaults(flexmatch_thresh_warmup=True) # Default to True
    parser.add_argument('--as3l_switching_epoch', default=60, type=int, help='AS3L switching point T (in epochs).')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs.')
    parser.add_argument('--labeled_batch_size', default=64, type=int, help='Batch size for labeled data.')
    parser.add_argument('--unlabeled_batch_size_ratio', default=7, type=float, help='Ratio of unlabeled to labeled batch size (lambda_u for batching).') 
    parser.add_argument('--learning_rate', default=0.03, type=float, help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for regularization.')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision.')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='Do NOT use Automatic Mixed Precision.')
    parser.set_defaults(use_amp=True)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--print_freq', default=100, type=int, help='Frequency to print training progress (in steps).')
    parser.add_argument('--eval_freq_epochs', default=5, type=int, help='Frequency to evaluate on test set (in epochs).')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index to use. Set to -1 for CPU.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.') 
    # NEW ARGUMENT FOR WANDB NAME GENERATION:
    parser.add_argument('--num_labeled_samples_per_class', default=10, type=int, 
                        help='Number of labeled samples per class (needed for WandB name).') 
    
    args = parser.parse_args()
    
    set_seed(args.seed) # Set seed after parsing arguments, using args.seed

    # Device configuration
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(args.output_model_dir, exist_ok=True)

    # Initialize WandB run
    wandb.init(
        mode="online",
        project="AS3L_CIFAR10", # Your WandB project name
        entity="k-yogendra", # <--- REPLACE WITH YOUR WANDB USERNAME/TEAM NAME
        config=args, # Logs all argparse arguments
        name=f"AS3L_L{args.num_labeled_samples_per_class}_Mu{args.flexmatch_lambda_u}_T{args.as3l_switching_epoch}_Epochs{args.epochs}" # Custom run name
    )
    print(f"WandB run initialized: {wandb.run.url}")


    print("--- AS3L Stage 3: Semi-Supervised Training Guided by Prior Pseudo-labels ---")

    # 1. Load outputs from Stage 2
    print(f"Loading Stage 2 outputs from {args.stage2_outputs_dir}...")
    selected_labeled_indices = np.load(os.path.join(args.stage2_outputs_dir, "selected_labeled_indices.npy"))
    selected_labeled_labels = np.load(os.path.join(args.stage2_outputs_dir, "selected_labeled_labels.npy"))
    prior_pseudo_labels = np.load(os.path.join(args.stage2_outputs_dir, "prior_pseudo_labels.npy"))
    print(f"Loaded {len(selected_labeled_indices)} labeled samples and {len(prior_pseudo_labels)} prior pseudo-labels.")

    # 2. Prepare datasets and DataLoaders
    labeled_dataset, unlabeled_dataset, test_dataset = get_as3l_ssl_datasets(
        args.data_root, selected_labeled_indices, selected_labeled_labels, prior_pseudo_labels, args.total_training_samples)

    unlabeled_batch_size = int(args.labeled_batch_size * args.unlabeled_batch_size_ratio)

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    
    num_unlabeled_batches_per_epoch = len(unlabeled_loader)
    num_labeled_samples_to_draw = num_unlabeled_batches_per_epoch * args.labeled_batch_size

    labeled_sampler = RandomSampler(labeled_dataset, replacement=True, num_samples=num_labeled_samples_to_draw)
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.labeled_batch_size, sampler=labeled_sampler, drop_last=True, num_workers=args.num_workers) 
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # 3. Initialize WideResNet Classifier
    model = WRN_28_2_Classifier(num_classes=args.num_classes)
    model.to(device)

    # 4. Load f_self weights from Stage 1 into the WideResNet Classifier backbone
    print(f"Loading f_self weights from {args.encoder_weights_path} into WRN-28-2 backbone...")
    encoder_state_dict = torch.load(args.encoder_weights_path, map_location=device)
    
    model_state_dict = model.state_dict()
    loaded_keys = 0
    
    for key, value in encoder_state_dict.items():
        if key.startswith('backbone.'):
            new_key = key[len('backbone.'):]
            
            if new_key.startswith('fc'): 
                continue

            if new_key in model_state_dict and model_state_dict[new_key].shape == value.shape:
                model_state_dict[new_key] = value
                loaded_keys += 1
            
    model.load_state_dict(model_state_dict) 
    print(f"Loaded {loaded_keys} parameters from f_self backbone.")

    # 5. Setup Optimizer and Learning Rate Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    total_training_steps = num_unlabeled_batches_per_epoch * args.epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: math.cos(step * math.pi / (2 * total_training_steps)))

    # 6. Instantiate FlexMatchAlgorithm (now including AS3L PPL logic)
    flexmatch_algo = FlexMatchAlgorithm(
        model=model,
        optimizer=optimizer,
        device=device,
        num_classes=args.num_classes,
        total_training_samples=args.total_training_samples,
        T=args.flexmatch_T,
        p_cutoff=args.flexmatch_p_cutoff,
        hard_label=args.flexmatch_hard_label,
        thresh_warmup=args.flexmatch_thresh_warmup,
        lambda_u=args.flexmatch_lambda_u,
        use_amp=args.use_amp,
        switching_epoch=args.as3l_switching_epoch
    )

    print("Starting AS3L Stage 3 training...")
    global_step = 0
    best_test_acc = 0.0

    for epoch in range(args.epochs):
        flexmatch_algo.model.train()
        
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        total_loss_epoch = 0
        total_sup_loss_epoch = 0
        total_unsup_loss_epoch = 0
        total_util_ratio = 0
        num_batches = 0

        for batch_idx in range(len(unlabeled_loader)):
            try:
                x_lb, y_lb, _ = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_lb, y_lb, _ = next(labeled_iter)
            
            x_ulb_w, x_ulb_s, idx_ulb_global, y_prior_ulb_batch = next(unlabeled_iter)

            log_dict = flexmatch_algo.train_step(
                x_lb, y_lb, idx_ulb_global, x_ulb_w, x_ulb_s, epoch, y_prior_ulb_batch
            )
            
            scheduler.step() 

            total_loss_epoch += log_dict['total_loss']
            total_sup_loss_epoch += log_dict['sup_loss']
            total_unsup_loss_epoch += log_dict['unsup_loss']
            total_util_ratio += log_dict['util_ratio']
            num_batches += 1
            global_step += 1

            if global_step % args.print_freq == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {global_step}: "
                      f"Total Loss = {log_dict['total_loss']:.4f}, "
                      f"Sup Loss = {log_dict['sup_loss']:.4f}, "
                      f"Unsup Loss = {log_dict['unsup_loss']:.4f}, "
                      f"Util Ratio = {log_dict['util_ratio']:.4f}, "
                      f"Classwise Acc (avg) = {log_dict['classwise_acc']:.4f}, "
                      f"LR = {scheduler.get_last_lr()[0]:.6f}")
                # Log to WandB at each print_freq step
                wandb.log({
                    "train/total_loss": log_dict['total_loss'],
                    "train/sup_loss": log_dict['sup_loss'],
                    "train/unsup_loss": log_dict['unsup_loss'],
                    "train/util_ratio": log_dict['util_ratio'],
                    "train/classwise_acc_avg": log_dict['classwise_acc'],
                    "train/learning_rate": scheduler.get_last_lr()[0]
                }, step=global_step)


        avg_total_loss = total_loss_epoch / num_batches
        avg_sup_loss = total_sup_loss_epoch / num_batches
        avg_unsup_loss = total_unsup_loss_epoch / num_batches
        avg_util_ratio = total_util_ratio / num_batches
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Avg Total Loss = {avg_total_loss:.4f}, "
              f"Avg Sup Loss = {avg_sup_loss:.4f}, "
              f"Avg Unsup Loss = {avg_unsup_loss:.4f}, "
              f"Avg Util Ratio = {avg_util_ratio:.4f}, "
              f"Current Classwise Acc (avg) = {flexmatch_algo.masking_hook.classwise_acc.mean().item():.4f}")
        # Log epoch summary to WandB
        wandb.log({
            "epoch_summary/avg_total_loss": avg_total_loss,
            "epoch_summary/avg_sup_loss": avg_sup_loss,
            "epoch_summary/avg_unsup_loss": avg_unsup_loss,
            "epoch_summary/avg_util_ratio": avg_util_ratio,
            "epoch_summary/current_classwise_acc_avg": flexmatch_algo.masking_hook.classwise_acc.mean().item()
        }, step=global_step) 

        # --- Evaluation ---
        if (epoch + 1) % args.eval_freq_epochs == 0:
            flexmatch_algo.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    logits = flexmatch_algo.eval_step(x_test)
                    _, predicted = torch.max(logits.data, 1)
                    total += y_test.size(0)
                    correct += (predicted == y_test.to(device)).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Test Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")
            # Log test accuracy to WandB
            wandb.log({"test/accuracy": accuracy}, step=global_step) 

            if accuracy > best_test_acc:
                best_test_acc = accuracy
                checkpoint_path = os.path.join(args.output_model_dir, "as3l_best_model.pth")
                flexmatch_algo.save_checkpoint(checkpoint_path)
                print(f"New best model saved to {checkpoint_path} with accuracy: {best_test_acc:.2f}%")
                # Also save the best model to WandB
                wandb.save(checkpoint_path)
        
    print(f"Training finished. Best Test Accuracy: {best_test_acc:.2f}%")
    print("--- AS3L Stage 3 Completed Successfully ---")

    wandb.finish() # End the WandB run


if __name__ == "__main__":
    main_as3l_stage3()

# This script is designed to be run as a standalone module, use following commands to run the script.
# python as3l_stage3.py \
#     --data_root ./data \
#     --encoder_weights_path ./output/my_as3l_runs/simsiam_pretrain_cifar10_wrn282/wrn_28_2_encoder_fself.pth \
#     --stage2_outputs_dir ./output/my_as3l_runs/as3l_stage2_outputs \
#     --output_model_dir ./output/my_as3l_runs/as3l_stage3_models \
#     --num_classes 10 \
#     --total_training_samples 50000 \
#     --model_arch wrn_28_2 \
#     --flexmatch_lambda_u 7.0 \
#     --as3l_switching_epoch 60 \
#     --epochs 200 \
#     --labeled_batch_size 64 \
#     --unlabeled_batch_size_ratio 7 \
#     --learning_rate 0.03 \
#     --weight_decay 5e-4 \
#     --gpu 0 \
#     --num_workers 4 \
#     --use_amp # Or --no_amp if you prefer not to use mixed precision