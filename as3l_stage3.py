# python as3l_stage3.py \
    # --data_root ./data \
    # --fself_weights_path ./output/my_as3l_runs/simsiam_pretrain_cifar10/resnet18_backbone_fself.pth \
    # --stage2_outputs_dir ./output/my_as3l_runs/as3l_stage2_outputs \
    # --output_model_dir ./output/my_as3l_runs/as3l_stage3_models \
    # --num_classes 10 \
    # --model_arch resnet18 \
    # --flexmatch_T 0.7 \
    # --flexmatch_p_cutoff 0.95 \
    # --flexmatch_lambda_u 7.0 \
    # --flexmatch_hard_label True \
    # --flexmatch_thresh_warmup True \
    # --as3l_switching_epoch 60 \
    # --epochs 200 \
    # --labeled_batch_size 64 \
    # --unlabeled_batch_size_ratio 7 \
    # --learning_rate 0.03 \
    # --momentum 0.9 \
    # --weight_decay 5e-4 \
    # --use_amp \
    # --num_workers 4 \
    # --print_freq 100 \
    # --eval_freq_epochs 5 \
    # --gpu 0

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

# --- Set random seeds for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Wide ResNet / ResNet Definitions (copied directly from previous files) ---
# BasicBlock and Bottleneck from original SimSiam pretrain script
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet for classification (will be used in Stage 3, to load SimSiam pre-trained weights)
# Modified to return dict like WideResNet for FlexMatch compatibility
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.nChannels = 512 * block.expansion # Feature dimension before FC layer

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feat = out.view(-1, self.nChannels) # Feature vector
        logits = self.fc(feat)
        return {'logits': logits, 'feat': feat}

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

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
        self.indices = indices # These are the original global indices from the full CIFAR-10 training set (0 to 49999)
        self.prior_pseudo_labels_all = prior_pseudo_labels_all # The full array of PPL, indexed by original_idx
        self.transforms_w = transforms_w
        self.transforms_s = transforms_s
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx_in_subset):
        original_idx = self.indices[idx_in_subset] # This is the global index for the sample
        img, label_gt = self.base_dataset[original_idx] # label_gt is the ground truth

        if self.is_labeled:
            # For labeled data: img, label (ground truth), and original_idx
            img_aug = self.transforms_w(img) if self.transforms_w else img
            return img_aug, label_gt, original_idx
        else:
            # For unlabeled data: weak_aug_img, strong_aug_img, original_idx, prior_pseudo_label
            img_w = self.transforms_w(img) if self.transforms_w else img
            img_s = self.transforms_s(img) if self.transforms_s else img
            y_prior = self.prior_pseudo_labels_all[original_idx] # Get PPL using global index
            return img_w, img_s, original_idx, y_prior

# --- FlexMatchThresholdingHook (copied and fixed) ---
class FlexMatchThresholdingHook:
    def __init__(self, total_training_samples, num_classes, p_cutoff, thresh_warmup=True):
        self.total_training_samples = total_training_samples # Total samples in the base training dataset
        self.num_classes = num_classes
        self.p_cutoff = p_cutoff
        self.thresh_warmup = thresh_warmup
        
        # Initialize selected_label to track pseudo-labels for ALL training samples (50000 for CIFAR-10)
        self.selected_label = torch.ones((self.total_training_samples,), dtype=torch.long) * -1 
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update_class_accuracy(self):
        """
        Updates the class-wise accuracy based on selected pseudo-labels.
        """
        # Ensure selected_label is on CPU for Counter, then move back to device if needed later
        selected_label_cpu = self.selected_label.cpu().tolist()
        pseudo_counter = Counter(selected_label_cpu)

        # Remove -1 from the counter, as it means "unselected"
        wo_negative_one = deepcopy(pseudo_counter)
        if -1 in wo_negative_one.keys():
            wo_negative_one.pop(-1)

        if len(wo_negative_one) > 0: # Check if any positive labels have been selected
            max_count = max(wo_negative_one.values())
            for i in range(self.num_classes):
                if self.thresh_warmup:
                    self.classwise_acc[i] = pseudo_counter[i] / max_count
                else:
                    self.classwise_acc[i] = pseudo_counter[i] / max_count
        else: # If no positive labels yet, all classwise_acc remain 0.0
            self.classwise_acc.fill_(0.0)

    @torch.no_grad()
    def generate_mask(self, probs_x_ulb_w, idx_ulb_global, device):
        """
        Generates the mask for unlabeled data based on adaptive thresholding.
        Args:
            probs_x_ulb_w (torch.Tensor): Probabilities from unlabeled weak-augmented data (model prediction).
            idx_ulb_global (torch.Tensor): Original global indices of the unlabeled batch samples (0 to 49999).
            device (torch.device): The device (e.g., 'cuda', 'cpu') to perform operations on.
        Returns:
            torch.Tensor: The binary mask.
        """
        # Ensure selected_label and classwise_acc are on the correct device for tensor operations
        # Move selected_label to device only if not already there, to avoid repeated moves
        if self.selected_label.device != device:
            self.selected_label = self.selected_label.to(device)
        if self.classwise_acc.device != device:
            self.classwise_acc = self.classwise_acc.to(device)

        max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)

        # The core FlexMatch adaptive thresholding formula (convex function)
        # `max_idx` contains class IDs for the current batch, so `classwise_acc[max_idx]` is valid.
        threshold_per_sample = self.p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx]))
        mask = max_probs.ge(threshold_per_sample).float() # Mask for the current batch

        # Update selected_label based on base p_cutoff for classwise_acc update
        # Only samples passing the *base* p_cutoff are considered for `selected_label`
        select_for_update = max_probs.ge(self.p_cutoff)
        
        if idx_ulb_global[select_for_update == 1].nelement() != 0:
            # Update the global `selected_label` array at the global indices
            self.selected_label[idx_ulb_global[select_for_update == 1]] = max_idx[select_for_update == 1]
        
        # Update class-wise accuracy after potentially adding new selected labels
        self.update_class_accuracy()

        return mask

    def state_dict(self):
        return {
            'selected_label': self.selected_label.cpu(),
            'classwise_acc': self.classwise_acc.cpu(),
        }

    def load_state_dict(self, state_dict, device):
        # Resize selected_label if checkpoint was saved with a different size
        # (e.g., if total_training_samples changed, but it shouldn't here)
        if self.selected_label.shape != state_dict['selected_label'].shape:
            print(f"Warning: Resizing selected_label from {state_dict['selected_label'].shape} to {self.selected_label.shape}")
            self.selected_label = torch.ones((self.total_training_samples,), dtype=torch.long) * -1 # Re-init
            # Only copy over valid indices if resizing happened, otherwise just load.
            # For this scenario, assuming same total_training_samples always.
        self.selected_label = state_dict['selected_label'].to(device)
        self.classwise_acc = state_dict['classwise_acc'].to(device)


# --- FlexMatch Algorithm (copied and adapted) ---
class FlexMatchAlgorithm:
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 num_classes: int,
                 total_training_samples: int, # Pass total samples for hook initialization
                 T: float = 0.5,
                 p_cutoff: float = 0.95,
                 hard_label: bool = True,
                 thresh_warmup: bool = True,
                 lambda_u: float = 1.0, # Unsupervised loss weight
                 use_amp: bool = False,
                 switching_epoch: int = 60 # AS3L switching point T
                 ):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        
        # FlexMatch hyperparameters
        self.T = T # Temperature for pseudo-label sharpening
        self.p_cutoff = p_cutoff # Base confidence threshold
        self.hard_label = hard_label # Use hard (one-hot) or soft pseudo-labels
        self.thresh_warmup = thresh_warmup # Warmup for adaptive thresholding
        self.lambda_u = lambda_u # Unsupervised loss weight

        # AS3L specific
        self.switching_epoch = switching_epoch

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean') # Supervised CE loss

        # Initialize FlexMatchThresholdingHook with total training samples
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
        idx_ulb_global = idx_ulb_global.to(self.device) # Ensure global indices are on device
        x_ulb_w, x_ulb_s = x_ulb_w.to(self.device), x_ulb_s.to(self.device)
        y_prior_ulb_batch = y_prior_ulb_batch.to(self.device)

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

        # Determine y_post (pseudo-label target) based on switching epoch (T)
        if current_epoch < self.switching_epoch: # Use < to include epoch 0 to switching_epoch-1
            y_prior_one_hot = F.one_hot(y_prior_ulb_batch.long(), num_classes=self.num_classes).float()
            y_prior_one_hot = y_prior_one_hot.to(self.device)

            # Combine y_prior and model's weak predictions as per AS3L paper Eq. (5)
            # y_post = normalize(yprior + ypre,w)
            combined_logits = y_prior_one_hot + model_probs_x_ulb_w 
            y_post_soft = F.normalize(combined_logits, p=1, dim=-1) # L1-normalize to sum to 1
        else:
            y_post_soft = model_probs_x_ulb_w

        # Compute mask using FlexMatchThresholdingHook.
        mask = self.masking_hook.generate_mask(model_probs_x_ulb_w, idx_ulb_global, self.device)
        
        # Generate pseudo-labels (actual target for CE or KL loss) from y_post_soft
        if self.hard_label:
            pseudo_label_target = torch.argmax(y_post_soft, dim=-1)
        else:
            pseudo_label_target = y_post_soft

        # Calculate consistency loss
        unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label_target, mask)

        # Total loss
        total_loss = sup_loss + self.lambda_u * unsup_loss

        # Backpropagation
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
        # Pass device to hook's load_state_dict so tensors are on correct device
        self.masking_hook.load_state_dict(checkpoint['flexmatch_hook_state'], self.device)
        print(f"Loaded checkpoint from {path}")
        return checkpoint

# --- Function to prepare AS3L datasets ---
def get_as3l_ssl_datasets(data_dir, selected_labeled_indices, selected_labeled_labels, prior_pseudo_labels, total_training_samples):
    base_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    base_test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    # Ensure total_training_samples aligns with the loaded dataset
    assert total_training_samples == len(base_train_dataset)
    
    total_train_indices_list = list(range(total_training_samples))
    
    selected_labeled_indices_set = set(selected_labeled_indices.tolist())
    unlabeled_indices = np.array([idx for idx in total_train_indices_list if idx not in selected_labeled_indices_set], dtype=np.int64)

    # Sanity check: Ensure no overlap and sum of lengths is correct
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
    parser.add_argument('--fself_weights_path', default='./my_as3l_runs/simsiam_pretrain_cifar10/resnet18_backbone_fself.pth', type=str,
                        help='Path to the pre-trained f_self backbone weights from Stage 1.')
    parser.add_argument('--stage2_outputs_dir', default='./my_as3l_runs/as3l_stage2_outputs', type=str,
                        help='Directory containing outputs from Stage 2 (selected labels, prior pseudo-labels).')
    parser.add_argument('--output_model_dir', default='./my_as3l_runs/as3l_stage3_models', type=str,
                        help='Directory to save the final trained AS3L model.')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in the dataset.')
    parser.add_argument('--total_training_samples', default=50000, type=int, 
                        help='Total number of training samples in the base dataset (e.g., 50000 for CIFAR-10).')
    
    # Model Architecture (changed to ResNet18 to match Stage 1)
    parser.add_argument('--model_arch', default='resnet18', type=str, help='Model architecture for Stage 3 (must match Stage 1 for f_self loading).')
    
    # FlexMatch Hyperparameters
    parser.add_argument('--flexmatch_T', default=0.7, type=float, help='Temperature for pseudo-label sharpening in FlexMatch.')
    parser.add_argument('--flexmatch_p_cutoff', default=0.95, type=float, help='Base confidence threshold for FlexMatch.')
    parser.add_argument('--flexmatch_lambda_u', default=7.0, type=float, help='Unsupervised loss weight (lambda_u/Mu in paper).')
    parser.add_argument('--flexmatch_hard_label', default=True, type=bool, help='Use hard (one-hot) or soft pseudo-labels for consistency loss.')
    parser.add_argument('--flexmatch_thresh_warmup', default=True, type=bool, help='Use threshold warm-up in FlexMatch.')
    parser.add_argument('--as3l_switching_epoch', default=60, type=int, help='AS3L switching point T (in epochs).')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs.')
    parser.add_argument('--labeled_batch_size', default=64, type=int, help='Batch size for labeled data.')
    parser.add_argument('--unlabeled_batch_size_ratio', default=7, type=float, help='Ratio of unlabeled to labeled batch size (lambda_u for batching).') 
    parser.add_argument('--learning_rate', default=0.03, type=float, help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for regularization.')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision.') # Changed to action='store_true' for boolean flags
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='Do NOT use Automatic Mixed Precision.')
    parser.set_defaults(use_amp=True) # Default to True unless --no_amp is present
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--print_freq', default=100, type=int, help='Frequency to print training progress (in steps).')
    parser.add_argument('--eval_freq_epochs', default=5, type=int, help='Frequency to evaluate on test set (in epochs).')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index to use. Set to -1 for CPU.') # Changed default to -1 for clearer CPU intent

    args = parser.parse_args()

    # Device configuration
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(args.output_model_dir, exist_ok=True)

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

    # 3. Initialize ResNet18 model (matching Stage 1 architecture)
    model = ResNet18(num_classes=args.num_classes)
    model.to(device)

    # 4. Load f_self weights from Stage 1 into ResNet18 backbone (excluding FC layer)
    print(f"Loading f_self weights from {args.fself_weights_path} into ResNet18 backbone...")
    fself_state_dict = torch.load(args.fself_weights_path, map_location=device)
    
    model_state_dict = model.state_dict()
    loaded_keys = 0
    # Iterate through f_self weights and load only matching backbone layers
    for key, value in fself_state_dict.items():
        # Exclude the final 'fc' layer (SimSiam's ResNet has an 'fc' before its projection head)
        if 'fc' not in key and key in model_state_dict and model_state_dict[key].shape == value.shape:
            model_state_dict[key] = value
            loaded_keys += 1
        # else: # Optional: print keys that are skipped for debugging
        #     print(f"Skipping key {key} from f_self: not a backbone layer or shape mismatch.")
            
    model.load_state_dict(model_state_dict) # Load modified state dict
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
        total_training_samples=args.total_training_samples, # Pass total samples
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
        
        # Reset iterators for each epoch
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        total_loss_epoch = 0
        total_sup_loss_epoch = 0
        total_unsup_loss_epoch = 0
        total_util_ratio = 0
        num_batches = 0

        for batch_idx in range(len(unlabeled_loader)):
            try:
                x_lb, y_lb, _ = next(labeled_iter) # _ for original_idx
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_lb, y_lb, _ = next(labeled_iter)
            
            # idx_ulb is original_idx (global index)
            x_ulb_w, x_ulb_s, idx_ulb_global, y_prior_ulb_batch = next(unlabeled_iter)

            log_dict = flexmatch_algo.train_step(
                x_lb, y_lb, idx_ulb_global, x_ulb_w, x_ulb_s, epoch, y_prior_ulb_batch
            )
            
            # The warning "Detected call of `lr_scheduler.step()` before `optimizer.step()`" is common.
            # For cosine annealing with step-wise updates, it's often placed after optimizer.step().
            # If `warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`" ...)`
            # is annoying, you can safely ignore it or switch the order if it fits your scheduler.
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

            if accuracy > best_test_acc:
                best_test_acc = accuracy
                checkpoint_path = os.path.join(args.output_model_dir, "as3l_best_model.pth")
                flexmatch_algo.save_checkpoint(checkpoint_path)
                print(f"New best model saved to {checkpoint_path} with accuracy: {best_test_acc:.2f}%")
        
    print(f"Training finished. Best Test Accuracy: {best_test_acc:.2f}%")
    print("--- AS3L Stage 3 Completed Successfully ---")

if __name__ == "__main__":
    main_as3l_stage3()