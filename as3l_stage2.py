import argparse
import os
import random
import time
import shutil
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm # For progress bars

# --- 1. Wide ResNet (WRN) Backbone Implementation (Copied from simsiam_pretrain.py) ---
# Reference: https://github.com/meliketoy/wide-resnet.pytorch/blob/master/models/wideresnet.py

class WideBasicBlock(nn.Module):
    """
    Wide Residual Network Basic Block.
    Differs from standard BasicBlock by increasing the number of filters by a widen_factor.
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
        out = self.relu1(self.bn1(x)) # Apply BN1 and ReLU1 to the input 'x'
        
        shortcut = out # Default shortcut is activated input after first BN/ReLU
        if not self.equal_channels:
            shortcut = self.convShortcut(out) # If channels change, apply conv shortcut to activated input
        
        out = self.relu2(self.bn2(self.conv1(out))) 
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
            
        out = self.conv2(out)
        
        out = torch.add(shortcut, out)
        
        return out


class WideResNet(nn.Module):
    """
    Wide Residual Network (WRN) as used in the paper.
    WRN-28-2 means 28 layers deep and a widen_factor of 2.
    """
    def __init__(self, depth, widen_factor, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k] # For WRN-28-2 (k=2): [16, 32, 64, 128]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(WideBasicBlock, nStages[1], n, stride=1, dropRate=dropRate)
        self.layer2 = self._make_layer(WideBasicBlock, nStages[2], n, stride=2, dropRate=dropRate)
        self.layer3 = self._make_layer(WideBasicBlock, nStages[3], n, stride=2, dropRate=dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)

        # The final FC layer of the backbone. For SimSiam, its in_features is important
        self.fc = nn.Linear(nStages[3], nStages[3]) 

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
        out = out.view(out.size(0), -1)
        out = self.fc(out) # Pass through the final FC layer
        return out

def WRN_28_2_backbone_raw(): # Renamed to indicate it's the raw backbone before SimSiam's Identity replacement
    """Returns a Wide ResNet 28-2 model."""
    return WideResNet(depth=28, widen_factor=2)

# --- 2. SimSiam Specific Modules (Copied from simsiam_pretrain.py) ---
# These are needed to reconstruct the encoder

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim 
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

class prediction_MLP(nn.Module): # Not directly used but included for completeness if needed later
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# --- Helper function to get the full encoder ---
def get_simsiam_encoder(arch_name, backbone_feature_dim, feat_dim, num_proj_layers):
    """
    Constructs the SimSiam encoder (backbone + projector).
    """
    if arch_name == 'wrn_28_2':
        backbone = WRN_28_2_backbone_raw()
        # In SimSiam, the backbone's final FC layer is replaced by Identity *before* the projector.
        backbone.fc = nn.Identity() # Apply this modification here
    else:
        raise ValueError(f"Unsupported backbone architecture: {arch_name}")
    
    projector = projection_MLP(backbone_feature_dim, feat_dim, num_proj_layers)
    encoder = nn.Sequential(backbone, projector)
    return encoder


# --- Custom Dataset to return index ---
class CIFAR10WithIndex(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

# --- Helper classes (AverageMeter, ProgressMeter) ---
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'; return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches); self.meters = meters; self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]; entries += [str(meter) for meter in self.meters]; print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)); fmt = '{:' + str(num_digits) + 'd}'; return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# --- Main AS3L Stage 2 Class ---
class AS3LStage2:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu is not None else 'cpu')
        self.set_seed(args.seed)

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # 1. Load SimSiam Encoder (backbone + projector)
        print(f"Loading pre-trained encoder from {args.encoder_input_path}...")
        self.encoder = get_simsiam_encoder(args.arch, args.backbone_feature_dim, args.feat_dim, args.num_proj_layers)
        
        loaded_state_dict = torch.load(args.encoder_input_path, map_location=self.device)
        self.encoder.load_state_dict(loaded_state_dict)
        
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval() # Set to eval mode for feature extraction (f_self extraction)

        # Data transformations for feature extraction (no augmentation)
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 mean/std
        ])

        # Load CIFAR-10 dataset (training split contains all 50,000 images)
        self.cifar_train_dataset = CIFAR10WithIndex(root=args.data_root, train=True, download=True, transform=self.eval_transform)
        self.cifar_train_loader = DataLoader(self.cifar_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        print(f"Dataset loaded. Total samples: {len(self.cifar_train_dataset)}")


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # For reproducibility

    @torch.no_grad()
    def extract_features(self, model, data_loader, feature_dim):
        """Extracts features (f_self or f_fine) for all samples."""
        features = np.zeros((len(data_loader.dataset), feature_dim), dtype=np.float32)
        labels = np.zeros(len(data_loader.dataset), dtype=np.int64)
        indices = np.zeros(len(data_loader.dataset), dtype=np.int64)

        print(f"Extracting features using {model.__class__.__name__}...")
        for batch_idx, (imgs, lbls, idxs) in enumerate(tqdm(data_loader, desc="Feature Extraction")):
            imgs = imgs.to(self.device)
            feats = model(imgs) # This is now the output of the full encoder
            feats = F.normalize(feats, dim=1) # L2 normalize
            features[idxs.numpy()] = feats.cpu().numpy()
            labels[idxs.numpy()] = lbls.cpu().numpy()
            indices[idxs.numpy()] = idxs.cpu().numpy()
        print("Feature extraction complete.")
        return features, labels, indices

    def finetune_features(self, f_self_features, ground_truth_labels, total_indices):
        """
        Trains a linear layer on f_self features using ground truth labels to make f_fine discriminative.
        The output of this linear classifier serves as f_fine features.
        """
        print("Fine-tuning features (f_fine linear layer)...")
        in_dim = f_self_features.shape[1] # This will now be args.feat_dim (2048)
        out_dim = self.args.num_classes # f_fine features will be logits for classification

        linear_classifier = nn.Linear(in_dim, out_dim).to(self.device)
        optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=self.args.finetune_lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        class FeatureDataset(Dataset):
            def __init__(self, features, labels):
                self.features = features
                self.labels = labels
            def __len__(self):
                return len(self.features)
            def __getitem__(self, idx):
                feature = torch.from_numpy(self.features[idx]).float()
                label = self.labels[idx]
                return feature, label

        finetune_dataset = FeatureDataset(f_self_features, ground_truth_labels)
        finetune_loader = DataLoader(finetune_dataset, batch_size=self.args.finetune_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

        linear_classifier.train()
        for epoch in range(self.args.finetune_epochs):
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            progress = ProgressMeter(len(finetune_loader), [losses, top1], prefix=f"Fine-tune Epoch [{epoch+1}/{self.args.finetune_epochs}]")

            for i, (features, labels) in enumerate(finetune_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                output = linear_classifier(features)
                loss = criterion(output, labels)

                acc1 = self.accuracy(output, labels, topk=(1,))[0].item()
                
                losses.update(loss.item(), features.size(0))
                top1.update(acc1, features.size(0)) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % self.args.print_freq == 0:
                    progress.display(i)
            print(f"Fine-tune Epoch {epoch+1} finished. Avg Loss: {losses.avg:.4f}, Top1 Acc: {top1.avg:.2f}")
        
        # After fine-tuning, the linear_classifier itself *is* the f_fine transformation.
        # We need to extract the f_fine features for ALL samples using this trained linear layer.
        
        f_self_dataset_for_ff_extraction = FeatureDataset(f_self_features, ground_truth_labels) 
        loader_for_ff_extraction = DataLoader(f_self_dataset_for_ff_extraction, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        # Extract f_fine features
        f_fine_features_out = np.zeros((len(f_self_features), out_dim), dtype=np.float32) 
        linear_classifier.eval()
        with torch.no_grad():
            start_idx = 0
            for batch_idx, (feats_batch, _) in enumerate(tqdm(loader_for_ff_extraction, desc="Extracting f_fine features")):
                feats_batch = feats_batch.to(self.device)
                f_fine_batch = linear_classifier(feats_batch) 
                
                current_batch_size = f_fine_batch.size(0)
                f_fine_features_out[start_idx : start_idx + current_batch_size] = f_fine_batch.cpu().numpy()
                start_idx += current_batch_size
        
        print("Fine-tuning features complete.")
        return f_fine_features_out


    def select_labeled_samples(self, f_fine_features, ground_truth_labels, total_indices):
        """
        Actively selects a subset of samples for labeling.
        Performs K-means on f_fine and selects samples closest to inferred class centroids,
        ensuring `num_labeled_samples_per_class` for each ground truth class.
        """
        print(f"Actively selecting {self.args.num_labeled_samples_per_class} labeled samples per class...")
        
        n_clusters = self.args.num_classes 

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed, n_init='auto')
        cluster_assignments = kmeans.fit_predict(f_fine_features)
        cluster_centers = kmeans.cluster_centers_

        final_selected_indices = []
        final_selected_labels = []
        
        samples_by_gt_label = defaultdict(list) 

        for i, original_idx in enumerate(total_indices):
            gt_label = ground_truth_labels[original_idx]
            assigned_cluster_id = cluster_assignments[i]
            
            dist = np.linalg.norm(f_fine_features[i] - cluster_centers[assigned_cluster_id])
            
            samples_by_gt_label[gt_label].append((dist, original_idx, assigned_cluster_id))

        selected_indices_set = set()

        for gt_label in range(self.args.num_classes):
            candidates_for_this_class = sorted(samples_by_gt_label[gt_label])
            
            count_for_this_class = 0
            for dist, original_idx, assigned_cluster_id in candidates_for_this_class:
                if count_for_this_class < self.args.num_labeled_samples_per_class and original_idx not in selected_indices_set:
                    final_selected_indices.append(original_idx)
                    final_selected_labels.append(gt_label)
                    selected_indices_set.add(original_idx)
                    count_for_this_class += 1
                
                if count_for_this_class >= self.args.num_labeled_samples_per_class:
                    break

        selected_labeled_indices = np.array(final_selected_indices, dtype=np.int64) 
        selected_labeled_labels = np.array(final_selected_labels, dtype=np.int64)   

        print(f"Selected {len(selected_labeled_indices)} labeled samples.")
        print(f"Selected samples per class distribution: {np.bincount(selected_labeled_labels, minlength=self.args.num_classes)}")

        return selected_labeled_indices, selected_labeled_labels


    def generate_prior_pseudo_labels(self, f_fine_features, selected_labeled_indices, selected_labeled_labels, total_indices):
        """
        Generates Prior Pseudo-Labels (y_prior) for all unlabeled samples.
        Uses Constrained Seed K-means interpretation: K-means with majority vote propagation from labeled samples.
        """
        print("Generating Prior Pseudo-Labels (y_prior)...")

        num_total_samples = len(total_indices)
        y_prior_votes = np.zeros((num_total_samples, self.args.num_classes), dtype=np.int32)

        selected_labeled_map = {idx: label for idx, label in zip(selected_labeled_indices, selected_labeled_labels)}

        for run_idx in tqdm(range(self.args.num_clustering_runs_C), desc="Clustering for PPL"):
            kmeans = KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed + run_idx, n_init='auto')
            cluster_assignments = kmeans.fit_predict(f_fine_features)
            
            cluster_label_votes = defaultdict(lambda: defaultdict(int))
            
            for i, cluster_id in enumerate(cluster_assignments):
                original_idx = total_indices[i] 
                if original_idx in selected_labeled_map: 
                    gt_label = selected_labeled_map[original_idx] 
                    cluster_label_votes[cluster_id][gt_label] += 1
            
            cluster_dominant_labels = {}
            for cluster_id in range(self.args.num_classes):
                if cluster_id in cluster_label_votes and cluster_label_votes[cluster_id]:
                    dominant_label = max(cluster_label_votes[cluster_id], key=cluster_label_votes[cluster_id].get)
                    cluster_dominant_labels[cluster_id] = dominant_label
                else:
                    cluster_dominant_labels[cluster_id] = random.randint(0, self.args.num_classes - 1)
            
            for i, cluster_id in enumerate(cluster_assignments):
                original_idx = total_indices[i] 
                y_prior_votes[original_idx, cluster_dominant_labels[cluster_id]] += 1

        y_prior = np.argmax(y_prior_votes, axis=1)

        print("Prior Pseudo-Labels generation complete.")
        return y_prior


    def run(self):
        """Orchestrates the entire AS3L Stage 2 process."""
        print("--- AS3L Stage 2: Active Learning & Prior Pseudo-labels ---")

        # 1. Extract f_self features (output of the SimSiam encoder) for all training data
        # The feature_dim is now args.feat_dim (2048)
        f_self_features, ground_truth_labels, total_indices = self.extract_features(self.encoder, self.cifar_train_loader, self.args.feat_dim)
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        np.save(os.path.join(self.args.output_dir, "f_self_features.npy"), f_self_features)
        np.save(os.path.join(self.args.output_dir, "ground_truth_labels.npy"), ground_truth_labels)
        np.save(os.path.join(self.args.output_dir, "total_indices.npy"), total_indices)
        print(f"Saved f_self features, ground truth labels, and indices to {self.args.output_dir}")

        # 2. Fine-tune features (linear layer from f_self to f_fine)
        f_fine_features = self.finetune_features(f_self_features, ground_truth_labels, total_indices)
        np.save(os.path.join(self.args.output_dir, "f_fine_features.npy"), f_fine_features)
        print(f"Saved f_fine features to {self.args.output_dir}")

        # 3. Active Labeled Sample Selection
        selected_labeled_indices, selected_labeled_labels = self.select_labeled_samples(f_fine_features, ground_truth_labels, total_indices)
        
        np.save(os.path.join(self.args.output_dir, "selected_labeled_indices.npy"), selected_labeled_indices)
        np.save(os.path.join(self.args.output_dir, "selected_labeled_labels.npy"), selected_labeled_labels)
        print(f"Saved selected labeled samples to {self.args.output_dir}")

        # 4. Prior Pseudo-label Generation
        prior_pseudo_labels = self.generate_prior_pseudo_labels(f_fine_features, selected_labeled_indices, selected_labeled_labels, total_indices)
        np.save(os.path.join(self.args.output_dir, "prior_pseudo_labels.npy"), prior_pseudo_labels)
        print(f"Saved Prior Pseudo-labels to {self.args.output_dir}")

        print("--- AS3L Stage 2 Completed Successfully ---")

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


def main():
    parser = argparse.ArgumentParser(description='AS3L Stage 2: Active Learning & Prior Pseudo-labels')
    parser.add_argument('--data_root', default='./data', type=str, help='Path to the root directory containing dataset.')
    parser.add_argument('--encoder_input_path', default='./output/as3l_run_1/simsiam_pretrain_cifar10_wrn282/wrn_28_2_encoder_fself.pth', type=str,
                        help='Path to the pre-trained encoder weights (f_self) from Stage 1.')
    parser.add_argument('--output_dir', default='./output/as3l_run_1/as3l_stage2_outputs', type=str,
                        help='Directory to save actively selected labeled samples and generated pseudo-labels.')
    parser.add_argument('--img_dim', default=32, type=int, help='Dimension of the input images.')
    parser.add_argument('--arch', default='wrn_28_2', type=str, help='Backbone architecture (must match Stage 1, e.g., wrn_28_2).')
    parser.add_argument('--backbone_feature_dim', type=int, default=128, # Fixed for WRN-28-2 (64*2)
                        help='Output feature dimension of the backbone BEFORE the SimSiam projection head. '
                             'For WRN-28-2, this is 128.')
    parser.add_argument('--feat_dim', default=2048, type=int,
                        help='Dimensionality of the projected features (d in SimSiam paper). This is the f_self dimension.')
    parser.add_argument('--num_proj_layers', default=2, type=int,
                        help='Number of layers in the projection MLP from SimSiam.')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in the dataset (e.g., CIFAR-10).')
    parser.add_argument('--num_labeled_samples_per_class', default=10, type=int,
                        help='Number of labeled samples to actively select PER CLASS (e.g., 10 for CIFAR-10 in paper).')
    parser.add_argument('--finetune_epochs', default=40, type=int,
                        help='Number of epochs to train the linear layer for f_fine. (Paper uses 40)')
    parser.add_argument('--finetune_lr', default=0.01, type=float, help='Learning rate for f_fine fine-tuning.')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for feature extraction and f_fine extraction.')
    parser.add_argument('--finetune_batch_size', default=256, type=int, help='Batch size for training the linear layer for f_fine.') 
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_clustering_runs_C', default=6, type=int,
                        help='Number of clustering runs (C) for stability and PPL generation (Paper uses 6).')
    parser.add_argument('--print_freq', default=100, type=int, help='Frequency to print fine-tuning progress.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer (for f_fine fine-tuning).')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay for regularization in f_fine fine-tuning.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index to use. Set to None for CPU.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')

    args = parser.parse_args()
        
    print("Parsed Arguments:", args)

    as3l_stage2 = AS3LStage2(args)
    as3l_stage2.run()

if __name__ == '__main__':
    main()


# This code is designed to be run as a script, so it will execute the main function when run directly.
# python as3l_stage2.py     --data_root ./data     --encoder_input_path ./output/my_as3l_runs/simsiam_pretrain_cifar10_wrn282/wrn_28_2_encoder_fself.pth     --output_dir ./output/my_as3l_runs/as3l_stage2_outputs     --arch wrn_28_2     --backbone_feature_dim 128     --feat_dim 2048     --num_proj_layers 2     --num_classes 10     --num_labeled_samples_per_class 10     --finetune_epochs 40     --num_clustering_runs_C 6     --gpu 0     --seed 42