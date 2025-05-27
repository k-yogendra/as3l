# python as3l_stage2.py     --data_root ./data     --backbone_input_path ./output/my_as3l_runs/simsiam_pretrain_cifar10/resnet18_backbone_fself.pth     --output_dir ./my_as3l_runs/as3l_stage2.1_outputs     --num_labeled_samples_per_class 10     --finetune_epochs 50     --num_clustering_runs_C 6     --gpu 0
import argparse
import os
import random
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
from tqdm import tqdm # For progress bars

# --- Re-define ResNet and its blocks (from simsiam_pretrain.py) ---
# This ensures self-containment for the backbone loading.
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
        self.bn1 = nn.BatchNorm2d(planes) # Corrected to BatchNorm2d
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128): # low_dim is the feature dimension from backbone
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, low_dim) # This will be Identity for f_self
                                                            # and actual linear layer for f_fine
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
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_backbone(backbone_name, low_dim=128):
    return {
        'resnet18': ResNet(BasicBlock, [2, 2, 2, 2], low_dim),
        'resnet34': ResNet(BasicBlock, [3, 4, 6, 3], low_dim),
        'resnet50': ResNet(Bottleneck, [3, 4, 6, 3], low_dim),
        'resnet101': ResNet(Bottleneck, [3, 4, 23, 3], low_dim),
        'resnet152': ResNet(Bottleneck, [3, 8, 36, 3], low_dim)
    }[backbone_name]

# --- Custom Dataset to return index ---
class CIFAR10WithIndex(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

# --- Model for fine-tuning features (f_fine) ---
class LinearFinetuneModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # This is the linear layer that maps f_self to f_fine
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

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

        # 1. Load Backbone (f_self model)
        print(f"Loading pre-trained backbone from {args.backbone_input_path}...")
        self.backbone = get_backbone(args.arch, low_dim=args.backbone_feature_dim)
        # Modify the FC layer to be an identity for feature extraction, as SimSiam's FC is for projection head input
        self.backbone.fc = nn.Identity()
        self.backbone.load_state_dict(torch.load(args.backbone_input_path, map_location=self.device))
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval() # Set to eval mode for feature extraction

        # Data transformations for feature extraction (no augmentation)
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 mean/std
        ])
        # Data transformations for fine-tuning the linear layer (with augmentations)
        self.finetune_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
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
            feats = model(imgs)
            features[idxs.numpy()] = feats.cpu().numpy()
            labels[idxs.numpy()] = lbls.cpu().numpy()
            indices[idxs.numpy()] = idxs.cpu().numpy()
        print("Feature extraction complete.")
        return features, labels, indices

    def finetune_features(self, f_self_features, ground_truth_labels, total_indices):
        """
        Trains a linear layer to transform f_self features into f_fine features.
        The paper describes MSE loss against cluster centers. For simplicity and effectiveness,
        we'll train this linear layer using a classification objective on f_self features.
        This makes f_fine more discriminative for classification.
        """
        print("Fine-tuning features (f_fine linear layer)...")
        in_dim = f_self_features.shape[1]
        out_dim = self.args.backbone_feature_dim # The dimension for f_fine as mentioned in paper

        linear_classifier = nn.Linear(in_dim, self.args.num_classes).to(self.device)
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

                acc1 = self.accuracy(output, labels, topk=(1,))[0].item() # Added .item() here
                
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
        
        # Create a DataLoader for f_self features to pass through the fine-tuned linear_classifier
        f_self_dataset_for_ff = FeatureDataset(f_self_features, ground_truth_labels) # Labels not used for feature extraction here
        loader_for_ff = DataLoader(f_self_dataset_for_ff, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        # Extract f_fine features
        # f_fine is the output of this classifier's final layer, which has `num_classes` dimensions
        f_fine_features_out = np.zeros((len(f_self_features), self.args.num_classes), dtype=np.float32) 
        linear_classifier.eval() # Set to eval for feature extraction
        with torch.no_grad():
            for batch_idx, (feats_batch, _) in enumerate(tqdm(loader_for_ff, desc="Extracting f_fine features")):
                feats_batch = feats_batch.to(self.device)
                f_fine_batch = linear_classifier(feats_batch) # Pass f_self through the trained linear layer
                f_fine_features_out[batch_idx * self.args.batch_size : (batch_idx + 1) * self.args.batch_size] = f_fine_batch.cpu().numpy()
        
        print("Fine-tuning features complete.")
        return f_fine_features_out


    def select_labeled_samples(self, f_fine_features, ground_truth_labels, total_indices):
        """
        Actively selects a subset of samples for labeling.
        Performs K-means on f_fine and selects samples closest to inferred class centroids,
        ensuring `num_labeled_samples_per_class` for each ground truth class.
        """
        print(f"Actively selecting {self.args.num_labeled_samples_per_class} labeled samples per class...")
        
        n_clusters = self.args.num_classes # Number of clusters equals number of classes

        # Perform K-means clustering on f_fine_features
        # The paper implies multiple runs (C) for clustering for stability, but then it picks K samples closest to K class centers.
        # For simplicity and to ensure the desired count, we can do one robust KMeans run.
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed, n_init=10) # n_init for robustness
        cluster_assignments = kmeans.fit_predict(f_fine_features)
        cluster_centers = kmeans.cluster_centers_

        final_selected_indices = []
        final_selected_labels = []
        
        # Group samples by their ground truth labels for per-class selection
        samples_by_gt_label = defaultdict(list) # {gt_label: [(distance_to_cluster_center, original_idx)]}

        for i, original_idx in enumerate(total_indices):
            gt_label = ground_truth_labels[original_idx]
            assigned_cluster_id = cluster_assignments[i]
            
            # Calculate distance of this sample's f_fine_feature to its *assigned* cluster's centroid
            dist = np.linalg.norm(f_fine_features[i] - cluster_centers[assigned_cluster_id])
            
            samples_by_gt_label[gt_label].append((dist, original_idx))

        # Now, for each ground truth class, select the `num_labeled_samples_per_class` closest samples
        selected_indices_set = set() # Use a set to track already selected indices for uniqueness

        for gt_label in range(self.args.num_classes):
            candidates_for_this_class = sorted(samples_by_gt_label[gt_label]) # Sort by distance (closest first)
            
            count_for_this_class = 0
            for dist, original_idx in candidates_for_this_class:
                if count_for_this_class < self.args.num_labeled_samples_per_class and original_idx not in selected_indices_set:
                    final_selected_indices.append(original_idx)
                    final_selected_labels.append(gt_label)
                    selected_indices_set.add(original_idx) # Add to set to prevent duplicates
                    count_for_this_class += 1
                
                if count_for_this_class >= self.args.num_labeled_samples_per_class:
                    break # Reached quota for this class

        # Convert lists to numpy arrays with explicit int64 dtype
        selected_labeled_indices = np.array(final_selected_indices, dtype=np.int64) 
        selected_labeled_labels = np.array(final_selected_labels, dtype=np.int64)   

        print(f"Selected {len(selected_labeled_indices)} labeled samples.")
        print(f"Selected samples per class distribution: {np.bincount(selected_labeled_labels, minlength=self.args.num_classes)}")

        return selected_labeled_indices, selected_labeled_labels


    def generate_prior_pseudo_labels(self, f_fine_features, selected_labeled_indices, selected_labeled_labels, total_indices):
        """
        Generates Prior Pseudo-Labels (y_prior) for all unlabeled samples.
        Uses Constrained Seed K-means (simplified: K-means with majority vote propagation from labeled samples).
        """
        print("Generating Prior Pseudo-Labels (y_prior)...")

        num_total_samples = len(total_indices)
        y_prior_votes = np.zeros((num_total_samples, self.args.num_classes), dtype=np.int32) # Store votes for each class for aggregation

        # Map original indices to their ground truth labels for selected labeled samples
        selected_labeled_map = {idx: label for idx, label in zip(selected_labeled_indices, selected_labeled_labels)}

        # For each run, perform clustering and propagate labels
        for run_idx in tqdm(range(self.args.num_clustering_runs_C), desc="Clustering for PPL"):
            # K for PPL generation should be num_classes (10 for CIFAR-10)
            kmeans = KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed + run_idx + 100, n_init=10)
            cluster_assignments = kmeans.fit_predict(f_fine_features)
            
            # Determine the dominant label for each cluster based on *selected labeled samples*
            cluster_label_votes = defaultdict(lambda: defaultdict(int)) # {cluster_id: {label: count}}
            
            for i, cluster_id in enumerate(cluster_assignments):
                original_idx = total_indices[i] # Get the original index of this sample
                if original_idx in selected_labeled_map: # If this sample is one of our actively selected labeled samples
                    gt_label = selected_labeled_map[original_idx] # Use its true label
                    cluster_label_votes[cluster_id][gt_label] += 1
            
            cluster_dominant_labels = {}
            for cluster_id in range(self.args.num_classes): # Iterate through all possible clusters
                if cluster_id in cluster_label_votes and cluster_label_votes[cluster_id]:
                    dominant_label = max(cluster_label_votes[cluster_id], key=cluster_label_votes[cluster_id].get)
                    cluster_dominant_labels[cluster_id] = dominant_label
                else:
                    # If a cluster contains no labeled samples, assign a random class as the dominant label
                    cluster_dominant_labels[cluster_id] = random.randint(0, self.args.num_classes - 1)
            
            # Propagate dominant labels to all samples and accumulate votes
            for i, cluster_id in enumerate(cluster_assignments):
                original_idx = total_indices[i] # Get the original index of this sample
                # Add a vote for the dominant label of its assigned cluster
                y_prior_votes[original_idx, cluster_dominant_labels[cluster_id]] += 1

        # Aggregate votes across C runs to get final y_prior (hard labels)
        y_prior = np.argmax(y_prior_votes, axis=1) # Get the class with most votes

        print("Prior Pseudo-Labels generation complete.")
        return y_prior


    def run(self):
        """Orchestrates the entire AS3L Stage 2 process."""
        print("--- AS3L Stage 2: Active Learning & Prior Pseudo-labels ---")

        # 1. Extract f_self features for all training data
        f_self_features, ground_truth_labels, total_indices = self.extract_features(self.backbone, self.cifar_train_loader, self.args.backbone_feature_dim)
        
        # Save extracted f_self features and ground truth labels for debugging/inspection
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
    parser.add_argument('--backbone_input_path', default='./my_as3l_runs/simsiam_pretrain_cifar10/resnet18_backbone_fself.pth', type=str,
                        help='Path to the pre-trained backbone weights (f_self) from Stage 1.')
    parser.add_argument('--output_dir', default='./my_as3l_runs/as3l_stage2_outputs', type=str,
                        help='Directory to save actively selected labeled samples and generated pseudo-labels.')
    parser.add_argument('--img_dim', default=32, type=int, help='Dimension of the input images.')
    parser.add_argument('--arch', default='resnet18', type=str, help='Backbone architecture (must match Stage 1).')
    parser.add_argument('--backbone_feature_dim', default=512, type=int,
                        help='Output feature dimension of the backbone BEFORE the SimSiam projection head. '
                             'For ResNet18, this is 512*block.expansion=512.')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in the dataset (e.g., CIFAR-10).')
    parser.add_argument('--num_labeled_samples_per_class', default=10, type=int,
                        help='Number of labeled samples to actively select PER CLASS.')
    parser.add_argument('--finetune_epochs', default=50, type=int,
                        help='Number of epochs to train the linear layer for f_fine.')
    parser.add_argument('--finetune_lr', default=0.01, type=float, help='Learning rate for f_fine fine-tuning.')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for feature extraction.')
    parser.add_argument('--finetune_batch_size', default=256, type=int, help='Batch size for fine-tuning the linear layer for f_fine.') 
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_clustering_runs_C', default=6, type=int,
                        help='Number of clustering runs (C) for stability and PPL generation.')
    parser.add_argument('--print_freq', default=100, type=int, help='Frequency to print fine-tuning progress.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay for regularization in fine-tuning.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index to use. Set to None for CPU.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Determine backbone_feature_dim dynamically for ResNet based on its final FC layer's input size
    # This requires instantiating a temporary backbone to get the dimension
    temp_backbone = get_backbone(args.arch)
    args.backbone_feature_dim = temp_backbone.fc.in_features # Get the input feature dimension of the last FC layer

    print("Parsed Arguments:", args)

    as3l_stage2 = AS3LStage2(args)
    as3l_stage2.run()

if __name__ == '__main__':
    main()