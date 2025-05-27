# python simsiam_pretrain.py \
#     --data_root ./data \
#     --exp_dir ./output/my_as3l_runs \
#     --trial simsiam_pretrain_cifar10 \
#     --epochs 10 \
#     --gpu 0 \
#     --backbone_output_path ./output/my_as3l_runs/simsiam_pretrain_cifar10/resnet18_backbone_fself.pth


## Pretraining SimSiam

from PIL import ImageFilter, Image
import random
import time
import math
from os import path, makedirs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import argparse
import sys # Keep sys import as argparse.Namespace is used directly

class TwoCropsTransform:
    """
    A transformation class to create two different random crops of the same image.
    This is used to generate a query (q) and key (k) pair for contrastive learning.
    """

    def __init__(self, base_transform):
        # Initialize with a base transformation (e.g., augmentation pipeline).
        self.base_transform = base_transform

    def __call__(self, x):
        # Apply the base transformation twice to produce two augmented views.
        q = self.base_transform(x)  # First crop (query)
        k = self.base_transform(x)  # Second crop (key)
        return [q, k]  # Return as a list of query and key


class GaussianBlur(object):
    """
    Apply Gaussian blur as an augmentation.
    This is inspired by SimCLR: https://arxiv.org/abs/2002.05709.
    """

    def __init__(self, sigma=[.1, 2.]):
        # Define the range for the standard deviation (sigma) of the Gaussian kernel.
        self.sigma = sigma

    def __call__(self, x):
        # Randomly select a sigma value within the defined range.
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        # Apply Gaussian blur with the selected radius to the input image.
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# '''
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# '''

# from torch.autograd import Variable # Not used, so commented out


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.
    Implements two convolutional layers with Batch Normalization and ReLU activation.
    Includes a shortcut connection to handle dimensionality changes.
    """
    expansion = 1  # Defines how the number of output channels expands

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection for matching dimensions if necessary
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Forward pass through convolutional layers and shortcut connection
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add shortcut connection
        out = F.relu(out)  # Final ReLU activation
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet.
    Implements a three-layer structure to reduce computation while maintaining performance.
    """
    expansion = 4  # Output channels are 4x the input channels

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # First convolutional layer (1x1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer (3x3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Third convolutional layer (1x1)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut connection for matching dimensions if necessary
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Forward pass through the bottleneck layers and shortcut connection
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # Add shortcut connection
        out = F.relu(out)  # Final ReLU activation
        return out


class ResNet(nn.Module):
    """
    ResNet model definition.
    Builds the full network by stacking blocks and applying transformations.
    """

    def __init__(self, block, num_blocks, low_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64  # Initial number of input channels

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stacked layers using blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Fully connected layer for output
        self.fc = nn.Linear(512 * block.expansion, low_dim)
        # self.l2norm = Normalize(2)  # Optional normalization (commented out)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create a layer by stacking multiple blocks.
        Handles downsampling when stride > 1.
        """
        strides = [stride] + [1] * (num_blocks - 1)  # First block handles downsampling
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # Update input channels for the next block
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through all layers of the network
        out = F.relu(self.bn1(self.conv1(x)))  # Initial layer
        out = self.layer1(out)  # Layer 1
        out = self.layer2(out)  # Layer 2
        out = self.layer3(out)  # Layer 3
        out = self.layer4(out)  # Layer 4
        out = F.avg_pool2d(out, 4)  # Global average pooling
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)  # Fully connected layer
        # out = self.l2norm(out)  # Optional normalization (commented out)
        return out


# ResNet variants with different depths
def ResNet18(low_dim=128):
    return ResNet(BasicBlock, [2, 2, 2, 2], low_dim)

def ResNet34(low_dim=128):
    return ResNet(BasicBlock, [3, 4, 6, 3], low_dim)

def ResNet50(low_dim=128):
    return ResNet(Bottleneck, [3, 4, 6, 3], low_dim)

def ResNet101(low_dim=128):
    return ResNet(Bottleneck, [3, 4, 23, 3], low_dim)

def ResNet152(low_dim=128):
    return ResNet(Bottleneck, [3, 8, 36, 3], low_dim)


class SimSiamLoss(nn.Module):
    """
    Implementation of the SimSiam loss function.
    This loss is designed for self-supervised learning by comparing the similarity
    between pairs of projections and predictions from two augmented views of the same image.

    Reference:
    SimSiam: Exploring Simple Siamese Representation Learning (https://arxiv.org/abs/2011.10566)
    """

    def __init__(self, version='simplified'):
        """
        Initialize the SimSiam loss module.

        Args:
            version (str): Specifies the version of the loss.
                           'original' uses the original dot-product-based formulation,
                           'simplified' uses cosine similarity (default).
        """
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        """
        Compute the asymmetric loss between the prediction (p) and the projection (z).
        This enforces similarity between the two while detaching the gradient from `z`.

        Args:
            p (torch.Tensor): Prediction vector.
            z (torch.Tensor): Projection vector.

        Returns:
            torch.Tensor: Computed loss.
        """
        if self.ver == 'original':
            # Detach z to stop gradient flow
            z = z.detach()

            # Normalize vectors
            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            # Original formulation: negative dot product
            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            # Detach z to stop gradient flow
            z = z.detach()

            # Simplified formulation: negative cosine similarity
            return -nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):
        """
        Compute the SimSiam loss for two pairs of projections and predictions.

        Args:
            z1 (torch.Tensor): Projection vector from the first augmented view.
            z2 (torch.Tensor): Projection vector from the second augmented view.
            p1 (torch.Tensor): Prediction vector corresponding to z1.
            p2 (torch.Tensor): Prediction vector corresponding to z2.

        Returns:
            torch.Tensor: Averaged SimSiam loss.
        """
        # Compute the loss for each pair (p1, z2) and (p2, z1)
        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        # Average the two losses
        return 0.5 * loss1 + 0.5 * loss2


# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py


class KNNValidation(object):
    """
    Perform K-Nearest Neighbors (KNN) validation for self-supervised learning.
    This evaluates the learned representations by checking how well the model
    can classify images using KNN on feature embeddings.

    Args:
        args (Namespace): Configuration arguments including dataset paths, batch size, etc.
        model (nn.Module): The feature extraction model to evaluate.
        K (int): Number of neighbors to consider in KNN. Default is 1.
    """

    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K

        # Define base transformations for preprocessing CIFAR-10 dataset
        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 normalization
        ])

        # Load CIFAR-10 training dataset
        train_dataset = datasets.CIFAR10(root=args.data_root,
                                         train=True,
                                         download=True,
                                         transform=base_transforms)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,  # No shuffle for consistent feature extraction
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

        # Load CIFAR-10 validation dataset
        val_dataset = datasets.CIFAR10(root=args.data_root,
                                       train=False,
                                       download=True,
                                       transform=base_transforms)

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         drop_last=True)

    def _topk_retrieval(self):
        """
        Extract features from the validation dataset and perform KNN search on
        the training dataset features to classify validation images.

        Returns:
            float: Top-1 accuracy of KNN classification on the validation dataset.
        """
        # Number of training data points
        n_data = self.train_dataloader.dataset.data.shape[0]
        feat_dim = self.args.feat_dim  # Feature dimension from the model

        self.model.eval()  # Set model to evaluation mode
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()  # Clear GPU cache for efficient memory usage

        # Create tensor to store all training features
        train_features = torch.zeros([feat_dim, n_data], device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # Forward pass to extract features
                features = self.model(inputs)
                features = nn.functional.normalize(features)  # Normalize feature vectors
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

            # Get training labels
            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)

                # Extract features for validation inputs
                features = self.model(inputs.to(self.device))

                # Compute pairwise cosine similarity between validation and training features
                dist = torch.mm(features, train_features)

                # Retrieve top-K neighbors
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)

                # Get corresponding labels of top-K neighbors
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                # Take the most likely label (top-1 retrieval)
                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                # Update total and correct predictions
                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()

        # Compute top-1 accuracy
        top1 = correct / total
        return top1

    def eval(self):
        """
        Public method to evaluate the model using KNN validation.

        Returns:
            float: Top-1 accuracy.
        """
        return self._topk_retrieval()


class projection_MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for projection in SimSiam.
    This module projects the backbone's output to a feature space for contrastive learning.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        num_layers (int): Number of layers in the MLP (default: 2).
    """
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim  # Hidden layer dimension
        self.num_layers = num_layers

        # First layer: Fully connected + BatchNorm + ReLU
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Second layer: Fully connected + BatchNorm + ReLU (optional)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Third layer: Fully connected + BatchNorm without learnable affine parameters
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # See SimSiam paper (Page 5, Paragraph 2)
        )

    def forward(self, x):
        """
        Forward pass through the projection MLP.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Projected features.
        """
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x


class prediction_MLP(nn.Module):
    """
    MLP for prediction in SimSiam.
    Maps the projected features to the prediction space.

    Args:
        in_dim (int): Input feature dimension (default: 2048).
    """
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim  # Output dimension matches input dimension
        hidden_dim = int(out_dim / 4)  # Reduce feature dimension in the hidden layer

        # First layer: Fully connected + BatchNorm + ReLU
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Second layer: Fully connected (no activation)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        Forward pass through the prediction MLP.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted features.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    """
    SimSiam network implementation.
    Combines a backbone, a projection MLP, and a prediction MLP for self-supervised learning.

    Args:
        args (Namespace): Configuration arguments for the model.
    """
    def __init__(self, args):
        super(SimSiam, self).__init__()
        # Initialize the backbone (e.g., ResNet variants)
        self.backbone = SimSiam.get_backbone(args.arch)
        out_dim = self.backbone.fc.weight.shape[1]  # Feature dimension from the backbone
        self.backbone.fc = nn.Identity()  # Remove the fully connected layer from the backbone

        # Initialize the projection MLP
        self.projector = projection_MLP(out_dim, args.feat_dim, args.num_proj_layers)

        # Combine backbone and projector into a single encoder
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        # Initialize the prediction MLP
        self.predictor = prediction_MLP(args.feat_dim)

    @staticmethod
    def get_backbone(backbone_name):
        """
        Retrieve the backbone model based on the specified architecture.

        Args:
            backbone_name (str): Name of the backbone architecture.

        Returns:
            nn.Module: Backbone model instance.
        """
        return {
            'resnet18': ResNet18(),
            'resnet34': ResNet34(),
            'resnet50': ResNet50(),
            'resnet101': ResNet101(),
            'resnet152': ResNet152()
        }[backbone_name]

    def forward(self, im_aug1, im_aug2):
        """
        Forward pass through the SimSiam model.

        Args:
            im_aug1 (torch.Tensor): Augmented view 1 of the input image batch.
            im_aug2 (torch.Tensor): Augmented view 2 of the input image batch.

        Returns:
            dict: Output projections and predictions for both views.
                  Keys: 'z1', 'z2', 'p1', 'p2'
        """
        # Pass the first augmented view through the encoder
        z1 = self.encoder(im_aug1)
        # Pass the second augmented view through the encoder
        z2 = self.encoder(im_aug2)

        # Predict features for both views
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Return projections and predictions
        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}


# Define configuration parameters for the SimSiam experiment using argparse.Namespace.
# sys.argv manipulation is removed as it's not needed in a standalone script.
# Add argparse to handle command-line arguments for better control.
parser = argparse.ArgumentParser(description='SimSiam Self-supervised Pre-training')
parser.add_argument('--data_root', default='./data', type=str, help='Path to the root directory containing dataset.')
parser.add_argument('--exp_dir', default='./experiments', type=str, help='Directory for saving experimental results (e.g., checkpoints, logs).')
parser.add_argument('--trial', default='1', type=str, help='Identifier for the experiment trial.')
parser.add_argument('--img_dim', default=32, type=int, help='Dimension of the input images (e.g., 32x32 for CIFAR-10).')
parser.add_argument('--arch', default='resnet18', type=str, help='Backbone architecture to use (e.g., ResNet18).')
parser.add_argument('--feat_dim', default=2048, type=int, help='Dimensionality of the projected features.')
parser.add_argument('--num_proj_layers', default=2, type=int, help='Number of layers in the projection MLP.')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size for training and validation.')
parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers.')
parser.add_argument('--epochs', default=800, type=int, help='Number of training epochs.')
parser.add_argument('--gpu', default=0, type=int, help='GPU index to use for training (e.g., 0 for the first GPU).')
parser.add_argument('--loss_version', default='original', type=str, help='Version of the loss function (\'simplified\' or \'original\').')
parser.add_argument('--print_freq', default=10, type=int, help='Frequency (in batches) to print training progress.')
parser.add_argument('--eval_freq', default=5, type=int, help='Frequency (in epochs) to perform KNN evaluation.')
parser.add_argument('--save_freq', default=50, type=int, help='Frequency (in epochs) to save model checkpoints.')
parser.add_argument('--resume', default=None, type=str, help='Path to a checkpoint file to resume training, if any.')
parser.add_argument('--learning_rate', default=0.06, type=float, help='Initial learning rate for the optimizer.')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for regularization.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for the SGD optimizer.')
parser.add_argument('--backbone_output_path', default='./experiments/pretrained_backbone.pth', type=str,
                    help='Path to save the pre-trained backbone weights (f_self).')

args = parser.parse_args(sys.argv[1:]) # Parse arguments. sys.argv[1:] to exclude the script name itself.

# Print the parsed arguments for verification and debugging
print("Parsed Arguments:", args)


def main():
    """
    Main function to set up training and validation for SimSiam.
    Handles directory creation, data preparation, model setup, training loop, and checkpointing.
    """
    # Create experiment directory if it doesn't exist
    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)

    # Setup trial-specific directory and logger for TensorBoard
    trial_dir = path.join(args.exp_dir, args.trial)
    if not path.exists(trial_dir): # Create trial directory if it doesn't exist
        makedirs(trial_dir)
    logger = SummaryWriter(trial_dir)
    print(vars(args))  # Print experiment configuration

    # Define data augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # Random brightness, contrast, saturation, hue
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(sigma=[.1, 2.]), # Added Gaussian Blur as per SimSiam paper
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
    ])

    # Load CIFAR-10 training dataset with TwoCropsTransform for SimSiam
    train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True,
                                 transform=TwoCropsTransform(train_transforms))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    # Initialize SimSiam model
    model = SimSiam(args)

    # Define SGD optimizer with momentum and weight decay
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # Initialize loss function (original or simplified version)
    criterion = SimSiamLoss(args.loss_version)

    # Move model and loss to GPU if available
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        cudnn.benchmark = True  # Enable auto-tuning for faster training

    # Resume from a checkpoint if provided
    start_epoch = 1
    if args.resume is not None:
        if path.isfile(args.resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.resume)
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # Training and validation loop
    best_acc = 0.0
    validation = KNNValidation(args, model.encoder)  # Initialize KNN validation
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)  # Update learning rate
        print("Training...")

        # Train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        logger.add_scalar('Loss/train', train_loss, epoch)  # Log training loss

        # Perform KNN validation periodically
        if epoch % args.eval_freq == 0:
            print("Validating...")
            val_top1_acc = validation.eval()  # Evaluate KNN accuracy
            print('Top1: {}'.format(val_top1_acc))

            # Save the best model checkpoint
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc
                save_checkpoint(epoch, model, optimizer, best_acc,
                                path.join(trial_dir, '{}_best.pth'.format(args.trial)),
                                'Saving the best model!')
            logger.add_scalar('Acc/val_top1', val_top1_acc, epoch)  # Log validation accuracy

        # Save model periodically
        if epoch % args.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, val_top1_acc,
                            path.join(trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)),
                            'Saving...')

    print('Best accuracy:', best_acc)

    # Save the final full model checkpoint
    save_checkpoint(epoch, model, optimizer, val_top1_acc,
                    path.join(trial_dir, '{}_last.pth'.format(args.trial)),
                    'Saving the model at the last epoch.')
    
    # --- AS3L Stage 1: Save the pre-trained backbone (f_self) weights ---
    print(f"Saving pre-trained backbone weights to {args.backbone_output_path}...")
    torch.save(model.backbone.state_dict(), args.backbone_output_path)
    print("Pre-trained backbone weights saved successfully.")


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
    Train the SimSiam model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): SimSiam model.
        criterion (nn.Module): Loss function (e.g., SimSiamLoss).
        optimizer (Optimizer): Optimizer (e.g., SGD).
        epoch (int): Current epoch number.
        args (Namespace): Experiment arguments.

    Returns:
        float: Average training loss for the epoch.
    """
    batch_time = AverageMeter('Time', ':6.3f')  # Measure batch processing time
    losses = AverageMeter('Loss', ':.4e')  # Track average loss
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()  # Set model to training mode
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # Forward pass through the model
        outs = model(im_aug1=images[0], im_aug2=images[1])
        loss = criterion(outs['z1'], outs['z2'], outs['p1'], outs['p2'])  # Compute SimSiam loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss and batch time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:  # Display progress periodically
            progress.display(i)

    return losses.avg  # Return average loss


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate using a cosine annealing schedule.

    Args:
        optimizer (Optimizer): Optimizer to update.
        epoch (int): Current epoch number.
        args (Namespace): Experiment arguments.
    """
    lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """
    Helper class to compute and store the average and current value of metrics.
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Helper class to display progress during training or validation.
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(epoch, model, optimizer, acc, filename, msg):
    """
    Save full SimSiam model checkpoint (including backbone, projector, predictor).

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): SimSiam model to save.
        optimizer (Optimizer): Optimizer to save.
        acc (float): Accuracy value to save.
        filename (str): Path to save the checkpoint file.
        msg (str): Message to display after saving.
    """
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(), # Saves the full SimSiam model state_dict
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    """
    Load model checkpoint.

    Args:
        model (nn.Module): Model to load checkpoint into.
        optimizer (Optimizer): Optimizer to load checkpoint into.
        filename (str): Path to the checkpoint file.

    Returns:
        tuple: (start_epoch, model, optimizer)
    """
    # Use map_location to handle loading on different devices
    map_location = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
    checkpoint = torch.load(filename, map_location=map_location)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, model, optimizer


if __name__ == '__main__':
    main()