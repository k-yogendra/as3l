import argparse
import math
import random
import time
from os import makedirs, path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFilter, Image
import wandb


# --- 1. Data Augmentation and Transforms ---

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


# --- 2. Wide ResNet (WRN) Backbone Implementation ---
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
        # Store initial x for shortcut later
        shortcut = x 
        
        # First BN-ReLU sequence
        out = self.relu1(self.bn1(x))
        
        # If channels are not equal, apply shortcut convolution
        if not self.equal_channels:
            shortcut = self.convShortcut(out)
        
        # Main path through conv1 and its BN/ReLU
        out = self.relu2(self.bn2(self.conv1(out)))
        
        # Dropout
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
            
        # Second conv layer
        out = self.conv2(out)
        
        # Add shortcut connection
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

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(WideBasicBlock, nStages[1], n, stride=1, dropRate=dropRate)
        self.layer2 = self._make_layer(WideBasicBlock, nStages[2], n, stride=2, dropRate=dropRate)
        self.layer3 = self._make_layer(WideBasicBlock, nStages[3], n, stride=2, dropRate=dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], nStages[3])

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
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def WRN_28_2():
    """Returns a Wide ResNet 28-2 model."""
    return WideResNet(depth=28, widen_factor=2)


# --- 3. SimSiam Specific Modules ---

class SimSiamLoss(nn.Module):
    """
    Implementation of the SimSiam loss function.
    """

    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        z = z.detach()
        if self.ver == 'original':
            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()
        elif self.ver == 'simplified':
            return -nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):
        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)
        return 0.5 * loss1 + 0.5 * loss2


class projection_MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for projection in SimSiam.
    """
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


class prediction_MLP(nn.Module):
    """
    MLP for prediction in SimSiam.
    """
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


class SimSiam(nn.Module):
    """
    SimSiam network implementation.
    """
    def __init__(self, args):
        super(SimSiam, self).__init__()
        self.backbone = SimSiam.get_backbone(args.arch)
        backbone_out_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projector = projection_MLP(backbone_out_dim, args.feat_dim, args.num_proj_layers)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP(args.feat_dim)

    @staticmethod
    def get_backbone(backbone_name):
        if backbone_name == 'wrn_28_2':
            return WRN_28_2()
        else:
            raise ValueError(f"Unsupported backbone architecture: {backbone_name}")

    def forward(self, im_aug1, im_aug2):
        z1 = self.encoder(im_aug1)
        z2 = self.encoder(im_aug2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}


# --- 4. KNN Validation for Evaluation ---
class KNNValidation(object):
    """
    Perform K-Nearest Neighbors (KNN) validation for self-supervised learning.
    """

    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K

        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root=args.data_root,
                                         train=True,
                                         download=True,
                                         transform=base_transforms)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           drop_last=False)

        val_dataset = datasets.CIFAR10(root=args.data_root,
                                       train=False,
                                       download=True,
                                       transform=base_transforms)

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         drop_last=False)

    def _topk_retrieval(self):
        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = []
        train_labels = []
        with torch.no_grad():
            for inputs, targets in self.train_dataloader:
                inputs = inputs.to(self.device)
                features = self.model(inputs)
                features = nn.functional.normalize(features, dim=1)
                train_features.append(features)
                train_labels.append(targets.to(self.device))
        
        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)

                features = self.model(inputs.to(self.device))
                features = nn.functional.normalize(features, dim=1)

                dist = torch.mm(features, train_features.t())
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                retrieval = train_labels[yi]

                if self.K > 1:
                    predicted_labels, _ = torch.mode(retrieval, dim=1)
                else:
                    predicted_labels = retrieval.squeeze(1)

                total += targets.size(0)
                correct += predicted_labels.eq(targets.data).sum().item()

        top1 = correct / total
        return top1

    def eval(self):
        return self._topk_retrieval()


# --- 5. Training Utilities ---

def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate using a cosine annealing schedule.
    """
    lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
    Train the SimSiam model for one epoch.
    """
    model.train()
    batch_time_avg = 0
    loss_avg = 0
    num_batches = len(train_loader)

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # Forward pass through the model
        outs = model(im_aug1=images[0], im_aug2=images[1])
        loss = criterion(outs['z1'], outs['z2'], outs['p1'], outs['p2'])

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_time = time.time() - end
        batch_time_avg = (batch_time_avg * i + batch_time) / (i + 1)
        loss_avg = (loss_avg * i + loss.item()) / (i + 1)
        end = time.time()

        # Log to wandb
        wandb.log({
            'batch': epoch * num_batches + i,
            'batch_loss': loss.item(),
            'batch_time': batch_time
        })

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{num_batches}] '
                  f'Time {batch_time_avg:.3f} '
                  f'Loss {loss_avg:.4f}')

    return loss_avg


def save_checkpoint(epoch, model, optimizer, acc, filename, msg):
    """
    Save full SimSiam model checkpoint.
    """
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename, gpu_id):
    """
    Load model checkpoint.
    """
    map_location = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
    checkpoint = torch.load(filename, map_location=map_location)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, model, optimizer


# --- 6. Main Function and Argparse Setup ---

parser = argparse.ArgumentParser(description='SimSiam Self-supervised Pre-training')
parser.add_argument('--data_root', default='./data', type=str, help='Path to the root directory containing dataset.')
parser.add_argument('--exp_dir', default='./output/my_as3l_runs', type=str, help='Directory for saving experimental results.')
parser.add_argument('--trial', default='simsiam_pretrain_cifar10_wrn282', type=str, help='Identifier for the experiment trial.')
parser.add_argument('--img_dim', default=32, type=int, help='Dimension of the input images.')
parser.add_argument('--arch', default='wrn_28_2', type=str, help='Backbone architecture to use.')
parser.add_argument('--feat_dim', default=2048, type=int, help='Dimensionality of the projected features.')
parser.add_argument('--num_proj_layers', default=2, type=int, help='Number of layers in the projection MLP.')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size for training and validation.')
parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers.')
parser.add_argument('--epochs', default=800, type=int, help='Number of training epochs.')
parser.add_argument('--gpu', default=0, type=int, help='GPU index to use for training.')
parser.add_argument('--loss_version', default='original', type=str, help='Version of the loss function.')
parser.add_argument('--print_freq', default=10, type=int, help='Frequency to print training progress.')
parser.add_argument('--eval_freq', default=5, type=int, help='Frequency to perform KNN evaluation.')
parser.add_argument('--save_freq', default=50, type=int, help='Frequency to save model checkpoints.')
parser.add_argument('--resume', default=None, type=str, help='Path to a checkpoint file to resume training.')
parser.add_argument('--learning_rate', default=0.06, type=float, help='Initial learning rate.')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for regularization.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for the SGD optimizer.')
parser.add_argument('--backbone_output_path', type=str,
                    help='Path to save the pre-trained backbone weights.')
parser.add_argument('--wandb_project', default='simsiam-pretrain', type=str,
                    help='Project name for Weights & Biases logging')
parser.add_argument('--wandb_entity', default=None, type=str,
                    help='Entity name for Weights & Biases logging')
parser.add_argument('--wandb_name', default=None, type=str,
                    help='Run name for Weights & Biases logging')

args = parser.parse_args()

# Auto-generate backbone_output_path if not provided
if args.backbone_output_path is None:
    args.backbone_output_path = path.join(args.exp_dir, args.trial, f"{args.arch}_backbone_fself.pth")


def main():
    """
    Main function to set up training and validation for SimSiam.
    """
    # Initialize wandb
    wandb_name = args.wandb_name if args.wandb_name else f"{args.trial}__{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        config=vars(args)
    )

    # Create experiment directory if it doesn't exist
    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)

    # Setup trial-specific directory
    trial_dir = path.join(args.exp_dir, args.trial)
    if not path.exists(trial_dir):
        makedirs(trial_dir)
    print("Parsed Arguments:", vars(args))

    # Define data augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(sigma=[.1, 2.]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 training dataset
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

    # Initialize model, optimizer, and criterion
    model = SimSiam(args)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = SimSiamLoss(args.loss_version)

    # Move to GPU if available
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        cudnn.benchmark = True
    elif args.gpu is not None and not torch.cuda.is_available():
        print(f"Warning: GPU {args.gpu} requested but CUDA is not available. Using CPU.")
        args.gpu = None

    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume is not None:
        if path.isfile(args.resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.resume, args.gpu)
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Training and validation loop
    best_acc = 0.0
    validation = KNNValidation(args, model.encoder)
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Update learning rate
        current_lr = adjust_learning_rate(optimizer, epoch, args)
        
        print(f"Epoch {epoch}/{args.epochs} - Training...")
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'learning_rate': current_lr,
            'train_loss': train_loss,
        })

        # Perform KNN validation
        if epoch % args.eval_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} - Validating (KNN)...")
            val_top1_acc = validation.eval()
            print(f'KNN Top1 Accuracy: {val_top1_acc:.4f}')

            wandb.log({
                'epoch': epoch,
                'val_top1_knn': val_top1_acc,
            })

            # Save best model
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc
                save_checkpoint(epoch, model, optimizer, best_acc,
                              path.join(trial_dir, f'{args.trial}_best.pth'),
                              'Saving the best full SimSiam model!')

        # Save periodic checkpoints
        if epoch % args.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, val_top1_acc,
                          path.join(trial_dir, f'ckpt_epoch_{epoch}_{args.trial}.pth'),
                          'Saving checkpoint...')

    print(f'Training Finished. Best KNN accuracy achieved: {best_acc:.4f}')

    # Save the pre-trained backbone weights
    output_dir = path.dirname(args.backbone_output_path)
    if not path.exists(output_dir):
        makedirs(output_dir)

    print(f"Saving pre-trained backbone weights (f_self) to {args.backbone_output_path}...")
    torch.save(model.backbone.state_dict(), args.backbone_output_path)
    print("Pre-trained backbone weights saved successfully.")

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()


# This script is designed to be run as a standalone module.
# python simsiam_pretrain_wandb.py \
#     --data_root ./data \
#     --exp_dir ./output/as3l_run_1 \
#     --trial simsiam_pretrain_cifar10_wrn282 \
#     --epochs 10 \
#     --gpu 0 \
#     --arch wrn_28_2 \
#     --batch_size 512 \
#     --num_workers 4 \
#     --wandb_project simsiam-pretrain \
#     --wandb_name my_simsiam_run