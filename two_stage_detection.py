import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import time
import datetime
import math
from tqdm import tqdm

# Target classes for E4888
TARGET_CLASSES = ['diningtable', 'sofa']


class VOCCustomDataset(Dataset):
    """Dataset for loading images with only diningtable and sofa classes"""

    def __init__(self, root, image_set='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_set = image_set

        # Read image IDs from the specified split
        with open(os.path.join(root, f'{image_set}.txt'), 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        # Map class names to indices (1=diningtable, 2=sofa)
        # Background is 0
        self.class_to_idx = {cls: i + 1 for i, cls in enumerate(TARGET_CLASSES)}

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_path = os.path.join(self.root, 'images', f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        anno_path = os.path.join(self.root, 'annotations', f"{img_id}.xml")
        target = self.parse_voc_xml(ET.parse(anno_path).getroot())

        # Prepare target tensors
        boxes = []
        labels = []

        for obj in target['object']:
            # Get bounding box
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])

            # Skip invalid boxes
            if xmin >= xmax or ymin >= ymax:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj['name']])

        # Create target dict for Faster R-CNN
        target = {}

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Empty image case
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)

        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def parse_voc_xml(self, node):
        target = {}

        children = list(node)
        if children:
            # If node contains other nodes, recurse
            def_dic = {}
            for dc in children:
                # Process object node separately
                if dc.tag == 'object':
                    obj = {}
                    for x in list(dc):
                        if x.tag == 'bndbox':
                            bbox = {}
                            for box_child in list(x):
                                bbox[box_child.tag] = box_child.text
                            obj['bndbox'] = bbox
                        else:
                            obj[x.tag] = x.text
                    if 'object' not in target:
                        target['object'] = []
                    target['object'].append(obj)
                else:
                    def_dic[dc.tag] = self.parse_voc_xml(dc)
            if not def_dic:
                target[node.tag] = node.text
            else:
                target[node.tag] = def_dic
        else:
            target[node.tag] = node.text

        return target


class CustomTransforms:
    """Data transformations for training and validation"""

    def __init__(self, train=True):
        self.train = train

    def __call__(self, image, target):
        # Convert to tensor
        image = T.ToTensor()(image)

        # Normalize
        image = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        if self.train:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = T.functional.hflip(image)

                width = image.shape[2]
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target["boxes"] = boxes

        return image, target


def create_faster_rcnn_model(num_classes=3, pretrained=False, backbone='resnet18'):
    """
    Create a memory-efficient Faster R-CNN model.
    num_classes should include background (so 3 for 2 target classes)
    """
    # Load a pre-trained model for transfer learning
    if backbone == 'resnet18':
        if pretrained:
            backbone_model = torchvision.models.resnet18(weights='DEFAULT')
        else:
            backbone_model = torchvision.models.resnet18(weights=None)
    elif backbone == 'resnet34':
        if pretrained:
            backbone_model = torchvision.models.resnet34(weights='DEFAULT')
        else:
            backbone_model = torchvision.models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Extract and trim backbone
    backbone = torch.nn.Sequential(*list(backbone_model.children())[:-2])

    # FasterRCNN needs to know the number of output channels in the backbone
    backbone.out_channels = 512  # ResNet18 and ResNet34 both have 512 channels in final layer

    # Memory-efficient anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Memory-efficient RoI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Create model with memory-efficient parameters
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=600,
        max_size=800,  # Reduce from default 1333 to save memory
        rpn_pre_nms_top_n_train=1000,  # Reduce from 2000
        rpn_pre_nms_top_n_test=500,  # Reduce from 1000
        rpn_post_nms_top_n_train=500,  # Reduce from 1000
        rpn_post_nms_top_n_test=300  # Reduce from 500
    )

    return model


def train_model(model, data_loader, optimizer, device, epoch, print_freq=10):
    """Training function with memory optimizations"""
    model.train()

    # Metric tracking
    start_time = time.time()
    loss_sum = 0
    accumulation_steps = 4  # Accumulate gradients over 4 batches

    # Clear gradients
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(data_loader):
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Mixed precision for memory savings
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Normalize loss and backward
        losses = losses / accumulation_steps
        losses.backward()

        # Only update weights after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Track metrics
        loss_sum += losses.item() * accumulation_steps

        # Print progress
        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: [{epoch}][{i + 1}/{len(data_loader)}] '
                  f'Loss: {loss_sum / (i + 1):.4f} '
                  f'Time: {elapsed:.1f}s')

    # Make sure to update if dataset size is not divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = loss_sum / len(data_loader)
    print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

    return avg_loss


def evaluate_model(model, data_loader, device):
    """Validation function"""
    model.eval()

    # Initialize stats
    loss_sum = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Calculate loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_sum += losses.item()

    avg_loss = loss_sum / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss


def collate_fn(batch):
    """Custom collate function for the DataLoader"""
    return tuple(zip(*batch))


def main(pretrained=False, batch_size=2, epochs=10, backbone='resnet18'):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # Create and transform datasets
    dataset_path = 'dataset_E4888'

    # Set up transforms
    train_transforms = CustomTransforms(train=True)
    val_transforms = CustomTransforms(train=False)

    # Create datasets and dataloaders
    train_dataset = VOCCustomDataset(dataset_path, image_set='train', transforms=train_transforms)
    val_dataset = VOCCustomDataset(dataset_path, image_set='val', transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # Create model (background + 2 classes)
    model = create_faster_rcnn_model(num_classes=3, pretrained=pretrained, backbone=backbone)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0001)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_val_loss = float('inf')
    model_type = "pretrained" if pretrained else "scratch"

    print(f"Starting training for Faster R-CNN ({model_type})...")

    for epoch in range(epochs):
        # Train
        train_loss = train_model(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = evaluate_model(model, val_loader, device)

        # Update learning rate
        lr_scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'faster_rcnn_{model_type}_best.pth')
            print(f"Saved best model with validation loss: {val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), f'faster_rcnn_{model_type}_final.pth')
    print(f"Training complete. Final model saved as faster_rcnn_{model_type}_final.pth")


if __name__ == "__main__":
    # Train from scratch
    print("\n=== Training Faster R-CNN from scratch ===")
    main(pretrained=False, batch_size=2, epochs=10)

    # Train with pretrained weights
    print("\n=== Training Faster R-CNN with pretrained weights ===")
    main(pretrained=True, batch_size=2, epochs=10)