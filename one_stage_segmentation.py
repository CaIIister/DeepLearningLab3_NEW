import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import time
import datetime
import math
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms as T

# Target classes for E4888
TARGET_CLASSES = ['diningtable', 'sofa']


class VOCInstanceSegDataset(Dataset):
    """Dataset for instance segmentation using VOC data"""

    def __init__(self, root, image_set='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_set = image_set

        # Read image IDs from the specified split
        with open(os.path.join(root, f'{image_set}.txt'), 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        # Map class names to indices
        self.class_to_idx = {cls: i + 1 for i, cls in enumerate(TARGET_CLASSES)}

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_path = os.path.join(self.root, 'images', f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        anno_path = os.path.join(self.root, 'annotations', f"{img_id}.xml")
        target = self.parse_voc_xml(ET.parse(anno_path).getroot())

        # Get image dimensions
        width, height = img.size

        # Prepare target tensors
        boxes = []
        labels = []
        masks = []  # For instance segmentation

        # Create binary masks for each object
        for obj in target['object']:
            # Get bounding box
            bbox = obj['bndbox']
            xmin = max(0, float(bbox['xmin']))
            ymin = max(0, float(bbox['ymin']))
            xmax = min(width, float(bbox['xmax']))
            ymax = min(height, float(bbox['ymax']))

            # Skip invalid boxes
            if xmin >= xmax or ymin >= ymax:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj['name']])

            # Create a simple rectangular mask for demonstration
            mask = torch.zeros((height, width), dtype=torch.uint8)
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
            masks.append(mask)

        # Create target dict
        target = {}

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["masks"] = torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            # Empty image case
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)

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


class YOLACTLite(torch.nn.Module):
    """
    A lightweight version of YOLACT for instance segmentation, optimized for lower memory usage.
    """

    def __init__(self, num_classes, pretrained=False, backbone='resnet18'):
        super(YOLACTLite, self).__init__()

        # Load a pre-trained backbone
        if backbone == 'resnet18':
            backbone_model = torchvision.models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            backbone_model = torchvision.models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract backbone layers
        self.backbone = torch.nn.Sequential(*list(backbone_model.children())[:-2])
        self.backbone_channels = 512  # ResNet18 and ResNet34 both have 512 output channels

        # Feature Pyramid Network
        self.fpn_channels = 256

        # FPN layers
        self.lateral_conv = torch.nn.Conv2d(self.backbone_channels, self.fpn_channels, kernel_size=1)
        self.smooth_conv = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)

        # Protonet for mask coefficients
        self.num_prototypes = 32  # Reduced for memory efficiency
        self.protonet = self._create_protonet()

        # Prediction heads
        self.num_classes = num_classes
        self.num_anchors = 9  # 3 scales x 3 aspect ratios

        # Classification head
        self.cls_head = torch.nn.Conv2d(
            self.fpn_channels,
            self.num_anchors * self.num_classes,
            kernel_size=3,
            padding=1
        )

        # Box regression head
        self.box_head = torch.nn.Conv2d(
            self.fpn_channels,
            self.num_anchors * 4,
            kernel_size=3,
            padding=1
        )

        # Mask coefficient head
        self.mask_head = torch.nn.Conv2d(
            self.fpn_channels,
            self.num_anchors * self.num_prototypes,
            kernel_size=3,
            padding=1
        )

        # Initialize weights
        self._initialize_weights()

    def _create_protonet(self):
        # Simplified protonet with fewer layers for memory efficiency
        layers = [
            torch.nn.Conv2d(self.fpn_channels, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, self.num_prototypes, kernel_size=1)
        ]
        return torch.nn.Sequential(*layers)

    def _initialize_weights(self):
        # Initialize prediction heads
        for m in [self.cls_head, self.box_head, self.mask_head]:
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.constant_(m.bias, 0)

        # Initialize protonet
        for m in self.protonet.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, images, targets=None):
        """
        Forward pass with training and inference modes
        """
        # Extract features from backbone
        features = self.backbone(images)

        # FPN processing
        fpn_features = self.lateral_conv(features)
        fpn_features = self.smooth_conv(fpn_features)

        # Generate prototype masks
        prototype_masks = self.protonet(fpn_features)

        # Make predictions
        cls_pred = self.cls_head(fpn_features)
        box_pred = self.box_head(fpn_features)
        mask_coef_pred = self.mask_head(fpn_features)

        # Reshape predictions for loss calculation
        batch_size = images.shape[0]
        h, w = fpn_features.shape[2:]

        # Reshape for interpretation - (batch, anchors, classes/boxes/coeffs, h, w)
        cls_pred = cls_pred.view(batch_size, self.num_anchors, self.num_classes, h, w)
        box_pred = box_pred.view(batch_size, self.num_anchors, 4, h, w)
        mask_coef_pred = mask_coef_pred.view(batch_size, self.num_anchors, self.num_prototypes, h, w)

        # If training, calculate loss
        if self.training and targets is not None:
            loss_dict = self.compute_loss(
                cls_pred, box_pred, mask_coef_pred, prototype_masks, targets
            )
            return loss_dict

        # For inference, return predictions
        return {
            'cls_pred': cls_pred,
            'box_pred': box_pred,
            'mask_coef_pred': mask_coef_pred,
            'prototype_masks': prototype_masks
        }

    def compute_loss(self, cls_pred, box_pred, mask_coef_pred, prototype_masks, targets):
        """Basic loss calculation for YOLACT"""
        batch_size = cls_pred.size(0)
        device = cls_pred.device

        # Initialize losses
        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)
        mask_loss = torch.tensor(0.0, device=device)

        # Process each item in batch
        for b in range(batch_size):
            # Get target boxes and labels for this batch item
            target_boxes = targets[b].get("boxes", None)
            target_labels = targets[b].get("labels", None)
            target_masks = targets[b].get("masks", None)

            if target_boxes is None or len(target_boxes) == 0:
                continue

            # Reshape predictions for this batch item
            # cls: [batch, anchors, classes, h, w] -> [anchors*h*w, classes]
            b_cls_pred = cls_pred[b].permute(0, 3, 4, 1).reshape(-1, self.num_classes)

            # box: [batch, anchors, 4, h, w] -> [anchors*h*w, 4]
            b_box_pred = box_pred[b].permute(0, 3, 4, 1).reshape(-1, 4)

            # Basic classification loss (binary cross entropy)
            # Create a simple target - for each GT box, find the closest anchor
            num_anchors = b_cls_pred.size(0)
            gt_cls = torch.zeros((num_anchors, self.num_classes), device=device)

            # For simplicity, assign all anchors with IoU > threshold to target class
            for box_idx, (box, label) in enumerate(zip(target_boxes, target_labels)):
                # Create a simplified target - just use the class label
                if label > 0:  # Skip background class
                    gt_cls[:, label] = 1.0

            # Binary Cross Entropy loss for classification
            cls_loss += F.binary_cross_entropy_with_logits(
                b_cls_pred, gt_cls, reduction='sum'
            ) / max(1, len(target_boxes))

            # Simple L1 loss for box regression
            # For simplicity, use the GT boxes as targets for all predictions
            # In a real implementation, you'd match predictions to targets
            box_targets = target_boxes.repeat(num_anchors // len(target_boxes) + 1, 1)[:num_anchors]
            box_loss += F.smooth_l1_loss(
                b_box_pred, box_targets, reduction='sum'
            ) / max(1, len(target_boxes))

            # Simple mask loss if masks are available
            if target_masks is not None and len(target_masks) > 0:
                # For simplicity, use direct binary cross entropy on prototype masks
                mask_preds = prototype_masks[b]
                mask_targets = target_masks.float()

                # Reshape to match
                if mask_preds.size(0) != mask_targets.size(0):
                    mask_preds = mask_preds[:mask_targets.size(0)] if mask_preds.size(0) > mask_targets.size(
                        0) else mask_preds
                    mask_targets = mask_targets[:mask_preds.size(0)] if mask_targets.size(0) > mask_preds.size(
                        0) else mask_targets

                # Binary cross entropy for mask prediction
                mask_loss += F.binary_cross_entropy_with_logits(
                    mask_preds, mask_targets, reduction='sum'
                ) / max(1, len(target_masks))

        # Normalize by batch size
        cls_loss = cls_loss / batch_size
        box_loss = box_loss / batch_size
        mask_loss = mask_loss / batch_size

        # Weight the losses
        cls_weight = 1.0
        box_weight = 1.0
        mask_weight = 1.0

        # Return loss dictionary
        loss_dict = {
            'loss_cls': cls_loss * cls_weight,
            'loss_box': box_loss * box_weight,
            'loss_mask': mask_loss * mask_weight
        }

        return loss_dict


class CustomTransforms:
    """Data transformations for training and validation"""

    def __init__(self, train=True):
        self.train = train

    def __call__(self, image, target):
        # First resize to ensure consistent dimensions
        width, height = image.size

        # Fixed size for all images (e.g., 512x512)
        new_width, new_height = 512, 512

        # Resize image
        image = image.resize((new_width, new_height), Image.BILINEAR)

        # Need to adjust bounding boxes and masks for the new size
        if "boxes" in target and len(target["boxes"]) > 0:
            # Scale factors
            scale_x = new_width / width
            scale_y = new_height / height

            # Scale boxes
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
            target["boxes"] = boxes

            # Scale masks if they exist
            if "masks" in target:
                # Resize masks to new dimensions
                masks = target["masks"]
                resized_masks = []
                for mask in masks:
                    mask_pil = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                    mask_resized = mask_pil.resize((new_width, new_height), Image.NEAREST)
                    mask_tensor = torch.tensor(np.array(mask_resized) > 127, dtype=torch.uint8)
                    resized_masks.append(mask_tensor)

                target["masks"] = torch.stack(resized_masks) if resized_masks else masks

        # Convert to tensor
        image = T.ToTensor()(image)

        # Normalize
        image = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        if self.train:
            # Random horizontal flip - flip both image and masks
            if torch.rand(1).item() > 0.5:
                image = T.functional.hflip(image)
                if "masks" in target:
                    target["masks"] = torch.flip(target["masks"], [2])

                # Flip boxes
                if "boxes" in target and len(target["boxes"]) > 0:
                    width = image.shape[2]
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target["boxes"] = boxes

        return image, target


def collate_fn(batch):
    """Custom collate function for the DataLoader"""
    return tuple(zip(*batch))


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
        images = torch.stack(images)
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
            images = torch.stack(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # For evaluation, temporarily set to train mode to get losses
            model.train()
            loss_dict = model(images, targets)
            model.eval()

            # Calculate loss - handle both dict and direct loss cases
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                # If loss_dict is actually just a scalar loss
                losses = loss_dict

            loss_sum += losses.item()

    avg_loss = loss_sum / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss


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
    train_dataset = VOCInstanceSegDataset(dataset_path, image_set='train', transforms=train_transforms)
    val_dataset = VOCInstanceSegDataset(dataset_path, image_set='val', transforms=val_transforms)

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
    model = YOLACTLite(num_classes=3, pretrained=pretrained, backbone=backbone)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0001)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_val_loss = float('inf')
    model_type = "pretrained" if pretrained else "scratch"

    print(f"Starting training for YOLACT ({model_type})...")

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
            torch.save(model.state_dict(), f'yolact_{model_type}_best.pth')
            print(f"Saved best model with validation loss: {val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), f'yolact_{model_type}_final.pth')
    print(f"Training complete. Final model saved as yolact_{model_type}_final.pth")


if __name__ == "__main__":
    # Train from scratch
    print("\n=== Training YOLACT from scratch ===")
    main(pretrained=False, batch_size=2, epochs=10)

    # Train with pretrained weights
    print("\n=== Training YOLACT with pretrained weights ===")
    main(pretrained=True, batch_size=2, epochs=10)