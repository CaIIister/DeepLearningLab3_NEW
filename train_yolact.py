import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VOCInstanceSegDataset
import transforms as T
from yolact_complete import YOLACTComplete
import time
from tqdm import tqdm


def collate_fn(batch):
    """Custom collate function for variable-sized images"""
    images, targets = tuple(zip(*batch))
    images = tuple(image for image in images)
    targets = tuple(target for target in targets)
    return images, targets


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """Train model for one epoch"""
    model.train()
    metric_logger = {'loss': 0, 'loss_cls': 0, 'loss_box': 0, 'loss_mask': 0}

    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        # Move data to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Stack images into tensor [batch_size, channels, height, width]
        images = torch.stack(images)

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update metrics
        for k in loss_dict:
            metric_logger[k] += loss_dict[k].item()
        metric_logger['loss'] += losses.item()

        # Print progress
        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: [{epoch}][{i + 1}/{len(data_loader)}] '
                  f'Loss: {metric_logger["loss"] / (i + 1):.4f} '
                  f'Cls: {metric_logger["loss_cls"] / (i + 1):.4f} '
                  f'Box: {metric_logger["loss_box"] / (i + 1):.4f} '
                  f'Mask: {metric_logger["loss_mask"] / (i + 1):.4f} '
                  f'Time: {elapsed:.1f}s')

    # Calculate average metrics
    for k in metric_logger:
        metric_logger[k] /= len(data_loader)

    print(f"Epoch {epoch} complete. Average Loss: {metric_logger['loss']:.4f}")
    return metric_logger


def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    metric_logger = {'loss': 0, 'loss_cls': 0, 'loss_box': 0, 'loss_mask': 0}

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Stack images into tensor [batch_size, channels, height, width]
            images = torch.stack(images)

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Update metrics
            for k in loss_dict:
                metric_logger[k] += loss_dict[k].item()
            metric_logger['loss'] += losses.item()

    # Calculate average metrics
    for k in metric_logger:
        metric_logger[k] /= len(data_loader)

    print(f"Validation Loss: {metric_logger['loss']:.4f}")
    return metric_logger


def main(pretrained=False, batch_size=2, num_epochs=10, learning_rate=1e-4):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Dataset path
    dataset_path = 'dataset_E4888'

    # Create train and validation datasets
    train_dataset = VOCInstanceSegDataset(
        root=dataset_path,
        image_set='train',
        transform=T.get_transform(train=True)
    )

    val_dataset = VOCInstanceSegDataset(
        root=dataset_path,
        image_set='val',
        transform=T.get_transform(train=False)
    )

    # Create data loaders
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

    # Create model
    model = YOLACTComplete(num_classes=3, pretrained=pretrained)
    model.to(device)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[7], gamma=0.1
    )

    # Training loop
    best_val_loss = float('inf')
    model_type = "pretrained" if pretrained else "scratch"
    print(f"Starting training for YOLACT ({model_type})...")

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Update learning rate
        lr_scheduler.step()

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), f'yolact_{model_type}_best.pth')
            print(f"Saved best model with validation loss: {val_metrics['loss']:.4f}")

    # Save final model
    torch.save(model.state_dict(), f'yolact_{model_type}_final.pth')
    print(f"Training complete. Final model saved as yolact_{model_type}_final.pth")


if __name__ == "__main__":
    # Train from scratch
    print("\n=== Training YOLACT from scratch ===")
    main(pretrained=False, batch_size=2, num_epochs=10)

    # Train with pretrained weights
    print("\n=== Training YOLACT with pretrained weights ===")
    main(pretrained=True, batch_size=2, num_epochs=10)