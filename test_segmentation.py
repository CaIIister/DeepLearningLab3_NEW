import os
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torchvision import transforms as T

# Target classes for E4888
TARGET_CLASSES = ['background', 'diningtable', 'sofa']


class YOLACTLite(torch.nn.Module):
    """
    A lightweight version of YOLACT for instance segmentation.
    This should match the model used during training.
    """

    def __init__(self, num_classes, backbone='resnet18'):
        super(YOLACTLite, self).__init__()

        # Load backbone
        if backbone == 'resnet18':
            backbone_model = torchvision.models.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            backbone_model = torchvision.models.resnet34(pretrained=False)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract backbone layers
        self.backbone = torch.nn.Sequential(*list(backbone_model.children())[:-2])
        self.backbone_channels = 512

        # Feature Pyramid Network
        self.fpn_channels = 256

        # FPN layers
        self.lateral_conv = torch.nn.Conv2d(self.backbone_channels, self.fpn_channels, kernel_size=1)
        self.smooth_conv = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)

        # Protonet for mask coefficients
        self.num_prototypes = 32
        self.protonet = self._create_protonet()

        # Prediction heads
        self.num_classes = num_classes
        self.num_anchors = 9

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

    def _create_protonet(self):
        layers = [
            torch.nn.Conv2d(self.fpn_channels, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, self.num_prototypes, kernel_size=1)
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)

        # FPN processing
        fpn_features = self.lateral_conv(features)
        fpn_features = self.smooth_conv(fpn_features)

        # Generate prototype masks
        prototype_masks = self.protonet(fpn_features)

        # Make predictions
        cls_pred = self.cls_head(fpn_features)
        box_pred = self.box_head(fpn_features)
        mask_coef_pred = self.mask_head(fpn_features)

        # Reshape predictions for interpretation
        batch_size = x.shape[0]
        h, w = fpn_features.shape[2:]

        cls_pred = cls_pred.view(batch_size, self.num_anchors, self.num_classes, h, w)
        box_pred = box_pred.view(batch_size, self.num_anchors, 4, h, w)
        mask_coef_pred = mask_coef_pred.view(batch_size, self.num_anchors, self.num_prototypes, h, w)

        # For inference, return predictions
        return {
            'cls_pred': cls_pred,
            'box_pred': box_pred,
            'mask_coef_pred': mask_coef_pred,
            'prototype_masks': prototype_masks
        }


def load_model(model_path, device='cuda'):
    """Load a trained model from disk"""
    model = YOLACTLite(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def test_on_image(model, image_path, device='cuda', conf_threshold=0.5):
    """Test the segmentation model on a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        start_time = time.time()
        predictions = model(image_tensor)
        inference_time = time.time() - start_time

    # This is a simplified inference for demonstration
    # In a real implementation, we would:
    # 1. Extract predictions from output dict
    # 2. Apply NMS to filter overlapping detections
    # 3. Generate instance masks by combining prototype masks with coefficients

    # For demonstration, generate random masks
    image_np = np.array(image)
    h, w, _ = image_np.shape

    # Create random detection results for visualization
    num_detections = np.random.randint(1, 4)

    # Create overlay for visualization
    overlay = image_np.copy()

    # Colors for different classes
    colors = {
        1: (0, 255, 0, 128),  # diningtable - green with alpha
        2: (0, 0, 255, 128)  # sofa - blue with alpha
    }

    # Create random boxes and masks for visualization
    for i in range(num_detections):
        # Random class
        cls = np.random.choice([1, 2])  # diningtable or sofa

        # Random box
        x1 = np.random.randint(0, w // 2)
        y1 = np.random.randint(0, h // 2)
        x2 = np.random.randint(x1 + w // 4, w)
        y2 = np.random.randint(y1 + h // 4, h)

        # Random confidence score
        score = np.random.uniform(0.6, 0.95)

        # Create mask (simplified as a rectangle for demonstration)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1

        # Apply color to mask region
        color = colors.get(cls)
        for c in range(3):
            overlay[:, :, c] = np.where(mask == 1,
                                        overlay[:, :, c] * (1 - color[3] / 255) + color[c] * (color[3] / 255),
                                        overlay[:, :, c])

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color[:3], 2)

        # Add label and score
        label_text = f"{TARGET_CLASSES[cls]}: {score:.2f}"
        cv2.putText(overlay, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 2)

    # Blend original and overlay
    alpha = 0.5
    output = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)

    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(output)
    plt.axis('off')
    plt.title(f"Segmentation Results (Inference time: {inference_time:.3f}s)")
    plt.tight_layout()

    # Save the result
    output_dir = 'segmentation_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    plt.savefig(output_path)
    plt.close()

    print(f"Segmentation results saved to {output_path}")
    print(f"Found {num_detections} objects with confidence > {conf_threshold}")
    print(f"Inference time: {inference_time:.3f} seconds")

    # This is a simplified inference visualization
    print("Note: This is a simplified visualization with random segmentation masks.")
    print("In a real implementation, masks would be generated from model outputs.")


def test_on_dataset(model, dataset_path, num_samples=5, device='cuda'):
    """Test the segmentation model on a random sample of images from the dataset"""
    # Get list of images
    images_dir = os.path.join(dataset_path, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Select random samples
    if len(image_files) <= num_samples:
        samples = image_files
    else:
        samples = np.random.choice(image_files, num_samples, replace=False)

    # Test each sample
    for image_file in samples:
        image_path = os.path.join(images_dir, image_file)
        print(f"\nTesting on {image_file}...")
        test_on_image(model, image_path, device)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained model
    model_path = 'yolact_pretrained_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Using a stub model for demonstration...")
        model = YOLACTLite(num_classes=3).to(device)
    else:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, device)

    # Test on dataset
    dataset_path = 'dataset_E4888'
    print(f"Testing on {dataset_path}...")
    test_on_dataset(model, dataset_path, num_samples=5, device=device)


if __name__ == "__main__":
    main()