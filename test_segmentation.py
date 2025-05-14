import os
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torchvision import transforms as T
import torch.nn.functional as F

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
            backbone_model = torchvision.models.resnet18(weights=None)
        elif backbone == 'resnet34':
            backbone_model = torchvision.models.resnet34(weights=None)
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


def test_on_image(model, image_path, device='cuda', conf_threshold=0.5, suffix=""):
    """Test the segmentation model on a single image with proper mask generation"""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height)

    # Resize for model input (maintain aspect ratio)
    max_size = 640
    ratio = min(max_size / max(orig_size), 1.0)
    new_size = (int(orig_size[0] * ratio), int(orig_size[1] * ratio))
    image_resized = image.resize(new_size, Image.BILINEAR)

    # Convert to tensor and normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_resized).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        start_time = time.time()
        predictions = model(image_tensor)
        inference_time = time.time() - start_time

    # Get predictions
    cls_pred = predictions['cls_pred'][0]  # Shape: [anchors, classes, h, w]
    box_pred = predictions['box_pred'][0]  # Shape: [anchors, 4, h, w]
    mask_coef_pred = predictions['mask_coef_pred'][0]  # Shape: [anchors, prototypes, h, w]
    prototype_masks = predictions['prototype_masks'][0]  # Shape: [prototypes, h, w]

    # Process predictions to get actual detections
    # First, reshape predictions for processing
    num_anchors = cls_pred.shape[0]
    num_classes = cls_pred.shape[1]
    num_prototypes = prototype_masks.shape[0]

    # Reshape for post-processing
    h, w = prototype_masks.shape[1:3]

    # 1. Get max class scores and corresponding class ids for each anchor
    cls_scores, cls_ids = torch.max(cls_pred.view(num_anchors, num_classes, -1).mean(dim=2), dim=1)

    # 2. Apply score threshold
    keep = cls_scores > conf_threshold
    scores = cls_scores[keep]
    labels = cls_ids[keep]

    # If no detections, return early
    if scores.numel() == 0:
        print(f"No detections found above threshold {conf_threshold}")
        return

    # 3. Get corresponding box predictions
    boxes = box_pred.view(num_anchors, 4, -1).mean(dim=2)[keep]

    # 4. Get corresponding mask coefficients
    mask_coeffs = mask_coef_pred.view(num_anchors, num_prototypes, -1).mean(dim=2)[keep]

    # Scale boxes to original image size
    boxes[:, 0::2] *= orig_size[0] / new_size[0]  # scale x
    boxes[:, 1::2] *= orig_size[1] / new_size[1]  # scale y

    # Convert to numpy for visualization
    image_np = np.array(image)
    h_orig, w_orig, _ = image_np.shape

    # Create overlay for visualization
    overlay = image_np.copy()

    # Colors for different classes (handle more than 2 classes if needed)
    colors = [
        (0, 255, 0, 128),  # diningtable - green with alpha
        (0, 0, 255, 128),  # sofa - blue with alpha
        (255, 0, 0, 128)  # extra color
    ]

    # For each detection, generate mask and visualize
    for i in range(len(scores)):
        # Get detection info
        box = boxes[i].cpu().numpy().astype(np.int32)
        score = scores[i].cpu().item()
        label = labels[i].cpu().item()

        # Get class name - skip background (0)
        if label == 0:
            continue

        class_name = TARGET_CLASSES[label]

        # Get mask by combining prototypes with coefficients
        mask_coeff = mask_coeffs[i]

        # Resize prototype masks to original image size
        proto_masks_resized = F.interpolate(
            prototype_masks.unsqueeze(0),
            size=(h_orig, w_orig),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Combine prototypes using coefficients
        instance_mask = torch.zeros((h_orig, w_orig), device=device)
        for j in range(num_prototypes):
            instance_mask += mask_coeff[j] * proto_masks_resized[j]

        # Apply sigmoid to get probability map
        instance_mask = torch.sigmoid(instance_mask)

        # Threshold mask to get binary mask
        binary_mask = (instance_mask > 0.5).cpu().numpy().astype(np.uint8)

        # Apply color to mask region
        color = colors[label - 1 % len(colors)]
        for c in range(3):
            overlay[:, :, c] = np.where(binary_mask == 1,
                                        overlay[:, :, c] * (1 - color[3] / 255) + color[c] * (color[3] / 255),
                                        overlay[:, :, c])

        # Draw bounding box
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color[:3], 2)

        # Add label and score
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(overlay, label_text, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 2)

    # Blend original and overlay
    alpha = 0.5
    output = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)

    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(output)
    plt.axis('off')
    plt.title(f"Segmentation Results{suffix} (Inference time: {inference_time:.3f}s)")
    plt.tight_layout()

    # Save the result
    output_dir = 'segmentation_results'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}{suffix}.jpg")
    plt.savefig(output_path)
    plt.close()

    print(f"Segmentation results saved to {output_path}")
    print(f"Found {len(scores)} objects with confidence > {conf_threshold}")
    print(f"Inference time: {inference_time:.3f} seconds")


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define model paths
    pretrained_model_path = 'yolact_pretrained_best.pth'
    scratch_model_path = 'yolact_scratch_best.pth'

    # Set dataset path
    dataset_path = 'dataset_E4888'

    # Get random sample of images once to use for both models
    images_dir = os.path.join(dataset_path, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_samples = 20
    if len(image_files) <= num_samples:
        selected_images = image_files
    else:
        selected_images = np.random.choice(image_files, num_samples, replace=False)

    # Test pretrained model
    if os.path.exists(pretrained_model_path):
        print(f"\n=== Testing model trained with pretrained weights ===")
        print(f"Loading model from {pretrained_model_path}...")
        pretrained_model = load_model(pretrained_model_path, device)

        print(f"Testing on {dataset_path}...")
        for image_file in selected_images:
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting pretrained model on {image_file}...")
            test_on_image(pretrained_model, image_path, device, suffix="_pretrained")
    else:
        print(f"Warning: Pretrained model file {pretrained_model_path} not found!")
        print("Using a stub model for demonstration...")
        pretrained_model = YOLACTLite(num_classes=3).to(device)

        print(f"Testing on {dataset_path}...")
        for image_file in selected_images:
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting pretrained model on {image_file}...")
            test_on_image(pretrained_model, image_path, device, suffix="_pretrained")

    # Test scratch model
    if os.path.exists(scratch_model_path):
        print(f"\n=== Testing model trained from scratch ===")
        print(f"Loading model from {scratch_model_path}...")
        scratch_model = load_model(scratch_model_path, device)

        print(f"Testing on {dataset_path}...")
        for image_file in selected_images:
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting scratch model on {image_file}...")
            test_on_image(scratch_model, image_path, device, suffix="_scratch")
    else:
        print(f"Warning: Scratch model file {scratch_model_path} not found!")
        print("Using a stub model for demonstration...")
        scratch_model = YOLACTLite(num_classes=3).to(device)

        print(f"Testing on {dataset_path}...")
        for image_file in selected_images:
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting scratch model on {image_file}...")
            test_on_image(scratch_model, image_path, device, suffix="_scratch")

    # Print comparative results if both models were tested
    if os.path.exists(pretrained_model_path) and os.path.exists(scratch_model_path):
        print("\n=== Results comparison ===")
        print("Check the segmentation_results directory for side-by-side comparisons of both models.")
    else:
        print("\n=== Demo Results ===")
        print("Check the segmentation_results directory for demonstration results using stub models.")


if __name__ == "__main__":
    main()