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
    A lightweight version of YOLACT for instance segmentation, optimized for lower memory usage.
    """

    def __init__(self, num_classes, pretrained=False, backbone='resnet18'):
        super(YOLACTLite, self).__init__()

        # Load a pre-trained backbone
        if backbone == 'resnet18':
            backbone_model = torchvision.models.resnet18(weights=None)
        elif backbone == 'resnet34':
            backbone_model = torchvision.models.resnet34(weights=None)
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


def test_on_image(model, image_path, device='cuda', conf_threshold=0.3, suffix=""):
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
    num_anchors = cls_pred.shape[0]
    num_classes = cls_pred.shape[1]
    num_prototypes = prototype_masks.shape[0]
    feature_h, feature_w = cls_pred.shape[2:]

    # Debugging: Print prediction shape info
    print(f"Feature map size: {feature_h}x{feature_w}")
    print(f"Num anchors: {num_anchors}, Num classes: {num_classes}")

    # Apply sigmoid to get probability scores
    cls_pred_sigmoid = torch.sigmoid(cls_pred)

    # Get maximum score across all anchors, features, and classes
    max_score = cls_pred_sigmoid.max().item()
    print(f"Maximum confidence score: {max_score:.4f}")

    # FIX: Better detection extraction - don't average across spatial locations
    # Reshape to [anchors, classes, h*w]
    cls_scores = cls_pred_sigmoid.reshape(num_anchors, num_classes, -1)

    # Get best score for each anchor and class combination
    best_scores, best_loc = cls_scores.max(dim=2)  # [anchors, classes]

    # Get corresponding locations
    best_h = best_loc // feature_w
    best_w = best_loc % feature_w

    # Prepare lists for detections
    all_scores = []
    all_labels = []
    all_boxes = []
    all_mask_coeffs = []

    # For each anchor, check if any class exceeds threshold
    for a in range(num_anchors):
        for c in range(num_classes):
            # Skip background class (0)
            if c == 0:
                continue

            score = best_scores[a, c].item()
            if score > conf_threshold:
                # Get corresponding box
                h_idx, w_idx = best_h[a, c].item(), best_w[a, c].item()
                box = box_pred[a, :, h_idx, w_idx]

                # Get mask coefficients
                mask_coef = mask_coef_pred[a, :, h_idx, w_idx]

                all_scores.append(score)
                all_labels.append(c)
                all_boxes.append(box)
                all_mask_coeffs.append(mask_coef)

    # Convert to tensors
    if all_scores:
        scores = torch.tensor(all_scores, device=device)
        labels = torch.tensor(all_labels, device=device)
        boxes = torch.stack(all_boxes)
        mask_coeffs = torch.stack(all_mask_coeffs)
    else:
        print(f"No detections found above threshold {conf_threshold}")

        # Create output directories
        output_dir = 'segmentation_results'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}{suffix}.jpg")

        # Save the original image with a "No detections" overlay
        plt.figure(figsize=(12, 8))
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.title(
            f"Segmentation Results{suffix} (Inference time: {inference_time:.3f}s)\nNo detections found above threshold {conf_threshold}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Image saved to {output_path}")
        return

    # Scale boxes to original image size
    # Box coordinates are in feature map scale, need to scale to input image and then to original
    scale_x = orig_size[0] / new_size[0]
    scale_y = orig_size[1] / new_size[1]
    input_scale_x = new_size[0] / feature_w
    input_scale_y = new_size[1] / feature_h

    # Adjust box format: [x1, y1, x2, y2] where x1,y1 is top-left and x2,y2 is bottom-right
    for i in range(len(boxes)):
        # Extract coordinates
        x, y, w, h = boxes[i]
        # Convert to proper box format and scale to original image
        x1 = (x - w / 2) * input_scale_x * scale_x
        y1 = (y - h / 2) * input_scale_y * scale_y
        x2 = (x + w / 2) * input_scale_x * scale_x
        y2 = (y + h / 2) * input_scale_y * scale_y
        boxes[i] = torch.tensor([x1, y1, x2, y2], device=device)

    # Convert to numpy for visualization
    image_np = np.array(image)
    h_orig, w_orig, _ = image_np.shape

    # Create overlay for visualization
    overlay = image_np.copy()

    # IMPROVED: Colors for different classes with higher alpha for better visibility
    colors = [
        (0, 255, 0, 200),  # diningtable - green with higher alpha
        (0, 0, 255, 200),  # sofa - blue with higher alpha
        (255, 0, 0, 200)  # extra color with higher alpha
    ]

    # For each detection, generate mask and visualize
    for i in range(len(scores)):
        # Get detection info
        box = boxes[i].cpu().numpy().astype(np.int32)
        score = scores[i].cpu().item()
        label = labels[i].cpu().item()

        # Ensure box coordinates are within image bounds
        box[0] = max(0, min(box[0], w_orig - 1))
        box[1] = max(0, min(box[1], h_orig - 1))
        box[2] = max(0, min(box[2], w_orig - 1))
        box[3] = max(0, min(box[3], h_orig - 1))

        # Get class name
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

        # IMPROVED: Add a dilated edge to make mask boundaries more visible
        kernel = np.ones((3, 3), np.uint8)
        mask_edge = cv2.dilate(binary_mask, kernel) - binary_mask

        # Apply mask color with improved visibility
        color = colors[(label - 1) % len(colors)]
        mask_overlay = overlay.copy()

        # Apply color directly for stronger effect
        for c in range(3):
            mask_overlay[:, :, c] = np.where(binary_mask == 1, color[c], mask_overlay[:, :, c])
            # Add contrasting border at mask edges
            overlay[:, :, c] = np.where(mask_edge == 1, 0 if color[c] > 128 else 255, overlay[:, :, c])

        # Apply mask with higher opacity
        alpha_mask = 0.7  # Increased opacity for masks
        overlay = cv2.addWeighted(overlay, 1.0 - alpha_mask, mask_overlay, alpha_mask, 0)

        # IMPROVED: Draw thicker bounding box
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color[:3], 3)

        # IMPROVED: Add label with better visibility
        label_text = f"{class_name}: {score:.2f}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Add background for text
        cv2.rectangle(overlay,
                      (box[0], box[1] - 25),
                      (box[0] + text_size[0], box[1]),
                      color[:3], -1)  # -1 fills the rectangle

        # Add text with white color for contrast
        cv2.putText(overlay, label_text, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # IMPROVED: Blend with higher alpha for better visibility
    alpha = 0.8  # Increased from 0.5
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

    # Make sure output directory exists
    os.makedirs('segmentation_results', exist_ok=True)

    # Get random sample of images once to use for both models
    images_dir = os.path.join(dataset_path, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_samples = min(20, len(image_files))

    if len(image_files) <= num_samples:
        selected_images = image_files
    else:
        # Use seed for reproducibility
        np.random.seed(42)
        selected_images = np.random.choice(image_files, num_samples, replace=False)

    # Lower threshold for testing to see if any detections appear
    conf_threshold = 0.3  # Lowered from 0.5 for testing

    # Test pretrained model
    if os.path.exists(pretrained_model_path):
        print(f"\n=== Testing model trained with pretrained weights ===")
        print(f"Loading model from {pretrained_model_path}...")
        pretrained_model = load_model(pretrained_model_path, device)

        print(f"Testing on {dataset_path} with confidence threshold {conf_threshold}...")
        for image_file in selected_images:
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting pretrained model on {image_file}...")
            test_on_image(pretrained_model, image_path, device, conf_threshold=conf_threshold, suffix="_pretrained")
    else:
        print(f"Warning: Pretrained model file {pretrained_model_path} not found!")
        print("Using a stub model for demonstration...")
        pretrained_model = YOLACTLite(num_classes=3).to(device)

        print(f"Testing on {dataset_path} with confidence threshold {conf_threshold}...")
        for i, image_file in enumerate(selected_images):
            # Only process a few images with stub model to save time
            if i >= 3:
                break
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting stub pretrained model on {image_file}...")
            test_on_image(pretrained_model, image_path, device, conf_threshold=conf_threshold,
                          suffix="_pretrained_stub")

    # Test scratch model
    if os.path.exists(scratch_model_path):
        print(f"\n=== Testing model trained from scratch ===")
        print(f"Loading model from {scratch_model_path}...")
        scratch_model = load_model(scratch_model_path, device)

        print(f"Testing on {dataset_path} with confidence threshold {conf_threshold}...")
        for image_file in selected_images:
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting scratch model on {image_file}...")
            test_on_image(scratch_model, image_path, device, conf_threshold=conf_threshold, suffix="_scratch")
    else:
        print(f"Warning: Scratch model file {scratch_model_path} not found!")
        print("Using a stub model for demonstration...")
        scratch_model = YOLACTLite(num_classes=3).to(device)

        print(f"Testing on {dataset_path} with confidence threshold {conf_threshold}...")
        for i, image_file in enumerate(selected_images):
            # Only process a few images with stub model to save time
            if i >= 3:
                break
            image_path = os.path.join(images_dir, image_file)
            print(f"\nTesting stub scratch model on {image_file}...")
            test_on_image(scratch_model, image_path, device, conf_threshold=conf_threshold, suffix="_scratch_stub")

    # Print comparative results if both models were tested
    if os.path.exists(pretrained_model_path) and os.path.exists(scratch_model_path):
        print("\n=== Results comparison ===")
        print("Check the segmentation_results directory for side-by-side comparisons of both models.")
    else:
        print("\n=== Demo Results ===")
        print("Check the segmentation_results directory for demonstration results.")


if __name__ == "__main__":
    main()