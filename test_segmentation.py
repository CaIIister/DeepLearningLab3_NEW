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

    # Convert image to numpy for drawing
    image_np = np.array(image)
    result_img = image_np.copy()

    # Get feature dimensions
    feature_h, feature_w = predictions['cls_pred'][0].shape[2:]
    print(f"Feature map dimensions: {feature_h}x{feature_w}")

    # Directly use raw predictions for detections - SIMPLIFIED APPROACH
    # Get classification scores
    cls_preds = predictions['cls_pred'][0]  # [anchors, classes, h, w]
    box_preds = predictions['box_pred'][0]  # [anchors, 4, h, w]
    prototype_masks = predictions['prototype_masks'][0]  # [prototypes, h_mask, w_mask]
    mask_coeffs = predictions['mask_coef_pred'][0]  # [anchors, prototypes, h, w]

    # Apply sigmoid to class predictions
    cls_probs = torch.sigmoid(cls_preds)

    # Debug info
    print(f"Max classification score: {cls_probs.max().item():.4f}")

    # Non-background classes
    bg_class = 0  # Assuming 0 is background
    obj_classes = [i for i in range(cls_probs.shape[1]) if i != bg_class]

    # For simplicity, let's just get the top scoring detections
    num_anchors, num_classes, _, _ = cls_probs.shape
    num_prototypes = prototype_masks.shape[0]

    # Create detections
    detections = []

    # For simplicity, extract detections across all spatial locations
    for a in range(num_anchors):
        for c in obj_classes:
            for h in range(feature_h):
                for w in range(feature_w):
                    score = cls_probs[a, c, h, w].item()
                    if score > conf_threshold:
                        # Get box
                        box = box_preds[a, :, h, w].cpu().numpy()

                        # Get mask coefficients
                        mask_coeff = mask_coeffs[a, :, h, w]

                        # Save detection
                        detections.append({
                            'score': score,
                            'label': c,
                            'box': box,
                            'h_idx': h,
                            'w_idx': w,
                            'mask_coeff': mask_coeff
                        })

    print(f"Found {len(detections)} detections above threshold {conf_threshold}")

    # If no detections, create empty result
    if len(detections) == 0:
        # Save the original image
        plt.figure(figsize=(12, 8))
        plt.imshow(image_np)
        plt.axis('off')
        plt.title(f"No detections found above threshold {conf_threshold}")

        # Save result
        output_dir = 'segmentation_results'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}{suffix}.jpg")
        plt.savefig(output_path)
        plt.close()

        print(f"No detections found. Image saved to {output_path}")
        return

    # Sort detections by score
    detections.sort(key=lambda x: x['score'], reverse=True)

    # Colors for visualization
    colors = [
        (0, 255, 0),  # Green for first class
        (0, 0, 255),  # Blue for second class
        (255, 0, 0)  # Red for third class
    ]

    # For each detection, draw bounding box and mask
    for detection in detections:
        # Get class and score
        label = detection['label']
        score = detection['score']
        class_name = TARGET_CLASSES[label]
        color = colors[(label - 1) % len(colors)]  # Skip background

        # Scale box to original image size
        x, y, w, h = detection['box']
        input_scale_x = new_size[0] / feature_w
        input_scale_y = new_size[1] / feature_h
        scale_x = orig_size[0] / new_size[0]
        scale_y = orig_size[1] / new_size[1]

        # Convert to [x1, y1, x2, y2] format and scale to original image
        x1 = int((x - w / 2) * input_scale_x * scale_x)
        y1 = int((y - h / 2) * input_scale_y * scale_y)
        x2 = int((x + w / 2) * input_scale_x * scale_x)
        y2 = int((y + h / 2) * input_scale_y * scale_y)

        # Clip to image boundaries
        x1 = max(0, min(x1, image_np.shape[1] - 1))
        y1 = max(0, min(y1, image_np.shape[0] - 1))
        x2 = max(0, min(x2, image_np.shape[1] - 1))
        y2 = max(0, min(y2, image_np.shape[0] - 1))

        # Draw bounding box directly on the result image
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # Generate mask - use prototype masks and coefficients
        mask_coeff = detection['mask_coeff']

        # Resize prototype masks to original image size
        proto_masks_resized = F.interpolate(
            prototype_masks.unsqueeze(0),
            size=(image_np.shape[0], image_np.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Combine prototypes using coefficients
        mask = torch.zeros((image_np.shape[0], image_np.shape[1]), device=device)
        for j in range(num_prototypes):
            mask += mask_coeff[j] * proto_masks_resized[j]

        # Apply sigmoid and threshold to get binary mask
        mask = torch.sigmoid(mask) > 0.5
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Apply a color overlay on the mask region
        for c in range(3):
            # Apply RGB channels with alpha=0.5
            result_img[:, :, c] = np.where(
                mask_np == 1,
                result_img[:, :, c] * 0.5 + color[c] * 0.5,
                result_img[:, :, c]
            )

        # Draw label text with background
        label_text = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label_text, font, 0.5, 2)[0]

        # Draw filled rectangle for text background
        cv2.rectangle(result_img,
                      (x1, y1 - text_size[1] - 5),
                      (x1 + text_size[0], y1),
                      color, -1)

        # Draw text
        cv2.putText(result_img, label_text, (x1, y1 - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title(f"Segmentation Results{suffix} (Inference time: {inference_time:.3f}s)")

    # Save result
    output_dir = 'segmentation_results'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}{suffix}.jpg")
    plt.savefig(output_path)
    plt.close()

    print(f"Found {len(detections)} objects with confidence > {conf_threshold}")
    print(f"Image saved to {output_path}")
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