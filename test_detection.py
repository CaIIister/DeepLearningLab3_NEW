import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torchvision import transforms as T

# Target classes for E4888
TARGET_CLASSES = ['background', 'diningtable', 'sofa']


def create_faster_rcnn_model(num_classes=3, backbone='resnet18'):
    """Create a Faster R-CNN model with the same architecture as during training"""
    # Load backbone
    if backbone == 'resnet18':
        backbone_model = torchvision.models.resnet18(pretrained=False)
    elif backbone == 'resnet34':
        backbone_model = torchvision.models.resnet34(pretrained=False)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Extract backbone
    backbone = torch.nn.Sequential(*list(backbone_model.children())[:-2])
    backbone.out_channels = 512

    # Create anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Create RoI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Create model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=600,
        max_size=800,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=300
    )

    return model


def load_model(model_path, device='cuda'):
    """Load a trained model from disk"""
    model = create_faster_rcnn_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def test_on_image(model, image_path, device='cuda', conf_threshold=0.5, suffix=""):
    """Test the model on a single image"""
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

    # Convert back to numpy for visualization
    image_np = np.array(image)

    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter by confidence
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Visualize results
    colors = {
        1: (0, 255, 0),  # diningtable - green
        2: (0, 0, 255)  # sofa - blue
    }

    for box, score, label in zip(boxes, scores, labels):
        # Skip background class (0)
        if label == 0:
            continue

        # Convert box coordinates to integers
        box = box.astype(np.int32)

        # Draw bounding box
        color = colors.get(label, (255, 0, 0))
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Add label and score
        label_text = f"{TARGET_CLASSES[label]}: {score:.2f}"
        cv2.putText(image_np, label_text, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.axis('off')
    plt.title(f"Detection Results (Inference time: {inference_time:.3f}s)")
    plt.tight_layout()

    # Save the result
    output_dir = 'detection_results'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}{suffix}.jpg")
    plt.savefig(output_path)
    plt.close()

    print(f"Detection results saved to {output_path}")
    print(f"Found {len(boxes)} objects with confidence > {conf_threshold}")
    print(f"Inference time: {inference_time:.3f} seconds")

    return boxes, scores, labels


def test_on_dataset(model, dataset_path, num_samples=5, device='cuda'):
    """Test the model on a random sample of images from the dataset"""
    # Get list of images
    images_dir = os.path.join(dataset_path, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Select random samples
    if len(image_files) <= num_samples:
        samples = image_files
    else:
        samples = np.random.choice(image_files, num_samples, replace=False)

    # Test each sample
    results = []
    for image_file in samples:
        image_path = os.path.join(images_dir, image_file)
        print(f"\nTesting on {image_file}...")
        boxes, scores, labels = test_on_image(model, image_path, device)
        results.append({
            'image': image_file,
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })

    return results


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define model paths
    pretrained_model_path = 'faster_rcnn_pretrained_best.pth'
    scratch_model_path = 'faster_rcnn_scratch_best.pth'

    # Set dataset path
    dataset_path = 'dataset_E4888'

    # Get random sample of images once to use for both models
    images_dir = os.path.join(dataset_path, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_samples = 5
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

    # Print comparative results if both models were tested
    if os.path.exists(pretrained_model_path) and os.path.exists(scratch_model_path):
        print("\n=== Results comparison ===")
        print("Check the detection_results directory for side-by-side comparisons of both models.")


if __name__ == "__main__":
    main()