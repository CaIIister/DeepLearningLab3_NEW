import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from yolact_complete import YOLACTComplete

# Target classes for E4888
TARGET_CLASSES = ['background', 'diningtable', 'sofa']


def load_model(model_path, device='cuda'):
    """Load a trained model from disk"""
    model = YOLACTComplete(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def test_on_image(model, image_path, device='cuda', conf_threshold=0.5, suffix=""):
    """Test the segmentation model on a single image"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height)

    # Preprocess image
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        start_time = time.time()
        results = model(image_tensor)
        inference_time = time.time() - start_time

    # Get results
    result = results[0]  # First image in batch
    boxes = result['boxes'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    labels = result['labels'].cpu().numpy()
    masks = result['masks'].cpu().numpy()

    # Filter by confidence threshold
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    masks = masks[keep]

    # Convert to numpy for visualization
    image_np = np.array(image)

    # Create overlay for visualization
    overlay = image_np.copy()

    # Colors for different classes
    colors = {
        1: (0, 255, 0, 128),  # diningtable - green with alpha
        2: (0, 0, 255, 128)  # sofa - blue with alpha
    }

    # Draw each detection
    for i in range(len(boxes)):
        # Get detection info
        box = boxes[i].astype(np.int32)
        score = scores[i]
        label = labels[i]
        mask = masks[i]

        # Skip background class
        if label == 0:
            continue

        # Get color for this class
        color = colors.get(label, (255, 0, 0, 128))

        # Apply color to mask region
        for c in range(3):
            overlay[:, :, c] = np.where(mask == 1,
                                        overlay[:, :, c] * (1 - color[3] / 255) + color[c] * (color[3] / 255),
                                        overlay[:, :, c])

        # Draw bounding box
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color[:3], 2)

        # Add label and score
        label_text = f"{TARGET_CLASSES[label]}: {score:.2f}"
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
    print(f"Found {len(boxes)} objects with confidence > {conf_threshold}")
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
        print("Check the segmentation_results directory for side-by-side comparisons of both models.")


if __name__ == "__main__":
    main()