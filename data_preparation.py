import os
import shutil
import xml.etree.ElementTree as ET
from torchvision.datasets import VOCDetection
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

# Target classes for E4888
TARGET_CLASSES = ['diningtable', 'sofa']


def prepare_dataset(output_path='dataset_E4888', year='2012', train_ratio=0.8):
    """
    Prepares a dataset containing only images with diningtable and sofa classes.
    Resizes images to save memory and creates train/val splits.
    """
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)

    print(f"Preparing dataset for classes: {TARGET_CLASSES}")

    # Download VOC dataset using torchvision
    print(f"Downloading VOC{year} dataset (if not already downloaded)...")
    voc_dataset = VOCDetection(root='./data', year=year, image_set='trainval', download=True)

    # Statistics
    total_images = 0
    class_counts = {cls: 0 for cls in TARGET_CLASSES}

    # Process dataset
    print("Filtering and optimizing dataset...")
    filtered_image_ids = []

    for idx in tqdm(range(len(voc_dataset))):
        try:
            img, annotation = voc_dataset[idx]

            # Check for target classes
            has_target_class = False
            valid_objects = []

            for obj in annotation['annotation']['object']:
                class_name = obj['name']
                if class_name in TARGET_CLASSES:
                    has_target_class = True
                    valid_objects.append(obj)
                    class_counts[class_name] += 1

            if has_target_class:
                # Get image ID
                image_id = annotation['annotation']['filename'].split('.')[0]
                filtered_image_ids.append(image_id)

                # Save resized image (to 640x640 or smaller while maintaining aspect ratio)
                img_path = os.path.join(output_path, 'images', f"{image_id}.jpg")

                # Resize image to save memory
                width, height = img.size
                max_size = 640
                if width > max_size or height > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.BILINEAR)

                # Save with optimized quality
                img.save(img_path, quality=90, optimize=True)

                # Create a new annotation with only target classes
                new_annotation = {
                    'annotation': {
                        'folder': annotation['annotation']['folder'],
                        'filename': annotation['annotation']['filename'],
                        'size': annotation['annotation']['size'],
                        'object': valid_objects
                    }
                }

                # Save annotation
                save_annotation(new_annotation, os.path.join(output_path, 'annotations', f"{image_id}.xml"),
                                img.size[0], img.size[1])

                total_images += 1

        except Exception as e:
            print(f"Error processing image {idx}: {e}")

    # Create train/val split
    random.shuffle(filtered_image_ids)
    split_idx = int(len(filtered_image_ids) * train_ratio)
    train_ids = filtered_image_ids[:split_idx]
    val_ids = filtered_image_ids[split_idx:]

    # Save split information
    with open(os.path.join(output_path, 'train.txt'), 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")

    with open(os.path.join(output_path, 'val.txt'), 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}\n")

    print("\nDataset preparation complete!")
    print(f"Total images: {total_images}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    for cls, count in class_counts.items():
        print(f"Class '{cls}': {count} instances")

    return total_images, class_counts, len(train_ids), len(val_ids)


def save_annotation(annotation_dict, output_path, width, height):
    """Creates XML annotation file with only target classes"""
    annotation = annotation_dict['annotation']

    # Create root element
    root = ET.Element('annotation')

    # Add basic info
    ET.SubElement(root, 'folder').text = annotation['folder']
    ET.SubElement(root, 'filename').text = annotation['filename']

    # Add size information
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(annotation['size']['depth'])

    # Add object information (only target classes)
    for obj in annotation['object']:
        if obj['name'] in TARGET_CLASSES:
            obj_elem = ET.SubElement(root, 'object')
            ET.SubElement(obj_elem, 'name').text = obj['name']
            ET.SubElement(obj_elem, 'pose').text = obj['pose']
            ET.SubElement(obj_elem, 'truncated').text = str(obj['truncated'])
            ET.SubElement(obj_elem, 'difficult').text = str(obj['difficult'])

            # Get bounding box
            bbox = ET.SubElement(obj_elem, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(obj['bndbox']['xmin'])
            ET.SubElement(bbox, 'ymin').text = str(obj['bndbox']['ymin'])
            ET.SubElement(bbox, 'xmax').text = str(obj['bndbox']['xmax'])
            ET.SubElement(bbox, 'ymax').text = str(obj['bndbox']['ymax'])

    # Create XML tree and save
    tree = ET.ElementTree(root)
    tree.write(output_path)


if __name__ == "__main__":
    prepare_dataset()