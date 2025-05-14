import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np


class VOCInstanceSegDataset(Dataset):
    """Dataset for instance segmentation using PASCAL VOC data"""

    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.transform = transform
        self.image_set = image_set

        # Define target classes
        self.classes = ['__background__', 'diningtable', 'sofa']

        # Read image IDs from split file
        with open(os.path.join(root, f'{image_set}.txt'), 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

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
        masks = []

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

            # Get class label
            class_name = obj['name']
            if class_name not in self.class_to_idx:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[class_name])

            # Create a mask for this object (simplified as a rectangle)
            # In a real implementation, you'd use instance segmentation data
            mask = torch.zeros((height, width), dtype=torch.uint8)
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
            masks.append(mask)

        # Create target dict
        target = {}

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["masks"] = torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
            target["image_id"] = torch.tensor([idx])
            target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (
                        target["boxes"][:, 3] - target["boxes"][:, 1])
            target["iscrowd"] = torch.zeros_like(target["labels"])
        else:
            # Empty image case
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)
            target["image_id"] = torch.tensor([idx])
            target["area"] = torch.zeros(0, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(0, dtype=torch.int64)

        # Apply transforms
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def parse_voc_xml(self, node):
        """Parse VOC XML annotation file"""
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