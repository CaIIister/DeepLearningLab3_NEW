import random
import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class FixedResize:
    def __init__(self, size):
        """Size is a tuple of (height, width)"""
        self.size = size

    def __call__(self, image, target):
        # Get original dimensions (image should be PIL)
        orig_width, orig_height = image.size

        # New dimensions
        new_height, new_width = self.size

        # Resize image
        image = image.resize((new_width, new_height), Image.BILINEAR)

        # Scale boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()

            # Scale factors
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height

            # Apply scaling
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 3] *= scale_y  # y2

            target["boxes"] = boxes

        # Scale masks
        if "masks" in target and len(target["masks"]) > 0:
            masks = target["masks"]
            new_masks = []

            for mask in masks:
                # Convert to PIL
                mask_pil = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                # Resize
                resized_mask = mask_pil.resize((new_width, new_height), Image.NEAREST)
                # Convert back to tensor
                mask_tensor = torch.tensor(np.array(resized_mask) > 0, dtype=torch.uint8)
                new_masks.append(mask_tensor)

            if new_masks:
                target["masks"] = torch.stack(new_masks)

        return image, target


class ToTensor:
    def __call__(self, image, target):
        # Convert PIL image to tensor
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image (handle both PIL and tensor)
            if isinstance(image, torch.Tensor):
                image = image.flip(-1)
                width = image.shape[-1]
            else:  # PIL
                width, _ = image.size
                image = F.hflip(image)

            # Flip boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes

            # Flip masks
            if "masks" in target and len(target["masks"]) > 0:
                masks = target["masks"]
                flipped_masks = []
                for mask in masks:
                    flipped_masks.append(mask.flip(-1))
                target["masks"] = torch.stack(flipped_masks)

        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # Normalize image (assumes image is already tensor)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(train):
    transforms = []

    # Basic transforms for all data
    transforms.append(FixedResize((512, 512)))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # Data augmentation for training
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)