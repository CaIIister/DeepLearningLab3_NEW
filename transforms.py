import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # Resize image while maintaining aspect ratio
        w, h = image.size

        # Determine size to resize to
        min_orig = min(w, h)
        max_orig = max(w, h)

        # Scale factor to achieve min_size
        scale_factor = self.min_size / min_orig

        # Check if max dimension will exceed max_size
        if max_orig * scale_factor > self.max_size:
            scale_factor = self.max_size / max_orig

        # New dimensions
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Resize image
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Scale targets
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            boxes[:, [0, 2]] *= (new_w / w)
            boxes[:, [1, 3]] *= (new_h / h)
            target["boxes"] = boxes

        if "masks" in target and len(target["masks"]) > 0:
            masks = target["masks"]
            resized_masks = []
            for mask in masks:
                mask_pil = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                mask_resized = mask_pil.resize((new_w, new_h), Image.NEAREST)
                mask_tensor = torch.tensor(np.array(mask_resized) > 127, dtype=torch.uint8)
                resized_masks.append(mask_tensor)

            target["masks"] = torch.stack(resized_masks) if resized_masks else masks

        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image
            image = F.hflip(image)
            w, h = image.size

            # Flip boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

            # Flip masks
            if "masks" in target and len(target["masks"]) > 0:
                masks = target["masks"]
                flipped_masks = []
                for mask in masks:
                    flipped_masks.append(mask.flip(1))  # Flip horizontally
                target["masks"] = torch.stack(flipped_masks)

        return image, target


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        # Apply color jitter to image
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        image = ImageEnhance.Color(image).enhance(saturation_factor)

        # Convert to HSV, adjust hue, convert back to RGB
        image = F.adjust_hue(image, hue_factor)

        return image, target


class ToTensor:
    def __call__(self, image, target):
        # Convert image to tensor
        image = F.to_tensor(image)

        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # Normalize image
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target


def get_transform(train):
    transforms = []

    # Add fixed-size resize (e.g., 600x600)
    transforms.append(Resize((600, 600)))

    # Add other transforms
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)


class Resize:
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, image, target):
        # Get original size
        width, height = image.size

        # Resize image
        image = image.resize(self.size[::-1], Image.BILINEAR)  # PIL uses (width, height)

        # Scale bounding boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            scale_x = self.size[1] / width
            scale_y = self.size[0] / height

            # Apply scaling
            scaled_boxes = boxes.clone()
            scaled_boxes[:, 0] *= scale_x  # x1
            scaled_boxes[:, 2] *= scale_x  # x2
            scaled_boxes[:, 1] *= scale_y  # y1
            scaled_boxes[:, 3] *= scale_y  # y2

            target["boxes"] = scaled_boxes

        # Scale masks if they exist
        if "masks" in target and len(target["masks"]) > 0:
            masks = target["masks"]
            resized_masks = []

            for mask in masks:
                mask_pil = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                mask_resized = mask_pil.resize(self.size[::-1], Image.NEAREST)
                mask_tensor = torch.tensor(np.array(mask_resized) > 127, dtype=torch.uint8)
                resized_masks.append(mask_tensor)

            if resized_masks:
                target["masks"] = torch.stack(resized_masks)

        return image, target