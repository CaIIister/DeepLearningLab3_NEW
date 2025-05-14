import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np


class YOLACTComplete(nn.Module):
    """
    Complete YOLACT implementation for instance segmentation
    """

    def __init__(self, num_classes=3, backbone='resnet18', pretrained=False):
        super(YOLACTComplete, self).__init__()
        self.num_classes = num_classes
        self.fpn_channels = 256
        self.num_prototypes = 32

        # Create backbone
        if backbone == 'resnet18':
            if pretrained:
                self.backbone = torchvision.models.resnet18(weights='DEFAULT')
            else:
                self.backbone = torchvision.models.resnet18(weights=None)
        elif backbone == 'resnet34':
            if pretrained:
                self.backbone = torchvision.models.resnet34(weights='DEFAULT')
            else:
                self.backbone = torchvision.models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature pyramid layers
        self.fpn_layers = self._create_fpn_layers()

        # Protonet for generating mask prototypes
        self.protonet = self._create_protonet()

        # Anchor scales and aspect ratios
        self.anchor_scales = [1.0, 1.25, 1.5]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.anchor_scales) * len(self.aspect_ratios)

        # Detection heads
        self.prediction_heads = nn.ModuleDict({
            'cls': nn.Conv2d(self.fpn_channels, self.num_anchors * self.num_classes, kernel_size=3, padding=1),
            'box': nn.Conv2d(self.fpn_channels, self.num_anchors * 4, kernel_size=3, padding=1),
            'mask': nn.Conv2d(self.fpn_channels, self.num_anchors * self.num_prototypes, kernel_size=3, padding=1)
        })

        # Initialize weights
        self._initialize_weights()

    def _create_fpn_layers(self):
        """Create Feature Pyramid Network layers"""
        # Get channels from backbone
        C2_channels = 64  # After conv1 + bn1 in ResNet18/34
        C3_channels = 128  # After layer1 in ResNet18/34
        C4_channels = 256  # After layer2 in ResNet18/34
        C5_channels = 512  # After layer3 in ResNet18/34

        # Create lateral connections
        self.lateral_C3 = nn.Conv2d(C3_channels, self.fpn_channels, kernel_size=1)
        self.lateral_C4 = nn.Conv2d(C4_channels, self.fpn_channels, kernel_size=1)
        self.lateral_C5 = nn.Conv2d(C5_channels, self.fpn_channels, kernel_size=1)

        # Create smooth layers
        self.smooth_P3 = nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.smooth_P4 = nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.smooth_P5 = nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)

        return [self.lateral_C3, self.lateral_C4, self.lateral_C5,
                self.smooth_P3, self.smooth_P4, self.smooth_P5]

    def _create_protonet(self):
        """Create the prototype mask generation network"""
        layers = [
            nn.Conv2d(self.fpn_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_prototypes, kernel_size=1)
        ]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights properly for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _extract_backbone_features(self, x):
        """Extract feature maps from different stages of the backbone"""
        # ResNet18/34 feature extraction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)  # Used for P3
        c4 = self.backbone.layer3(c3)  # Used for P4
        c5 = self.backbone.layer4(c4)  # Used for P5

        return c3, c4, c5

    def _build_fpn(self, features):
        """Build FPN from extracted features"""
        c3, c4, c5 = features

        # Lateral connections
        p5 = self.lateral_C5(c5)
        p4 = self.lateral_C4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lateral_C3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')

        # Smooth layers
        p3 = self.smooth_P3(p3)
        p4 = self.smooth_P4(p4)
        p5 = self.smooth_P5(p5)

        return p3, p4, p5

    def _generate_anchors(self, feature_maps):
        """Generate anchors for each feature map"""
        anchors = []

        for i, feature_map in enumerate(feature_maps):
            _, _, h, w = feature_map.shape

            # Base scale for this feature level
            base_size = 2 ** (i + 3) * 16  # 32, 64, 128 pixels

            # Create grid of anchor centers
            grid_x = torch.arange(0, w, device=feature_map.device)
            grid_y = torch.arange(0, h, device=feature_map.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

            # Reshape for easier broadcasting
            grid_x = grid_x.reshape(-1)
            grid_y = grid_y.reshape(-1)

            # Create anchors for each combination of scale and aspect ratio
            for scale in self.anchor_scales:
                for ratio in self.aspect_ratios:
                    # Calculate width and height of anchor
                    size = base_size * scale
                    anchor_w = size * math.sqrt(ratio)
                    anchor_h = size / math.sqrt(ratio)

                    # Generate the anchor boxes [x1, y1, x2, y2]
                    anchor_x1 = grid_x - anchor_w / 2
                    anchor_y1 = grid_y - anchor_h / 2
                    anchor_x2 = grid_x + anchor_w / 2
                    anchor_y2 = grid_y + anchor_h / 2

                    # Stack into tensor [num_anchors, 4]
                    anchor = torch.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], dim=1)
                    anchors.append(anchor)

        # Concatenate all anchors [total_num_anchors, 4]
        return torch.cat(anchors, dim=0)

    def forward(self, x, targets=None):
        """Forward pass with training and inference modes"""
        # Get image size for anchor generation
        img_size = x.shape[2:]

        # Extract backbone features
        backbone_features = self._extract_backbone_features(x)

        # Build FPN feature maps
        feature_maps = self._build_fpn(backbone_features)

        # Generate prototypes from P3 (highest resolution)
        prototype_masks = self.protonet(feature_maps[0])

        # Make predictions on each feature map
        batch_size = x.shape[0]
        predictions = {
            'cls': [],
            'box': [],
            'mask': []
        }

        for feature_map in feature_maps:
            # Apply prediction heads to each feature map
            cls_pred = self.prediction_heads['cls'](feature_map)
            box_pred = self.prediction_heads['box'](feature_map)
            mask_pred = self.prediction_heads['mask'](feature_map)

            # Reshape for interpretation
            # cls: [batch, anchors*classes, h, w] -> [batch, h, w, anchors, classes] -> [batch, h*w*anchors, classes]
            h, w = feature_map.shape[2:]
            cls_pred = cls_pred.view(batch_size, self.num_anchors, self.num_classes, h, w)
            cls_pred = cls_pred.permute(0, 3, 4, 1, 2).contiguous().view(batch_size, -1, self.num_classes)

            # box: [batch, anchors*4, h, w] -> [batch, h, w, anchors, 4] -> [batch, h*w*anchors, 4]
            box_pred = box_pred.view(batch_size, self.num_anchors, 4, h, w)
            box_pred = box_pred.permute(0, 3, 4, 1, 2).contiguous().view(batch_size, -1, 4)

            # mask: [batch, anchors*prototypes, h, w] -> [batch, h, w, anchors, prototypes] -> [batch, h*w*anchors, prototypes]
            mask_pred = mask_pred.view(batch_size, self.num_anchors, self.num_prototypes, h, w)
            mask_pred = mask_pred.permute(0, 3, 4, 1, 2).contiguous().view(batch_size, -1, self.num_prototypes)

            predictions['cls'].append(cls_pred)
            predictions['box'].append(box_pred)
            predictions['mask'].append(mask_pred)

        # Concatenate predictions from all feature maps
        for k in predictions:
            predictions[k] = torch.cat(predictions[k], dim=1)

        # Generate anchors
        anchors = self._generate_anchors(feature_maps)

        if self.training and targets is not None:
            # Calculate loss during training
            loss_dict = self.compute_loss(predictions, prototype_masks, anchors, targets)
            return loss_dict
        else:
            # For inference, do post-processing
            return self.postprocess(predictions, prototype_masks, anchors, img_size)

    def compute_loss(self, predictions, prototype_masks, anchors, targets):
        """Compute loss for YOLACT training"""
        batch_size = len(targets)
        device = prototype_masks.device

        # Initialize losses
        cls_loss = (predictions['cls'][0].sum() * 0.0)
        box_loss = (predictions['box'][0].sum() * 0.0)
        mask_loss = (predictions['mask'][0].sum() * 0.0)

        # Process each image in batch
        for b in range(batch_size):
            target = targets[b]
            target_boxes = target.get("boxes")
            target_labels = target.get("labels")
            target_masks = target.get("masks")

            if len(target_boxes) == 0:
                continue

            # Match anchors to ground truth boxes
            iou_matrix = self._calc_iou(anchors, target_boxes)

            # Determine positive and negative anchors
            max_iou, max_idx = torch.max(iou_matrix, dim=1)
            pos_mask = max_iou >= 0.5  # IoU threshold for positive match
            neg_mask = max_iou < 0.4  # IoU threshold for negative match

            # Skip if no positive matches
            if not torch.any(pos_mask):
                continue

            # Get predictions for this batch
            cls_pred = predictions['cls'][b]
            box_pred = predictions['box'][b]
            mask_pred = predictions['mask'][b]

            # For positive anchors, get corresponding target
            pos_anchors = anchors[pos_mask]
            pos_idx = max_idx[pos_mask]
            pos_target_boxes = target_boxes[pos_idx]
            pos_target_labels = target_labels[pos_idx]

            # Classification loss - Focal Loss
            pos_cls_target = F.one_hot(pos_target_labels, num_classes=self.num_classes).float()
            neg_cls_target = F.one_hot(torch.zeros_like(max_idx[neg_mask]), num_classes=self.num_classes).float()

            # Focal loss parameters
            alpha = 0.25
            gamma = 2.0

            # Positive samples focal loss
            pos_cls_pred = cls_pred[pos_mask]
            pos_cls_prob = torch.sigmoid(pos_cls_pred)
            pos_focal_weight = alpha * (1 - pos_cls_prob) ** gamma
            pos_cls_loss = F.binary_cross_entropy_with_logits(
                pos_cls_pred, pos_cls_target, reduction='none'
            ) * pos_focal_weight
            pos_cls_loss = pos_cls_loss.sum() / max(1, pos_mask.sum())

            # Negative samples focal loss
            neg_cls_pred = cls_pred[neg_mask]
            neg_cls_prob = torch.sigmoid(neg_cls_pred)
            neg_focal_weight = (1 - alpha) * neg_cls_prob ** gamma
            neg_cls_loss = F.binary_cross_entropy_with_logits(
                neg_cls_pred, neg_cls_target, reduction='none'
            ) * neg_focal_weight
            neg_cls_loss = neg_cls_loss.sum() / max(1, neg_mask.sum())

            # Combined classification loss
            cls_loss += pos_cls_loss + neg_cls_loss

            # Box regression loss - Smooth L1
            # Convert anchor format [x1, y1, x2, y2] to [cx, cy, w, h]
            pos_anchor_cxcy = torch.zeros_like(pos_anchors)
            pos_anchor_cxcy[:, 0] = (pos_anchors[:, 0] + pos_anchors[:, 2]) / 2  # cx
            pos_anchor_cxcy[:, 1] = (pos_anchors[:, 1] + pos_anchors[:, 3]) / 2  # cy
            pos_anchor_cxcy[:, 2] = pos_anchors[:, 2] - pos_anchors[:, 0]  # width
            pos_anchor_cxcy[:, 3] = pos_anchors[:, 3] - pos_anchors[:, 1]  # height

            # Convert target boxes [x1, y1, x2, y2] to anchor deltas [dx, dy, dw, dh]
            pos_target_cxcy = torch.zeros_like(pos_target_boxes)
            pos_target_cxcy[:, 0] = (pos_target_boxes[:, 0] + pos_target_boxes[:, 2]) / 2  # cx
            pos_target_cxcy[:, 1] = (pos_target_boxes[:, 1] + pos_target_boxes[:, 3]) / 2  # cy
            pos_target_cxcy[:, 2] = pos_target_boxes[:, 2] - pos_target_boxes[:, 0]  # width
            pos_target_cxcy[:, 3] = pos_target_boxes[:, 3] - pos_target_boxes[:, 1]  # height

            # Calculate regression targets
            target_dx = (pos_target_cxcy[:, 0] - pos_anchor_cxcy[:, 0]) / pos_anchor_cxcy[:, 2]
            target_dy = (pos_target_cxcy[:, 1] - pos_anchor_cxcy[:, 1]) / pos_anchor_cxcy[:, 3]
            target_dw = torch.log(pos_target_cxcy[:, 2] / pos_anchor_cxcy[:, 2])
            target_dh = torch.log(pos_target_cxcy[:, 3] / pos_anchor_cxcy[:, 3])

            target_deltas = torch.stack([target_dx, target_dy, target_dw, target_dh], dim=1)

            # Get box predictions for positive anchors
            pos_box_pred = box_pred[pos_mask]

            # Smooth L1 loss for box regression
            box_loss += F.smooth_l1_loss(pos_box_pred, target_deltas, reduction='mean')

            # Mask loss
            if target_masks is not None and len(target_masks) > 0:
                # Get mask predictions for positive anchors
                pos_mask_pred = mask_pred[pos_mask]

                # For each positive anchor, combine prototypes using mask coefficients
                for i in range(len(pos_mask_pred)):
                    coeff = pos_mask_pred[i]
                    target_mask = target_masks[pos_idx[i]]

                    # Combine prototypes
                    combined_mask = prototype_masks.permute(1, 2, 0) @ coeff

                    # Match size of target mask
                    combined_mask = F.interpolate(
                        combined_mask.unsqueeze(0).unsqueeze(0),
                        size=target_mask.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    # Binary cross entropy loss for mask
                    mask_loss += F.binary_cross_entropy_with_logits(
                        combined_mask, target_mask.float(), reduction='mean'
                    )

        # Normalize losses by batch size
        cls_loss = cls_loss / batch_size
        box_loss = box_loss / batch_size
        mask_loss = mask_loss / batch_size

        # Weight the losses
        loss_dict = {
            'loss_cls': cls_loss * 1.0,
            'loss_box': box_loss * 1.5,
            'loss_mask': mask_loss * 1.0
        }

        return loss_dict

    def _calc_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Calculate intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]

        intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Calculate union
        union = area1[:, None] + area2 - intersection

        # IoU = intersection / union
        iou = intersection / union.clamp(min=1e-6)

        return iou

    def postprocess(self, predictions, prototype_masks, anchors, img_size):
        """Post-process predictions for inference"""
        batch_size = predictions['cls'].shape[0]
        results = []

        for b in range(batch_size):
            # Get predictions for this batch
            cls_pred = predictions['cls'][b]
            box_pred = predictions['box'][b]
            mask_pred = predictions['mask'][b]

            # Apply sigmoid to classification predictions
            cls_scores = torch.sigmoid(cls_pred)

            # Get scores and classes
            scores, cls_inds = cls_scores.max(dim=1)

            # Filter by confidence threshold
            keep = scores > 0.05  # Low threshold for NMS
            scores = scores[keep]
            cls_inds = cls_inds[keep]
            box_pred = box_pred[keep]
            mask_pred = mask_pred[keep]
            keep_anchors = anchors[keep]

            if scores.numel() == 0:
                # No detections
                results.append({
                    'boxes': torch.empty((0, 4), device=cls_pred.device),
                    'labels': torch.empty(0, dtype=torch.long, device=cls_pred.device),
                    'scores': torch.empty(0, device=cls_pred.device),
                    'masks': torch.empty((0, *img_size), device=cls_pred.device)
                })
                continue

            # Convert box predictions from deltas to absolute coordinates
            # Convert anchors format [x1, y1, x2, y2] to [cx, cy, w, h]
            anchor_cxcy = torch.zeros_like(keep_anchors)
            anchor_cxcy[:, 0] = (keep_anchors[:, 0] + keep_anchors[:, 2]) / 2  # cx
            anchor_cxcy[:, 1] = (keep_anchors[:, 1] + keep_anchors[:, 3]) / 2  # cy
            anchor_cxcy[:, 2] = keep_anchors[:, 2] - keep_anchors[:, 0]  # width
            anchor_cxcy[:, 3] = keep_anchors[:, 3] - keep_anchors[:, 1]  # height

            # Apply predicted deltas [dx, dy, dw, dh] to anchors
            pred_cx = box_pred[:, 0] * anchor_cxcy[:, 2] + anchor_cxcy[:, 0]
            pred_cy = box_pred[:, 1] * anchor_cxcy[:, 3] + anchor_cxcy[:, 1]
            pred_w = torch.exp(box_pred[:, 2]) * anchor_cxcy[:, 2]
            pred_h = torch.exp(box_pred[:, 3]) * anchor_cxcy[:, 3]

            # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
            pred_boxes = torch.zeros_like(keep_anchors)
            pred_boxes[:, 0] = pred_cx - pred_w / 2  # x1
            pred_boxes[:, 1] = pred_cy - pred_h / 2  # y1
            pred_boxes[:, 2] = pred_cx + pred_w / 2  # x2
            pred_boxes[:, 3] = pred_cy + pred_h / 2  # y2

            # Clip boxes to image boundaries
            pred_boxes[:, 0].clamp_(min=0, max=img_size[1])
            pred_boxes[:, 1].clamp_(min=0, max=img_size[0])
            pred_boxes[:, 2].clamp_(min=0, max=img_size[1])
            pred_boxes[:, 3].clamp_(min=0, max=img_size[0])

            # Apply NMS
            keep = torchvision.ops.nms(pred_boxes, scores, iou_threshold=0.5)

            # Keep top-k after NMS
            keep = keep[:100]  # Maximum 100 detections

            # Get final predictions
            final_boxes = pred_boxes[keep]
            final_scores = scores[keep]
            final_cls = cls_inds[keep]
            final_mask_coeff = mask_pred[keep]

            # Resize prototype masks to image size
            proto_masks = F.interpolate(
                prototype_masks.unsqueeze(0),
                size=img_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Generate instance masks
            instance_masks = torch.zeros((len(keep), *img_size), device=prototype_masks.device)

            for i in range(len(keep)):
                # Combine prototypes using mask coefficients
                coeff = final_mask_coeff[i]
                mask = (proto_masks.permute(1, 2, 0) @ coeff).sigmoid()

                # Threshold to get binary mask
                instance_masks[i] = (mask > 0.5).float()

            # Store results
            results.append({
                'boxes': final_boxes,
                'labels': final_cls,
                'scores': final_scores,
                'masks': instance_masks
            })

        return results