from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.ops as ops


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def extract_visual_features(
    image_path,
    faster_rcnn,
    device,
    max_regions=36,
    score_threshold=0.5,
    spatial_scale=1 / 16.0,
):
    image_tensor = preprocess_image(image_path).to(device)  # expected shape: (C, H, W)
    _, _, height, width = image_tensor.shape  # Get dimensions directly from the tensor

    # Forward pass through Faster R-CNN to get detections
    with torch.no_grad():
        outputs = faster_rcnn(image_tensor)

    # Extract bounding boxes and their scores
    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]

    # Filter out low-confidence detections
    valid_indices = scores >= score_threshold
    boxes = boxes[valid_indices]
    num_boxes = boxes.shape[0]

    # Handle the case with no valid detections: return padded zeros
    if num_boxes == 0:
        # Determine flattened feature dimension by running the backbone once
        with torch.no_grad():
            backbone_features = faster_rcnn.backbone(image_tensor)
            feature_map = list(backbone_features.values())[-1]
            channels = feature_map.shape[1]
        flattened_dim = channels * 7 * 7
        raw_features = torch.zeros((1, max_regions, flattened_dim), device=device)
        visual_attention_mask = torch.zeros((1, max_regions), device=device)
        norm_boxes = torch.zeros((1, max_regions, 4), device=device)
        return raw_features, visual_attention_mask, norm_boxes

    # Extract backbone features
    with torch.no_grad():
        backbone_features = faster_rcnn.backbone(image_tensor)
        feature_map = list(backbone_features.values())[-1]

    # ROI Align: extract region features using the boxes and spatial_scale
    roi_pooled_features = ops.roi_align(
        feature_map, [boxes], output_size=(7, 7), spatial_scale=spatial_scale
    )  # Shape: (num_boxes, C, 7, 7)

    # Flatten ROI pooled features to shape: (num_boxes, C * 7 * 7)
    roi_flat = roi_pooled_features.view(num_boxes, -1)
    flattened_dim = roi_flat.shape[1]

    # Pad or truncate features to have exactly max_regions regions
    if num_boxes < max_regions:
        pad_size = max_regions - num_boxes
        padding_features = torch.zeros((pad_size, flattened_dim), device=device)
        raw_features = torch.cat((roi_flat, padding_features), dim=0)
    else:
        raw_features = roi_flat[:max_regions, :]
    raw_features = raw_features.unsqueeze(0)  # Shape: (1, max_regions, D)

    # Create visual attention mask: 1 for valid regions, 0 for padded regions
    visual_attention_mask = torch.ones((1, max_regions), device=device)
    if num_boxes < max_regions:
        visual_attention_mask[:, num_boxes:] = 0

    # Normalize boxes to [0, 1] relative to the original image dimensions
    norm_boxes = boxes.clone()
    norm_boxes[:, 0] /= width
    norm_boxes[:, 1] /= height
    norm_boxes[:, 2] /= width
    norm_boxes[:, 3] /= height

    # Pad or truncate the boxes to match max_regions
    if num_boxes < max_regions:
        pad_size = max_regions - num_boxes
        padding_boxes = torch.zeros((pad_size, 4), device=device)
        norm_boxes = torch.cat((norm_boxes, padding_boxes), dim=0)
    else:
        norm_boxes = norm_boxes[:max_regions, :]
    norm_boxes = norm_boxes.unsqueeze(0)  # Shape: (1, max_regions, 4)

    return raw_features, visual_attention_mask, norm_boxes
