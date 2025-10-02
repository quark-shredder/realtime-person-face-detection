"""Faster R-CNN object detector using torchvision pretrained models."""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import cv2
import numpy as np
from typing import List, Tuple


# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ObjectDetector:
    """Fast object detector using Faster R-CNN MobileNetV3."""

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        max_detections: int = 50,
        device: str = "mps",
    ):
        """
        Initialize object detector.

        Args:
            confidence_threshold: Confidence threshold for detections
            max_detections: Maximum number of detections per image
            device: Device to run on ('mps', 'cuda', or 'cpu')
        """
        self.conf_thresh = confidence_threshold
        self.max_detections = max_detections
        self.class_names = COCO_CLASSES

        # Setup device
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Object detector using device: {self.device}")

        # Load pretrained model
        print("Loading Faster R-CNN MobileNetV3 (pretrained on COCO)...")
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get preprocessing transforms
        self.transforms = weights.transforms()

        print(f"✓ Object detector ready (91 COCO classes)")

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> List[Tuple[List[int], int, float, str]]:
        """
        Detect objects in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detections: (bbox, class_id, confidence, class_name)
            bbox is [x1, y1, x2, y2]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Inference
        predictions = self.model(image_tensor)[0]

        # Extract detections
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        # Filter by confidence
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.conf_thresh:
                # Convert box to integer coordinates
                x1, y1, x2, y2 = box.astype(int).tolist()

                # Get class name
                class_name = self.class_names[label] if label < len(self.class_names) else "unknown"

                # Skip background and N/A classes
                if class_name not in ['__background__', 'N/A']:
                    detections.append(([x1, y1, x2, y2], int(label), float(score), class_name))

        # Limit number of detections
        detections = detections[:self.max_detections]

        return detections
