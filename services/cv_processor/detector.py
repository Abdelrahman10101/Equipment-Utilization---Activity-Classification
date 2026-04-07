"""
Equipment Detector

Uses YOLOv8 for detecting construction equipment in video frames.
Supports both COCO-pretrained models (trucks) and custom-trained models
(excavators, loaders, etc.).
"""
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
# PyTorch 2.6 sets weights_only=True by default, which breaks Ultralytics <8.3
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **dict(k, weights_only=False))

from ultralytics import YOLO

from config import Config

logger = logging.getLogger("Detector")


class EquipmentDetector:
    """Detects construction equipment using YOLOv8."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.YOLO_MODEL
        self.conf_threshold = Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.IOU_THRESHOLD

        logger.info(f"Loading YOLOv8 model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Get class names from the model
        self.class_names = self.model.names
        logger.info(f"Model classes: {self.class_names}")

        # Build equipment class mapping
        self._build_class_mapping()

    def _build_class_mapping(self):
        """Build mapping from model class IDs to equipment types."""
        self.equipment_mapping = {}

        # Check if this is a custom model with equipment-specific classes
        custom_keywords = {
            "excavator": "excavator",
            "loader": "loader",
            "backhoe": "excavator",
            "dump_truck": "dump_truck",
            "dump truck": "dump_truck",
            "bulldozer": "bulldozer",
            "crane": "crane",
            "concrete_mixer": "concrete_mixer",
            "roller": "roller",
            "grader": "grader",
        }

        is_custom_model = False
        for class_id, class_name in self.class_names.items():
            name_lower = class_name.lower().replace(" ", "_")
            for keyword, equipment_type in custom_keywords.items():
                if keyword in name_lower:
                    self.equipment_mapping[class_id] = equipment_type
                    is_custom_model = True

        # If no custom classes found, use COCO mapping
        if not is_custom_model:
            self.equipment_mapping = Config.COCO_EQUIPMENT_CLASSES.copy()
            logger.info("Using COCO class mapping for equipment detection")
        else:
            logger.info(f"Using custom model mapping: {self.equipment_mapping}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect equipment in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of detection dicts with keys:
                - bbox: [x1, y1, x2, y2] absolute coordinates
                - class_id: model class ID
                - equipment_class: mapped equipment type string
                - confidence: detection confidence
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())

                    # Only keep equipment classes
                    if class_id not in self.equipment_mapping:
                        continue

                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].item())

                    detections.append({
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "class_id": class_id,
                        "equipment_class": self.equipment_mapping[class_id],
                        "confidence": confidence,
                    })

        return detections

    def detect_with_tracking(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track equipment using built-in ByteTrack.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dicts with tracking IDs
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())

                    if class_id not in self.equipment_mapping:
                        continue

                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].item())

                    # Get tracking ID
                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].item())

                    equipment_class = self.equipment_mapping[class_id]

                    # Generate a readable equipment ID
                    prefix_map = {
                        "excavator": "EX",
                        "dump_truck": "DT",
                        "loader": "LD",
                        "bulldozer": "BD",
                        "crane": "CR",
                        "vehicle": "VH",
                        "concrete_mixer": "CM",
                        "roller": "RL",
                        "grader": "GR",
                    }
                    prefix = prefix_map.get(equipment_class, "EQ")
                    equipment_id = f"{prefix}-{track_id:03d}" if track_id else f"{prefix}-UNK"

                    detections.append({
                        "bbox": bbox,
                        "class_id": class_id,
                        "equipment_class": equipment_class,
                        "confidence": confidence,
                        "track_id": track_id,
                        "equipment_id": equipment_id,
                    })

        return detections
