"""
Equipment Detector — Segmentation Enhanced

Uses YOLO11m-seg for detecting construction equipment with instance segmentation.
Returns both bounding boxes AND pixel-precise segmentation masks for each
detected piece of equipment.

The segmentation masks allow the motion analyzer to compute optical flow
ONLY within equipment pixels, eliminating background interference.
"""
import logging
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
# PyTorch 2.6 sets weights_only=True by default, which breaks Ultralytics <8.3
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **dict(k, weights_only=False))

from ultralytics import YOLO

from config import Config

logger = logging.getLogger("Detector")


class EquipmentDetector:
    """Detects construction equipment using YOLO11m-seg (instance segmentation)."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.YOLO_MODEL
        self.conf_threshold = Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.IOU_THRESHOLD

        logger.info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Check if this is a segmentation model
        self.has_segmentation = "-seg" in self.model_path or "seg" in self.model_path
        if self.has_segmentation:
            logger.info("✅ Segmentation model detected — masks will be available")
        else:
            logger.info("⚠️ Detection-only model — falling back to bounding box analysis")

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
            self.is_coco_model = True
            logger.info("Using COCO class mapping — will use shape heuristics to distinguish equipment")
        else:
            self.is_coco_model = False
            logger.info(f"Using custom model mapping: {self.equipment_mapping}")

    def _classify_equipment_by_shape(
        self, bbox: list, mask: np.ndarray = None
    ) -> str:
        """
        Distinguish excavators from dump trucks using shape analysis.

        COCO only gives us 'truck' (class 7) — it can't tell an excavator
        from a dump truck. This method uses geometric features to classify:

        Excavators:
          - Wider than tall (aspect ratio > 1.2)
          - Low solidity (mask fills <70% of bbox due to arm sticking out)
          - Asymmetric horizontal mass distribution
          - Irregular contour (arm/boom protrusions)

        Dump trucks:
          - Taller or roughly square (aspect ratio < 1.3)
          - High solidity (compact rectangular body fills >70% of bbox)
          - Symmetric shape
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return "dump_truck"

        aspect_ratio = w / h  # >1 = wider than tall

        # --- Mask-based classification (more accurate) ---
        if mask is not None:
            bbox_area = w * h
            mask_roi = mask[int(y1):int(y2), int(x1):int(x2)]

            if mask_roi.size > 0:
                mask_pixels = np.sum(mask_roi > 0)
                solidity = mask_pixels / mask_roi.size if mask_roi.size > 0 else 1.0

                # Horizontal center of mass (0=left, 1=right)
                cols = np.arange(mask_roi.shape[1])
                col_weights = np.sum(mask_roi, axis=0)
                if np.sum(col_weights) > 0:
                    h_center = float(np.sum(cols * col_weights) / np.sum(col_weights))
                    h_center_normalized = h_center / mask_roi.shape[1]  # 0..1
                    h_asymmetry = abs(h_center_normalized - 0.5)  # 0=symmetric
                else:
                    h_asymmetry = 0.0

                # Vertical mass distribution
                rows = np.arange(mask_roi.shape[0])
                row_weights = np.sum(mask_roi, axis=1)
                if np.sum(row_weights) > 0:
                    v_center = float(np.sum(rows * row_weights) / np.sum(row_weights))
                    v_center_normalized = v_center / mask_roi.shape[0]  # 0=top, 1=bottom
                else:
                    v_center_normalized = 0.5

                # Contour complexity (excavators have more irregular outlines)
                mask_uint8 = (mask_roi * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contour_complexity = 0.0
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(largest, True)
                    area = cv2.contourArea(largest)
                    if area > 0:
                        # Circularity: 1.0 = perfect circle, lower = more complex
                        circularity = (4 * 3.14159 * area) / (perimeter * perimeter)
                        contour_complexity = 1.0 - circularity  # higher = more complex

                # Scoring system: higher score = more likely excavator
                excavator_score = 0.0

                # Wide aspect ratio → excavator
                if aspect_ratio > 1.5:
                    excavator_score += 0.30
                elif aspect_ratio > 1.2:
                    excavator_score += 0.15

                # Low solidity → excavator (arm sticks out of bbox)
                if solidity < 0.55:
                    excavator_score += 0.30
                elif solidity < 0.70:
                    excavator_score += 0.15

                # Asymmetric → excavator
                if h_asymmetry > 0.12:
                    excavator_score += 0.20
                elif h_asymmetry > 0.06:
                    excavator_score += 0.10

                # Complex contour → excavator
                if contour_complexity > 0.5:
                    excavator_score += 0.20
                elif contour_complexity > 0.3:
                    excavator_score += 0.10

                if excavator_score >= 0.45:
                    return "excavator"
                else:
                    return "dump_truck"

        # --- Bbox-only fallback (less accurate) ---
        # Excavators are typically wider than tall
        if aspect_ratio > 1.4:
            return "excavator"

        return "dump_truck"

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
                - mask: binary segmentation mask (H, W) if available
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

                # Get masks if available (segmentation model)
                masks = None
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())

                    # Only keep equipment classes
                    if class_id not in self.equipment_mapping:
                        continue

                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].item())

                    det = {
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "class_id": class_id,
                        "equipment_class": self.equipment_mapping[class_id],
                        "confidence": confidence,
                    }

                    # Add segmentation mask if available
                    if masks is not None and i < len(masks):
                        mask_tensor = masks.data[i]
                        mask_np = mask_tensor.cpu().numpy()

                        h, w = frame.shape[:2]
                        if mask_np.shape != (h, w):
                            mask_np = self._resize_mask(mask_np, h, w)

                        det["mask"] = mask_np
                    else:
                        det["mask"] = None

                    # If COCO model, use shape heuristics to distinguish
                    # excavators from dump trucks
                    if self.is_coco_model and det["equipment_class"] in ("dump_truck", "vehicle"):
                        det["equipment_class"] = self._classify_equipment_by_shape(
                            bbox, det.get("mask")
                        )

                    detections.append(det)

        return detections

    def detect_with_tracking(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track equipment using built-in ByteTrack.
        Returns detections with tracking IDs and segmentation masks.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dicts with tracking IDs and masks
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

                # Get masks if available
                masks = None
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks

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

                    det = {
                        "bbox": bbox,
                        "class_id": class_id,
                        "equipment_class": equipment_class,
                        "confidence": confidence,
                        "track_id": track_id,
                        "equipment_id": equipment_id,
                    }

                    # Add segmentation mask if available
                    if masks is not None and i < len(masks):
                        mask_tensor = masks.data[i]
                        mask_np = mask_tensor.cpu().numpy()

                        h, w = frame.shape[:2]
                        if mask_np.shape != (h, w):
                            mask_np = self._resize_mask(mask_np, h, w)

                        det["mask"] = mask_np
                    else:
                        det["mask"] = None

                    # If COCO model, use shape heuristics to distinguish
                    # excavators from dump trucks (override class + ID)
                    if self.is_coco_model and equipment_class in ("dump_truck", "vehicle"):
                        equipment_class = self._classify_equipment_by_shape(
                            bbox, det.get("mask")
                        )
                        det["equipment_class"] = equipment_class
                        prefix = prefix_map.get(equipment_class, "EQ")
                        det["equipment_id"] = f"{prefix}-{track_id:03d}" if track_id else f"{prefix}-UNK"

                    detections.append(det)

        return detections

    def _resize_mask(self, mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """
        Resize a segmentation mask to match the target frame dimensions.
        Uses nearest-neighbor interpolation to keep the mask binary.
        """
        resized = cv2.resize(
            mask.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return (resized > 0.5).astype(np.float32)
