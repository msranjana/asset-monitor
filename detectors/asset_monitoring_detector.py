"""
detectors/asset_monitoring_detector.py

Municipal asset monitoring — implements the CPU cascade from the
architecture doc as one BaseDetector:

    Stage 0  Motion/change gate          MOG2 background subtraction (classical)
    Stage 1  Who is near the asset       YOLO (ultralytics, pluggable weights)
    Stage 2  Object removal/displacement embedding cosine-sim vs a reference crop
    Stage 3  Visible damage              anomaly score vs reference (pluggable model,
                                          falls back to a naive SSIM-lite score)
    Stage 4  Dwell / persistence         per-asset presence timer (loitering)
    Stage 5  Alert verification          optional VLM callable, called once per candidate
    Stage 6  View integrity              ORB homography vs the first frame

Heavy assets (YOLO weights, a trained embedder/anomaly model, a VLM) are not
bundled — pass them in as paths/callables. Any stage whose dependency is
missing logs once and is skipped rather than crashing the detector thread,
so you can wire this in before every model is ready.

Register it like the smoke/fire detector:

    from detectors.asset_monitoring_detector import AssetMonitoringDetector, AssetROI

    assets = [
        # Live baseline (recommended): omit reference_image_path; ROI crop is captured
        # from the stream after baseline_warmup_frames, preferring the quietest frame.
        AssetROI(name="bench_12", bbox=(400, 120, 620, 340)),
        # Optional approved snapshot on disk instead of (or before) live capture:
        # AssetROI(name="bench_12", bbox=(400, 120, 620, 340),
        #          reference_image_path="assets/bench_12_reference.jpg"),
    ]
    DetectionWrapper("asset_monitoring", 4, AssetMonitoringDetector(assets)),

pip install opencv-python numpy
# optional, enables Stage 1 (people near asset):
pip install ultralytics
# optional, enables a real embedder for Stage 2 instead of the histogram fallback:
pip install torch torchvision
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detectors.base_detector import BaseDetector
from engines.alert_engine import AlertType


@dataclass
class AssetROI:
    name: str
    bbox: Tuple[int, int, int, int]          # x1, y1, x2, y2 in full-frame pixels
    reference_image_path: Optional[str] = None  # if set, load baseline from disk; else live capture
    dwell_seconds: float = 15.0               # continuous nearby-person time before a loitering alert
    displacement_sim_threshold: float = 0.75  # cosine sim below this => asset moved/removed
    damage_score_threshold: float = 0.35      # anomaly score above this => possible damage


class AssetMonitoringDetector(BaseDetector):
    def __init__(
        self,
        assets: List[AssetROI],
        yolo_model_path: str = "yolov8n.pt",
        yolo_confidence: float = 0.4,
        embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        damage_model: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        vlm_verify: Optional[Callable[[np.ndarray, str], bool]] = None,
        motion_min_pixels: int = 400,
        alert_cooldown: float = 30.0,
        view_check_every_n_frames: int = 120,
        view_match_threshold: int = 25,
        baseline_warmup_frames: int = 60,
    ):
        self.assets = assets
        self.yolo_model_path = yolo_model_path
        self.yolo_confidence = yolo_confidence
        self._embedder = embedder
        self._damage_model = damage_model
        self._vlm_verify = vlm_verify
        self.motion_min_pixels = motion_min_pixels
        self.alert_cooldown = alert_cooldown
        self.view_check_every_n_frames = view_check_every_n_frames
        self.view_match_threshold = view_match_threshold
        self.baseline_warmup_frames = baseline_warmup_frames

        self._bg_subtractors: Dict[str, cv2.BackgroundSubtractorMOG2] = {}
        self._pending_live_baseline: set[str] = set()
        self._baseline_quietest_motion: Dict[str, int] = {}
        self._baseline_quietest_crop: Dict[str, np.ndarray] = {}
        self._reference_crops: Dict[str, np.ndarray] = {}
        self._reference_embeddings: Dict[str, np.ndarray] = {}
        self._dwell_started_at: Dict[str, Optional[float]] = {}
        self._last_alert_at: Dict[Tuple[str, str], float] = {}

        self._yolo = None
        self._orb = cv2.ORB_create(500)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._reference_frame = None
        self._reference_frame_kp_des = None
        self._frame_count = 0

        self._warned_no_yolo = False
        self._warned_no_embedder = False

    # ------------------------------------------------------------------ setup

    def on_start(self):
        for asset in self.assets:
            self._bg_subtractors[asset.name] = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=32, detectShadows=False
            )
            self._dwell_started_at[asset.name] = None

            if asset.reference_image_path:
                ref = cv2.imread(asset.reference_image_path)
                if ref is None:
                    self.log(f"[{asset.name}] could not read reference image at "
                              f"{asset.reference_image_path}", level="WARNING")
                    continue
                self._reference_crops[asset.name] = ref
                self._reference_embeddings[asset.name] = self._embed(ref)
            else:
                self._pending_live_baseline.add(asset.name)
                self._baseline_quietest_motion[asset.name] = 2**31 - 1

        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.yolo_model_path)
            self.log(f"loaded YOLO weights from {self.yolo_model_path}")
        except Exception as exc:
            self.log(f"YOLO unavailable ({exc}); Stage 1 (people near asset) "
                      f"and dwell/loitering will be skipped", level="WARNING")
            self._yolo = None

        if self._embedder is None:
            self._embedder = self._default_embedder()

    # ----------------------------------------------------------------- frame

    def process(self, frame: np.ndarray):
        self._frame_count += 1

        if self._reference_frame is None:
            self._reference_frame = frame.copy()
            self._reference_frame_kp_des = self._orb.detectAndCompute(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
            )
        elif self._frame_count % self.view_check_every_n_frames == 0:
            self._check_view_integrity(frame)

        people_boxes = self._detect_people(frame) if self._yolo is not None else []

        for asset in self.assets:
            crop = self._safe_crop(frame, asset.bbox)
            if crop.size == 0:
                continue

            if asset.name in self._pending_live_baseline:
                self._advance_live_baseline(asset, crop)
                if asset.name in self._pending_live_baseline:
                    continue

            self._update_dwell(asset, people_boxes)

            if not self._motion_detected(asset.name, crop):
                self.log(f"[{asset.name}] no change")
                continue

            if asset.name in self._reference_embeddings:
                if self._check_displacement(asset, crop):
                    self._raise(asset, AlertType.WARNING,
                                "possible asset displacement or removal", crop)
                    continue  # reference crop no longer meaningful until re-approved

                if self._check_damage(asset, crop):
                    self._raise(asset, AlertType.CRITICAL,
                                "possible visible damage", crop)

    # --------------------------------------------------------- live baseline

    def _advance_live_baseline(self, asset: AssetROI, crop: np.ndarray):
        """Feed MOG2 during warmup; after baseline_warmup_frames, lock the quietest ROI crop."""
        fg_mask = self._bg_subtractors[asset.name].apply(crop)
        changed = int(np.count_nonzero(fg_mask))
        if changed < self._baseline_quietest_motion[asset.name]:
            self._baseline_quietest_motion[asset.name] = changed
            self._baseline_quietest_crop[asset.name] = crop.copy()

        if self._frame_count < self.baseline_warmup_frames:
            return

        ref = self._baseline_quietest_crop.get(asset.name)
        if ref is None:
            ref = crop.copy()
        self._reference_crops[asset.name] = ref
        self._reference_embeddings[asset.name] = self._embed(ref)
        self._pending_live_baseline.discard(asset.name)
        self.log(
            f"[{asset.name}] live baseline captured after {self.baseline_warmup_frames} frames "
            f"(quietest motion in ROI: {self._baseline_quietest_motion[asset.name]} px)"
        )

    # --------------------------------------------------------------- stage 0

    def _motion_detected(self, asset_name: str, crop: np.ndarray) -> bool:
        fg_mask = self._bg_subtractors[asset_name].apply(crop)
        changed = int(np.count_nonzero(fg_mask))
        return changed >= self.motion_min_pixels

    # --------------------------------------------------------------- stage 1

    def _detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._yolo is None:
            return []
        try:
            results = self._yolo.predict(frame, conf=self.yolo_confidence, verbose=False)
        except Exception as exc:
            if not self._warned_no_yolo:
                self.log(f"YOLO inference failed ({exc}); skipping people detection",
                          level="WARNING")
                self._warned_no_yolo = True
            return []

        boxes = []
        for r in results:
            names = r.names
            for b in r.boxes:
                cls_name = names.get(int(b.cls[0]), "")
                if cls_name == "person":
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    # --------------------------------------------------------------- stage 2

    def _default_embedder(self) -> Callable[[np.ndarray], np.ndarray]:
        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.classifier = torch.nn.Identity()
            model.eval()
            preprocess = weights.transforms()

            def embed(img_bgr: np.ndarray) -> np.ndarray:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                tensor = preprocess(torch.from_numpy(img_rgb).permute(2, 0, 1)).unsqueeze(0)
                with torch.no_grad():
                    vec = model(tensor).squeeze(0).numpy()
                return vec / (np.linalg.norm(vec) + 1e-8)

            self.log("using torchvision mobilenet_v3_small as the displacement embedder")
            return embed
        except Exception:
            if not self._warned_no_embedder:
                self.log("torch/torchvision unavailable; falling back to a color-histogram "
                          "embedder for Stage 2 (less robust, no extra deps)", level="WARNING")
                self._warned_no_embedder = True
            return self._histogram_embedder

    @staticmethod
    def _histogram_embedder(img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist / (np.linalg.norm(hist) + 1e-8)

    def _embed(self, img: np.ndarray) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._default_embedder()
        return self._embedder(img)

    def _check_displacement(self, asset: AssetROI, crop: np.ndarray) -> bool:
        ref_vec = self._reference_embeddings[asset.name]
        cur_vec = self._embed(crop)
        sim = float(np.dot(ref_vec, cur_vec) /
                     (np.linalg.norm(ref_vec) * np.linalg.norm(cur_vec) + 1e-8))
        return sim < asset.displacement_sim_threshold

    # --------------------------------------------------------------- stage 3

    def _check_damage(self, asset: AssetROI, crop: np.ndarray) -> bool:
        ref = self._reference_crops[asset.name]
        if self._damage_model is not None:
            score = self._damage_model(ref, crop)
        else:
            score = 1.0 - self._ssim_lite(ref, crop)
        return score > asset.damage_score_threshold

    @staticmethod
    def _ssim_lite(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Single-scale, grayscale SSIM approximation — a stand-in until a
        trained anomaly model (e.g. EfficientAD-S) is plugged in via
        damage_model. Not a substitute for the real thing."""
        size = (128, 128)
        a = cv2.cvtColor(cv2.resize(img_a, size), cv2.COLOR_BGR2GRAY).astype(np.float64)
        b = cv2.cvtColor(cv2.resize(img_b, size), cv2.COLOR_BGR2GRAY).astype(np.float64)

        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu_a, mu_b = a.mean(), b.mean()
        var_a, var_b = a.var(), b.var()
        cov_ab = ((a - mu_a) * (b - mu_b)).mean()

        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)) / \
               ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
        return float(np.clip(ssim, -1.0, 1.0))

    # --------------------------------------------------------------- stage 4

    def _update_dwell(self, asset: AssetROI, people_boxes: List[Tuple[int, int, int, int]]):
        ax1, ay1, ax2, ay2 = asset.bbox
        near = any(self._boxes_overlap((ax1, ay1, ax2, ay2), pb) for pb in people_boxes)

        started = self._dwell_started_at[asset.name]
        now = time.time()
        if near:
            if started is None:
                self._dwell_started_at[asset.name] = now
            elif now - started >= asset.dwell_seconds:
                self._raise(asset, AlertType.WARNING,
                             f"person lingering near asset for over {asset.dwell_seconds:.0f}s",
                             self._safe_crop_from_bbox_source(asset))
                self._dwell_started_at[asset.name] = now  # reset so it doesn't spam every frame
        else:
            self._dwell_started_at[asset.name] = None

    @staticmethod
    def _boxes_overlap(box_a: Tuple[int, int, int, int],
                        box_b: Tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)

    def _safe_crop_from_bbox_source(self, asset: AssetROI) -> np.ndarray:
        # last resort image for the dwell alert if no crop is on hand
        return self._reference_crops.get(asset.name, np.zeros((10, 10, 3), dtype=np.uint8))

    # --------------------------------------------------------------- stage 5

    def _verify_with_vlm(self, asset: AssetROI, crop: np.ndarray, message: str) -> bool:
        """Returns True if the alert should still fire after VLM review.
        If no vlm_verify callable was provided, every candidate passes through."""
        if self._vlm_verify is None:
            return True
        try:
            return bool(self._vlm_verify(crop, message))
        except Exception as exc:
            self.log(f"[{asset.name}] VLM verification failed ({exc}); "
                      f"alerting anyway", level="WARNING")
            return True

    # --------------------------------------------------------------- stage 6

    def _check_view_integrity(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self._orb.detectAndCompute(gray, None)
        kp1, des1 = self._reference_frame_kp_des

        if des1 is None or des2 is None:
            self.log("view integrity check skipped (no keypoints)", level="WARNING")
            return

        matches = self._bf_matcher.match(des1, des2)
        good_matches = [m for m in matches if m.distance < 40]

        if len(good_matches) < self.view_match_threshold:
            self.alert(
                self.to_base64(frame),
                AlertType.WARNING,
                f"camera view has shifted or been tampered with "
                f"({len(good_matches)} good keypoint matches vs reference frame)",
            )

    # ------------------------------------------------------------------ util

    def _raise(self, asset: AssetROI, alert_type: AlertType, message: str, crop: np.ndarray):
        key = (asset.name, message)
        now = time.time()
        last = self._last_alert_at.get(key, 0.0)
        if now - last < self.alert_cooldown:
            return

        if not self._verify_with_vlm(asset, crop, message):
            self.log(f"[{asset.name}] {message} (suppressed by VLM verification)")
            return

        self._last_alert_at[key] = now
        self.alert(self.to_base64(crop), alert_type, f"[{asset.name}] {message}")

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]