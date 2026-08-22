"""Core video-analysis engine for the Exam Surveillance AI Review prototype.

Author: Hyunjun Yi
Mentor: Dr. Qingyang Xiao

The module flags reviewable events. It does NOT make a final cheating determination.
"""
from __future__ import annotations

import html
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "Ultralytics"))

import cv2
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    YOLO = None
    HAS_YOLO = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

# Newer MediaPipe builds can expose Tasks while omitting the historical
# mp.solutions namespace. Import Tasks through both supported access paths.
mp_tasks_python = None
mp_tasks_vision = None
if HAS_MEDIAPIPE and mp is not None:
    try:
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            mp_tasks_python = mp.tasks
            mp_tasks_vision = mp.tasks.vision
        else:
            from mediapipe.tasks import python as mp_tasks_python  # type: ignore
            from mediapipe.tasks.python import vision as mp_tasks_vision  # type: ignore
    except Exception:
        mp_tasks_python = None
        mp_tasks_vision = None

MEDIAPIPE_LEGACY_AVAILABLE = bool(
    HAS_MEDIAPIPE
    and mp is not None
    and hasattr(mp, "solutions")
    and hasattr(mp.solutions, "face_mesh")
)
MEDIAPIPE_TASKS_AVAILABLE = bool(
    HAS_MEDIAPIPE
    and mp is not None
    and mp_tasks_python is not None
    and mp_tasks_vision is not None
    and hasattr(mp_tasks_vision, "FaceLandmarker")
)

AUTHOR = "Hyunjun Yi"
MENTOR = "Dr. Qingyang Xiao"

SUSPICIOUS_OBJECT_LABELS = {
    "cell phone": "cell_phone_visible",
    "laptop": "disallowed_computer_or_screen_visible",
    "tv": "disallowed_computer_or_screen_visible",
    "keyboard": "online_search_proxy_visible",
    "mouse": "online_search_proxy_visible",
    "remote": "online_search_proxy_visible",
}

SCREEN_LIKE_LABELS = {"laptop", "tv", "cell phone"}
DEVICE_PROXY_LABELS = {"cell phone", "laptop", "tv", "keyboard", "mouse", "remote"}

PRETTY_FLAG_NAMES = {
    "multiple_people": "multiple people",
    "cell_phone_visible": "cell phone visible",
    "disallowed_computer_or_screen_visible": "disallowed computer/screen visible",
    "more_than_allowed_screen_devices_visible": "more than allowed screen/device count",
    "speaking_like_mouth_motion": "speaking-like mouth motion",
    "prewritten_scratch_sheet_possible": "possible pre-written scratch sheet",
    "scratch_sheet_content_changed": "scratch sheet content changed",
    "online_search_proxy_visible": "online-search proxy: device/input hardware visible",
}

FLAG_WEIGHTS = {
    "multiple_people": 0.25,
    "cell_phone_visible": 0.30,
    "disallowed_computer_or_screen_visible": 0.20,
    "more_than_allowed_screen_devices_visible": 0.20,
    "speaking_like_mouth_motion": 0.12,
    "prewritten_scratch_sheet_possible": 0.18,
    "scratch_sheet_content_changed": 0.16,
    "online_search_proxy_visible": 0.12,
}


@dataclass
class ExamProctorConfig:
    # YOLO model selection. The loader tries candidates in order.
    yolo_model_candidates: Tuple[str, ...] = ("yolo26n.pt", "yolo11n.pt", "yolov8n.pt")
    yolo_conf: float = 0.25
    yolo_iou: float = 0.50
    yolo_imgsz: int = 640
    yolo_device: Optional[str] = None

    # Performance controls.
    sample_every_n_frames: int = 3
    max_frames: Optional[int] = 900

    # Exam policy controls.
    allowed_people: int = 1
    allowed_screen_like_count: int = 1
    flag_cell_phone: bool = True
    flag_laptop: bool = True
    # A TV/monitor proxy is not individually suspicious by default because one
    # primary monitor may be permitted. Extra screen-like devices are still flagged.
    flag_tv_as_disallowed: bool = False
    # Camera-only online-search detection is a weak proxy and is opt-in.
    flag_online_search_proxy: bool = False
    speaking_enabled: bool = True

    # Output.
    output_dir: str = "/tmp/proctor_outputs"
    convert_to_h264: bool = True
    draw_all_detections: bool = True
    preserve_original_audio: bool = True

    # Box filtering.
    min_box_area_ratio: float = 0.002

    # Event smoothing / minimum durations.
    event_max_gap_sec: float = 0.70
    multiple_people_min_duration_sec: float = 1.00
    phone_min_duration_sec: float = 0.40
    device_min_duration_sec: float = 0.50
    screen_count_min_duration_sec: float = 1.00
    speaking_min_duration_sec: float = 0.80
    paper_min_duration_sec: float = 1.00

    # Speaking-like mouth movement heuristic.
    mouth_baseline_seconds: float = 2.00
    speaking_mar_min: float = 0.080
    speaking_mar_margin: float = 0.030

    # MediaPipe Tasks model fallback for releases without mp.solutions.
    face_landmarker_model_path: str = str(Path(tempfile.gettempdir()) / "face_landmarker.task")
    face_landmarker_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )

    # Optional scratch-paper ROI.
    paper_roi: Optional[Tuple[float, float, float, float]] = None
    auto_detect_paper_roi: bool = False
    paper_min_area_ratio: float = 0.020
    paper_baseline_seconds: float = 3.00
    prewritten_ink_density_threshold: float = 0.025
    ink_density_increase_threshold: float = 0.015

    risk_score_threshold: float = 0.50


def runtime_status() -> Dict[str, Any]:
    return {
        "yolo_available": HAS_YOLO,
        "mediapipe_available": HAS_MEDIAPIPE,
        "mediapipe_legacy_face_mesh": MEDIAPIPE_LEGACY_AVAILABLE,
        "mediapipe_tasks_face_landmarker": MEDIAPIPE_TASKS_AVAILABLE,
        "mediapipe_version": getattr(mp, "__version__", None) if mp is not None else None,
        "opencv_version": getattr(cv2, "__version__", None),
    }


def label_from_model_names(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def box_area_ratio(xyxy: Iterable[float], frame_shape: Tuple[int, int, int]) -> float:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    h, w = frame_shape[:2]
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / max(1.0, float(w * h))


def load_yolo_model(cfg: ExamProctorConfig):
    if not HAS_YOLO or YOLO is None:
        raise RuntimeError("Ultralytics YOLO is unavailable. Install dependencies from requirements.txt.")
    errors: List[str] = []
    seen: set[str] = set()
    for model_name in cfg.yolo_model_candidates:
        if model_name in seen:
            continue
        seen.add(model_name)
        try:
            model = YOLO(model_name)
            return model, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc!r}")
    raise RuntimeError("Could not load any configured YOLO model:\n" + "\n".join(errors))


def parse_yolo_boxes(result: Any, frame_shape: Tuple[int, int, int], cfg: ExamProctorConfig) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    names = getattr(result, "names", None)
    if names is None and hasattr(result, "model"):
        names = getattr(result.model, "names", None)
    if not hasattr(result, "boxes") or result.boxes is None:
        return boxes

    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
        conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
        cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
        label = label_from_model_names(names, cls_id)
        area_ratio = box_area_ratio(xyxy, frame_shape)
        if area_ratio < cfg.min_box_area_ratio:
            continue
        boxes.append(
            {
                "label": label,
                "cls_id": cls_id,
                "conf": conf,
                "xyxy": xyxy,
                "area_ratio": area_ratio,
            }
        )
    return boxes


class LegacyFaceMeshAdapter:
    def __init__(self) -> None:
        self.detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame: np.ndarray):
        return self.detector.process(rgb_frame)

    def close(self) -> None:
        self.detector.close()


class TasksFaceMeshAdapter:
    """Compatibility wrapper for the newer MediaPipe Tasks Face Landmarker API."""

    def __init__(self, cfg: ExamProctorConfig) -> None:
        model_path = Path(cfg.face_landmarker_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists() or model_path.stat().st_size < 100_000:
            urllib.request.urlretrieve(cfg.face_landmarker_model_url, str(model_path))

        vision = mp_tasks_vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = -1

    def process(self, rgb_frame: np.ndarray):
        self._timestamp_ms += 33
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        compatible_faces = [
            SimpleNamespace(landmark=face_landmarks)
            for face_landmarks in getattr(result, "face_landmarks", [])
        ]
        return SimpleNamespace(multi_face_landmarks=compatible_faces)

    def close(self) -> None:
        self.detector.close()


def make_face_mesh(cfg: ExamProctorConfig):
    if not cfg.speaking_enabled:
        return None
    if not HAS_MEDIAPIPE or mp is None:
        return None
    if MEDIAPIPE_LEGACY_AVAILABLE:
        try:
            return LegacyFaceMeshAdapter()
        except Exception:
            pass
    if MEDIAPIPE_TASKS_AVAILABLE:
        try:
            return TasksFaceMeshAdapter(cfg)
        except Exception:
            pass
    return None


def mouth_aspect_ratio_from_facemesh(face_landmarks: Any, frame_shape: Tuple[int, int, int]) -> Optional[float]:
    h, w = frame_shape[:2]
    lm = face_landmarks.landmark
    needed = [13, 14, 78, 308]
    if max(needed) >= len(lm):
        return None
    upper = np.array([lm[13].x * w, lm[13].y * h], dtype=float)
    lower = np.array([lm[14].x * w, lm[14].y * h], dtype=float)
    left = np.array([lm[78].x * w, lm[78].y * h], dtype=float)
    right = np.array([lm[308].x * w, lm[308].y * h], dtype=float)
    vertical = float(np.linalg.norm(upper - lower))
    horizontal = float(np.linalg.norm(left - right))
    if horizontal <= 1e-6:
        return None
    return vertical / horizontal


def resolve_roi(
    frame_shape: Tuple[int, int, int],
    roi: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi
    if all(0.0 <= float(v) <= 1.0 for v in roi):
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def auto_detect_paper_roi(frame: np.ndarray, cfg: ExamProctorConfig) -> Optional[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 150], dtype=np.uint8)
    upper = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area_ratio = (bw * bh) / max(1, w * h)
        if area_ratio < cfg.paper_min_area_ratio or area_ratio > 0.65:
            continue
        aspect = bw / max(1, bh)
        if 0.35 <= aspect <= 3.2:
            candidates.append((area_ratio, (x, y, x + bw, y + bh)))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def estimate_ink_density(frame: np.ndarray, roi_abs: Tuple[int, int, int, int]) -> Optional[float]:
    x1, y1, x2, y2 = roi_abs
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    min_dim = min(gray.shape[:2])
    block_size = 31 if min_dim >= 31 else max(3, (min_dim // 2) * 2 - 1)
    if block_size < 3:
        return None
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 12
    )
    kernel = np.ones((2, 2), np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    return float(np.count_nonzero(ink)) / float(ink.size)


def safe_median(values: List[float], default: Optional[float] = None) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return default
    return float(statistics.median(clean))


def event_min_duration(flag: str, cfg: ExamProctorConfig) -> float:
    mapping = {
        "multiple_people": cfg.multiple_people_min_duration_sec,
        "cell_phone_visible": cfg.phone_min_duration_sec,
        "disallowed_computer_or_screen_visible": cfg.device_min_duration_sec,
        "more_than_allowed_screen_devices_visible": cfg.screen_count_min_duration_sec,
        "speaking_like_mouth_motion": cfg.speaking_min_duration_sec,
        "prewritten_scratch_sheet_possible": cfg.paper_min_duration_sec,
        "scratch_sheet_content_changed": cfg.paper_min_duration_sec,
        "online_search_proxy_visible": cfg.device_min_duration_sec,
    }
    return mapping.get(flag, 0.5)


def compute_risk_score(flags: Dict[str, bool]) -> float:
    total_weight = sum(FLAG_WEIGHTS.values())
    if total_weight <= 0:
        return 0.0
    score = sum(FLAG_WEIGHTS.get(flag, 0.0) for flag, active in flags.items() if active)
    return float(min(1.0, score / total_weight))


def aggregate_events(frame_log: pd.DataFrame, cfg: ExamProctorConfig) -> pd.DataFrame:
    columns = ["flag", "label", "start_sec", "end_sec", "duration_sec", "max_risk_score", "frames"]
    if frame_log.empty:
        return pd.DataFrame(columns=columns)

    flag_cols = [c for c in frame_log.columns if c.startswith("flag_")]
    events: List[Dict[str, Any]] = []

    for col in flag_cols:
        flag = col.replace("flag_", "")
        min_duration = event_min_duration(flag, cfg)
        active = False
        start_t: Optional[float] = None
        last_true_t: Optional[float] = None
        max_risk = 0.0
        frame_count = 0

        for _, row in frame_log.iterrows():
            t = float(row["time_sec"])
            is_true = bool(row[col])
            if is_true:
                if not active:
                    active = True
                    start_t = t
                    max_risk = 0.0
                    frame_count = 0
                last_true_t = t
                max_risk = max(max_risk, float(row.get("risk_score", 0.0)))
                frame_count += 1
            elif active and last_true_t is not None and (t - last_true_t) > cfg.event_max_gap_sec:
                end_t = last_true_t
                duration = end_t - float(start_t)
                if duration >= min_duration:
                    events.append(
                        {
                            "flag": flag,
                            "label": PRETTY_FLAG_NAMES.get(flag, flag),
                            "start_sec": round(float(start_t), 2),
                            "end_sec": round(float(end_t), 2),
                            "duration_sec": round(float(duration), 2),
                            "max_risk_score": round(float(max_risk), 3),
                            "frames": int(frame_count),
                        }
                    )
                active = False
                start_t = None
                last_true_t = None

        if active and last_true_t is not None and start_t is not None:
            duration = float(last_true_t) - float(start_t)
            if duration >= min_duration:
                events.append(
                    {
                        "flag": flag,
                        "label": PRETTY_FLAG_NAMES.get(flag, flag),
                        "start_sec": round(float(start_t), 2),
                        "end_sec": round(float(last_true_t), 2),
                        "duration_sec": round(float(duration), 2),
                        "max_risk_score": round(float(max_risk), 3),
                        "frames": int(frame_count),
                    }
                )

    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(events).sort_values(["start_sec", "flag"]).reset_index(drop=True)


def draw_text_panel(frame: np.ndarray, lines: List[str], origin: Tuple[int, int] = (10, 25)) -> None:
    x, y = origin
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_h = 24
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, tw)
    panel_h = line_h * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 20), (x + max_w + 10, y - 20 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_box(frame: np.ndarray, box: Dict[str, Any]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
    label = box["label"]
    conf = box.get("conf", 0.0)
    color = (120, 220, 120)
    if label in DEVICE_PROXY_LABELS:
        color = (0, 180, 255)
    if label == "cell phone":
        color = (0, 0, 255)
    elif label == "person":
        color = (255, 180, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{label} {conf:.2f}",
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )


def annotate_frame(frame: np.ndarray, overlay_state: Dict[str, Any], cfg: ExamProctorConfig) -> np.ndarray:
    out = frame.copy()
    boxes = overlay_state.get("boxes", [])
    flags = overlay_state.get("flags", {})
    metrics = overlay_state.get("metrics", {})
    risk_score = overlay_state.get("risk_score", 0.0)

    for box in boxes:
        if cfg.draw_all_detections or box.get("label") in DEVICE_PROXY_LABELS or box.get("label") == "person":
            draw_box(out, box)

    roi_abs = overlay_state.get("paper_roi_abs")
    if roi_abs is not None:
        x1, y1, x2, y2 = roi_abs
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(out, "scratch-paper ROI", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 2, cv2.LINE_AA)

    active_labels = [PRETTY_FLAG_NAMES.get(k, k) for k, v in flags.items() if v]
    active_text = "; ".join(active_labels[:3])
    if len(active_labels) > 3:
        active_text += f"; +{len(active_labels) - 3} more"

    lines = [
        f"Review score: {risk_score:.2f}",
        f"People: {metrics.get('person_count', 0)} | Devices: {metrics.get('device_count', 0)} | Screens: {metrics.get('screen_like_count', 0)}",
    ]
    if metrics.get("mouth_aspect_ratio") is not None:
        threshold = metrics.get("speaking_threshold")
        lines.append(f"Mouth ratio: {metrics.get('mouth_aspect_ratio'):.3f} | threshold: {float(threshold):.3f}")
    if metrics.get("paper_ink_density") is not None:
        baseline = metrics.get("paper_baseline_density")
        if baseline is not None:
            lines.append(f"Paper ink: {metrics.get('paper_ink_density'):.3f} | baseline: {float(baseline):.3f}")
        else:
            lines.append(f"Paper ink: {metrics.get('paper_ink_density'):.3f} | baseline: collecting")
    if active_text:
        lines.append("FLAGS: " + active_text)
    draw_text_panel(out, lines)
    return out


def convert_video_to_h264(
    raw_video_path: Path,
    output_path: Path,
    audio_source: Optional[Path] = None,
) -> Path:
    if shutil.which("ffmpeg") is None:
        return raw_video_path

    if audio_source is not None and audio_source.exists():
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_video_path), "-i", str(audio_source),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-shortest", str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_video_path), "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output_path),
        ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_path if result.returncode == 0 and output_path.exists() else raw_video_path


def _sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return cleaned[:100] or "exam_video"


def process_video(
    video_path: str,
    cfg: ExamProctorConfig,
    model: Any,
    model_name: str = "",
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Dict[str, Any]:
    video_path = str(video_path)
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(video_path)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_stem(input_path.name)
    raw_annotated_path = output_dir / f"{stem}_annotated_raw.mp4"
    annotated_path = output_dir / f"{stem}_annotated.mp4"
    frame_log_path = output_dir / f"{stem}_frame_log.csv"
    events_csv_path = output_dir / f"{stem}_events.csv"
    events_json_path = output_dir / f"{stem}_events.json"
    report_html_path = output_dir / f"{stem}_review_report.html"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}")

    process_limit = min(total_frames, cfg.max_frames) if total_frames > 0 and cfg.max_frames is not None else (cfg.max_frames or total_frames or None)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_annotated_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open the output VideoWriter.")

    face_mesh = make_face_mesh(cfg)
    mouth_baseline_values: List[float] = []
    paper_baseline_values: List[float] = []
    paper_roi_abs: Optional[Tuple[int, int, int, int]] = None
    frame_records: List[Dict[str, Any]] = []
    last_overlay_state: Dict[str, Any] = {
        "boxes": [], "flags": {}, "metrics": {}, "risk_score": 0.0, "paper_roi_abs": None
    }
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if cfg.max_frames is not None and frame_idx >= cfg.max_frames:
                break

            t_sec = frame_idx / fps
            should_infer = frame_idx % max(1, cfg.sample_every_n_frames) == 0

            if should_infer:
                if paper_roi_abs is None:
                    paper_roi_abs = resolve_roi(frame.shape, cfg.paper_roi)
                    if paper_roi_abs is None and cfg.auto_detect_paper_roi:
                        paper_roi_abs = auto_detect_paper_roi(frame, cfg)

                predict_kwargs: Dict[str, Any] = {
                    "conf": cfg.yolo_conf,
                    "iou": cfg.yolo_iou,
                    "imgsz": cfg.yolo_imgsz,
                    "verbose": False,
                }
                if cfg.yolo_device is not None:
                    predict_kwargs["device"] = cfg.yolo_device
                result = model.predict(frame, **predict_kwargs)[0]
                boxes = parse_yolo_boxes(result, frame.shape, cfg)

                labels = [b["label"] for b in boxes]
                person_boxes = [b for b in boxes if b["label"] == "person"]
                device_boxes = [b for b in boxes if b["label"] in DEVICE_PROXY_LABELS]
                screen_like_boxes = [b for b in boxes if b["label"] in SCREEN_LIKE_LABELS]
                phone_boxes = [b for b in boxes if b["label"] == "cell phone"]
                laptop_boxes = [b for b in boxes if b["label"] == "laptop"]
                tv_boxes = [b for b in boxes if b["label"] == "tv"]
                input_proxy_boxes = [b for b in boxes if b["label"] in {"keyboard", "mouse", "remote"}]

                mouth_aspect_ratio = None
                speaking_threshold = cfg.speaking_mar_min
                speaking_like = False
                if face_mesh is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_result = face_mesh.process(rgb)
                    if getattr(face_result, "multi_face_landmarks", None):
                        mouth_aspect_ratio = mouth_aspect_ratio_from_facemesh(face_result.multi_face_landmarks[0], frame.shape)
                        if mouth_aspect_ratio is not None and t_sec <= cfg.mouth_baseline_seconds:
                            mouth_baseline_values.append(mouth_aspect_ratio)
                        baseline_mar = safe_median(mouth_baseline_values, cfg.speaking_mar_min)
                        speaking_threshold = max(cfg.speaking_mar_min, float(baseline_mar or 0.0) + cfg.speaking_mar_margin)
                        speaking_like = bool(
                            mouth_aspect_ratio is not None
                            and t_sec > cfg.mouth_baseline_seconds
                            and mouth_aspect_ratio > speaking_threshold
                        )

                paper_ink_density = None
                paper_baseline_density = None
                prewritten_scratch = False
                scratch_changed = False
                if paper_roi_abs is not None:
                    paper_ink_density = estimate_ink_density(frame, paper_roi_abs)
                    if paper_ink_density is not None and t_sec <= cfg.paper_baseline_seconds:
                        paper_baseline_values.append(paper_ink_density)
                    paper_baseline_density = safe_median(paper_baseline_values, None)
                    if paper_baseline_density is not None and t_sec > cfg.paper_baseline_seconds:
                        prewritten_scratch = bool(paper_baseline_density > cfg.prewritten_ink_density_threshold)
                        scratch_changed = bool(
                            paper_ink_density is not None
                            and (paper_ink_density - paper_baseline_density) > cfg.ink_density_increase_threshold
                        )

                disallowed_computer = bool(
                    (cfg.flag_laptop and len(laptop_boxes) > 0)
                    or (cfg.flag_tv_as_disallowed and len(tv_boxes) > 0)
                )
                online_proxy = bool(
                    cfg.flag_online_search_proxy
                    and (len(laptop_boxes) > 0 or len(tv_boxes) > 0 or len(input_proxy_boxes) > 0)
                )
                flags = {
                    "multiple_people": len(person_boxes) > cfg.allowed_people,
                    "cell_phone_visible": cfg.flag_cell_phone and len(phone_boxes) > 0,
                    "disallowed_computer_or_screen_visible": disallowed_computer,
                    "more_than_allowed_screen_devices_visible": len(screen_like_boxes) > cfg.allowed_screen_like_count,
                    "speaking_like_mouth_motion": cfg.speaking_enabled and speaking_like,
                    "prewritten_scratch_sheet_possible": prewritten_scratch,
                    "scratch_sheet_content_changed": scratch_changed,
                    "online_search_proxy_visible": online_proxy,
                }
                risk_score = compute_risk_score(flags)
                object_counts = {label: labels.count(label) for label in sorted(set(labels))}
                metrics = {
                    "person_count": len(person_boxes),
                    "device_count": len(device_boxes),
                    "screen_like_count": len(screen_like_boxes),
                    "mouth_aspect_ratio": mouth_aspect_ratio,
                    "speaking_threshold": speaking_threshold,
                    "paper_ink_density": paper_ink_density,
                    "paper_baseline_density": paper_baseline_density,
                    "object_counts": object_counts,
                }
                last_overlay_state = {
                    "boxes": boxes,
                    "flags": flags,
                    "metrics": metrics,
                    "risk_score": risk_score,
                    "paper_roi_abs": paper_roi_abs,
                }

            flags = last_overlay_state.get("flags", {})
            metrics = last_overlay_state.get("metrics", {})
            risk_score = float(last_overlay_state.get("risk_score", 0.0))
            record: Dict[str, Any] = {
                "video": input_path.name,
                "frame": int(frame_idx),
                "time_sec": round(float(t_sec), 3),
                "risk_score": round(risk_score, 4),
                "person_count": int(metrics.get("person_count", 0) or 0),
                "device_count": int(metrics.get("device_count", 0) or 0),
                "screen_like_count": int(metrics.get("screen_like_count", 0) or 0),
                "mouth_aspect_ratio": metrics.get("mouth_aspect_ratio"),
                "speaking_threshold": metrics.get("speaking_threshold"),
                "paper_ink_density": metrics.get("paper_ink_density"),
                "paper_baseline_density": metrics.get("paper_baseline_density"),
                "object_counts_json": json.dumps(metrics.get("object_counts", {})),
            }
            for flag_name in PRETTY_FLAG_NAMES:
                record[f"flag_{flag_name}"] = bool(flags.get(flag_name, False))
            frame_records.append(record)

            writer.write(annotate_frame(frame, last_overlay_state, cfg))
            frame_idx += 1
            if progress_callback is not None:
                progress_callback(frame_idx, process_limit)
    finally:
        cap.release()
        writer.release()
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass

    frame_log = pd.DataFrame(frame_records)
    events_df = aggregate_events(frame_log, cfg)
    frame_log.to_csv(frame_log_path, index=False)
    events_df.to_csv(events_csv_path, index=False)
    events_json_path.write_text(events_df.to_json(orient="records", indent=2), encoding="utf-8")

    final_annotated_path = raw_annotated_path
    if cfg.convert_to_h264:
        audio_source = input_path if cfg.preserve_original_audio else None
        final_annotated_path = convert_video_to_h264(raw_annotated_path, annotated_path, audio_source)
        if final_annotated_path != raw_annotated_path and raw_annotated_path.exists():
            raw_annotated_path.unlink(missing_ok=True)

    flag_counts = events_df["label"].value_counts().to_dict() if not events_df.empty else {}
    report_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Exam Review Report - {html.escape(input_path.name)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.45}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:7px}} th{{background:#f4f4f4}}</style></head>
<body><h1>Exam Surveillance AI Review Report</h1>
<p><b>Author:</b> {AUTHOR}<br><b>Mentor:</b> {MENTOR}</p>
<p><b>Input video:</b> {html.escape(input_path.name)}<br><b>YOLO model:</b> {html.escape(model_name)}<br>
<b>Frames processed:</b> {len(frame_log)}<br><b>FPS:</b> {fps:.2f}<br><b>Resolution:</b> {width} x {height}</p>
<h2>Flag counts</h2><pre>{html.escape(json.dumps(flag_counts, indent=2))}</pre>
<h2>Events for human review</h2>
{events_df.to_html(index=False, escape=True) if not events_df.empty else '<p>No suspicious events passed the configured duration thresholds.</p>'}
<h2>Interpretation</h2><p>These are automated review flags, not proof of cheating. A teacher, examiner, or other authorized human reviewer must inspect the video context before taking action.</p>
<p>Camera-only analysis cannot reliably determine whether a student actually searched the web. Browser/screen instrumentation would be required for stronger evidence and should be deployed only with appropriate consent and policy controls.</p>
</body></html>"""
    report_html_path.write_text(report_html, encoding="utf-8")

    return {
        "input_video": video_path,
        "annotated_video": str(final_annotated_path),
        "frame_log_csv": str(frame_log_path),
        "events_csv": str(events_csv_path),
        "events_json": str(events_json_path),
        "review_report_html": str(report_html_path),
        "event_count": int(len(events_df)),
        "max_risk_score": float(frame_log["risk_score"].max()) if not frame_log.empty else 0.0,
        "model_name": model_name,
        "frames_processed": int(len(frame_log)),
        "fps": fps,
        "resolution": f"{width}x{height}",
    }


def extract_event_clips(
    artifacts: Dict[str, Any],
    output_dir: Optional[str] = None,
    padding_sec: float = 2.0,
) -> List[str]:
    if shutil.which("ffmpeg") is None:
        return []
    events_df = pd.read_csv(artifacts["events_csv"])
    if events_df.empty:
        return []
    source_video = Path(artifacts["annotated_video"])
    clip_dir = Path(output_dir) if output_dir else source_video.parent / "event_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    for i, row in events_df.iterrows():
        start = max(0.0, float(row["start_sec"]) - padding_sec)
        end = float(row["end_sec"]) + padding_sec
        duration = max(0.1, end - start)
        flag = str(row["flag"]).replace("/", "_").replace(" ", "_")
        out_path = clip_dir / f"event_{i:03d}_{flag}_{start:.1f}s_{end:.1f}s.mp4"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source_video),
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and out_path.exists():
            created.append(str(out_path))
    return created


def create_results_zip(results: List[Dict[str, Any]], destination: str) -> str:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.suffix.lower() != ".zip":
        destination_path = destination_path.with_suffix(".zip")
    temp_root = Path(tempfile.mkdtemp(prefix="proctor_bundle_"))
    try:
        summary: List[Dict[str, Any]] = []
        for idx, artifacts in enumerate(results, start=1):
            video_name = _sanitize_stem(Path(artifacts["input_video"]).name)
            target = temp_root / f"{idx:02d}_{video_name}"
            target.mkdir(parents=True, exist_ok=True)
            for key in ("annotated_video", "frame_log_csv", "events_csv", "events_json", "review_report_html"):
                source = Path(artifacts[key])
                if source.exists():
                    shutil.copy2(source, target / source.name)
            clip_dir = Path(artifacts.get("clips_dir", "")) if artifacts.get("clips_dir") else None
            if clip_dir and clip_dir.exists():
                shutil.copytree(clip_dir, target / "event_clips", dirs_exist_ok=True)
            summary.append({k: v for k, v in artifacts.items() if k not in {"input_video"}})
        (temp_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        shutil.make_archive(str(destination_path.with_suffix("")), "zip", root_dir=temp_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return str(destination_path)
