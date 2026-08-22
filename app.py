# Exam Surveillance AI Review Streamlit App
# Author: Hyunjun Yi
# Mentor: Dr. Qingyang Xiao

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from proctor_engine import (
    AUTHOR,
    MENTOR,
    ExamProctorConfig,
    create_results_zip,
    extract_event_clips,
    load_yolo_model,
    process_video,
    runtime_status,
)

st.set_page_config(
    page_title="Exam Surveillance AI Review",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 4rem;}
.credit {font-size: 0.95rem; opacity: 0.8; margin-top: -0.5rem;}
.review-box {padding: 0.9rem 1rem; border: 1px solid rgba(128,128,128,.28); border-radius: 0.65rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎓 Exam Surveillance AI Review Prototype")
st.markdown(f'<div class="credit"><b>Author:</b> {AUTHOR} &nbsp; | &nbsp; <b>Mentor:</b> {MENTOR}</div>', unsafe_allow_html=True)
st.info(
    "This prototype flags suspicious-looking events for human review. It does not determine that a student cheated, "
    "and its outputs should not be used as the sole basis for disciplinary action."
)

with st.expander("What this prototype can flag", expanded=False):
    st.markdown(
        """
- More people than the configured exam policy allows.
- Cellphone visibility.
- Laptop/computer proxy visibility when that policy flag is enabled.
- More screen-like devices than the configured allowed count (default: one).
- Visual speaking-like mouth motion using MediaPipe face landmarks.
- Possible pre-written scratch-sheet content and later ink-density changes when a paper ROI is configured.
- Optional weak camera-only proxies for online-search activity.

**Important:** a webcam cannot reliably see what website is open on the student's authorized computer. Stronger online-search detection would require a consented browser extension, screen capture, or meeting-platform integration.
"""
    )

status = runtime_status()
with st.sidebar:
    st.header("Analysis settings")
    model_choice = st.selectbox("YOLO model", ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"], index=0)
    confidence = st.slider("Object confidence", 0.10, 0.80, 0.25, 0.05)
    sample_every = st.slider("Run YOLO every N frames", 1, 10, 3, 1)
    process_full = st.checkbox("Process full video", value=False)
    max_frames = None if process_full else st.number_input("Maximum frames per video", 30, 9000, 900, 30)

    st.subheader("Exam policy")
    allowed_people = st.number_input("Allowed people in view", 1, 5, 1, 1)
    allowed_screens = st.number_input("Allowed screen-like devices", 0, 5, 1, 1)
    flag_phone = st.checkbox("Flag cellphone", value=True)
    flag_laptop = st.checkbox("Flag laptop / computer proxy", value=True)
    flag_tv = st.checkbox("Flag any monitor/TV proxy individually", value=False, help="Leave off when one primary exam monitor is allowed.")
    online_proxy = st.checkbox("Enable weak online-search hardware proxy", value=False, help="May flag normal keyboards/mice or the authorized exam computer. Use cautiously.")
    speaking = st.checkbox("Enable speaking-like mouth-motion flag", value=True)

    st.subheader("Scratch paper")
    paper_mode = st.selectbox("Paper ROI mode", ["Disabled", "Auto-detect", "Normalized ROI"], index=0)
    paper_roi: Optional[Tuple[float, float, float, float]] = None
    if paper_mode == "Normalized ROI":
        c1, c2 = st.columns(2)
        x1 = c1.number_input("x1", 0.0, 1.0, 0.08, 0.01)
        y1 = c2.number_input("y1", 0.0, 1.0, 0.55, 0.01)
        x2 = c1.number_input("x2", 0.0, 1.0, 0.58, 0.01)
        y2 = c2.number_input("y2", 0.0, 1.0, 0.95, 0.01)
        if x2 > x1 and y2 > y1:
            paper_roi = (float(x1), float(y1), float(x2), float(y2))
        else:
            st.warning("ROI requires x2 > x1 and y2 > y1.")

    st.subheader("Outputs")
    create_clips = st.checkbox("Create short event clips", value=True)
    clip_padding = st.slider("Event clip padding (seconds)", 0.0, 5.0, 2.0, 0.5)

    with st.expander("Runtime diagnostics"):
        st.json(status)


@st.cache_resource(show_spinner=False)
def get_model(primary_model: str):
    candidates = tuple(dict.fromkeys([primary_model, "yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]))
    cfg = ExamProctorConfig(yolo_model_candidates=candidates)
    return load_yolo_model(cfg)


def safe_uploaded_name(name: str) -> str:
    base = Path(name).name
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
    return cleaned[:160] or "exam_video.mp4"


uploads = st.file_uploader(
    "Upload one or more short exam recordings",
    type=["mp4", "mov", "avi", "mkv", "m4v"],
    accept_multiple_files=True,
    help="Short recordings are recommended on Streamlit Community Cloud because CPU and memory are limited.",
)

analyze = st.button("▶ Run AI review", type="primary", disabled=not uploads, use_container_width=True)

if analyze and uploads:
    work_root = Path(tempfile.mkdtemp(prefix="exam_review_"))
    inputs_dir = work_root / "inputs"
    outputs_root = work_root / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    try:
        with st.spinner("Loading YOLO model. The first run may download model weights..."):
            model, model_name = get_model(model_choice)

        results = []
        for index, upload in enumerate(uploads, start=1):
            st.subheader(f"Processing {index}/{len(uploads)}: {upload.name}")
            input_path = inputs_dir / f"{index:02d}_{safe_uploaded_name(upload.name)}"
            input_path.write_bytes(upload.getbuffer())
            video_output_dir = outputs_root / f"video_{index:02d}"
            video_output_dir.mkdir(parents=True, exist_ok=True)

            cfg = ExamProctorConfig(
                yolo_model_candidates=(model_name,),
                yolo_conf=float(confidence),
                sample_every_n_frames=int(sample_every),
                max_frames=None if process_full else int(max_frames),
                allowed_people=int(allowed_people),
                allowed_screen_like_count=int(allowed_screens),
                flag_cell_phone=bool(flag_phone),
                flag_laptop=bool(flag_laptop),
                flag_tv_as_disallowed=bool(flag_tv),
                flag_online_search_proxy=bool(online_proxy),
                speaking_enabled=bool(speaking),
                paper_roi=paper_roi,
                auto_detect_paper_roi=paper_mode == "Auto-detect",
                output_dir=str(video_output_dir),
            )

            progress_bar = st.progress(0.0, text="Reading video...")

            def update_progress(current: int, total: Optional[int]) -> None:
                if total and total > 0:
                    value = min(1.0, current / total)
                    progress_bar.progress(value, text=f"Processed {current:,} / {total:,} frames")
                else:
                    progress_bar.progress(0.0, text=f"Processed {current:,} frames")

            try:
                artifacts = process_video(
                    str(input_path), cfg, model=model, model_name=model_name, progress_callback=update_progress
                )
                progress_bar.progress(1.0, text="Analysis complete")
                if create_clips:
                    clip_dir = video_output_dir / "event_clips"
                    clips = extract_event_clips(artifacts, str(clip_dir), float(clip_padding))
                    if clips:
                        artifacts["clips_dir"] = str(clip_dir)
                        artifacts["clip_count"] = len(clips)
                results.append(artifacts)
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Failed to process {upload.name}: {exc}")

        if results:
            bundle_path = work_root / "exam_surveillance_ai_review_outputs.zip"
            create_results_zip(results, str(bundle_path))
            st.session_state["last_results"] = results
            st.session_state["last_bundle"] = str(bundle_path)
            st.session_state["last_work_root"] = str(work_root)
            st.success(f"Completed {len(results)} video(s). Review the flags below before drawing conclusions.")
    except Exception as exc:
        st.error(f"Model initialization failed: {exc}")

results = st.session_state.get("last_results", [])
bundle = st.session_state.get("last_bundle")

if results:
    st.divider()
    st.header("Review results")
    summary_rows = [
        {
            "video": Path(r["input_video"]).name,
            "events": r.get("event_count", 0),
            "max_review_score": round(float(r.get("max_risk_score", 0.0)), 3),
            "frames_processed": r.get("frames_processed", 0),
            "model": r.get("model_name", ""),
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if bundle and Path(bundle).exists():
        st.download_button(
            "⬇ Download all review outputs (.zip)",
            data=Path(bundle).read_bytes(),
            file_name="exam_surveillance_ai_review_outputs.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    for idx, artifacts in enumerate(results, start=1):
        name = Path(artifacts["input_video"]).name
        with st.expander(f"Video {idx}: {name} — {artifacts.get('event_count', 0)} event(s)", expanded=idx == 1):
            c1, c2, c3 = st.columns(3)
            c1.metric("Events", artifacts.get("event_count", 0))
            c2.metric("Max review score", f"{float(artifacts.get('max_risk_score', 0.0)):.2f}")
            c3.metric("Frames processed", artifacts.get("frames_processed", 0))

            annotated = Path(artifacts["annotated_video"])
            if annotated.exists():
                st.video(str(annotated))

            events_path = Path(artifacts["events_csv"])
            events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
            st.markdown("#### Events requiring human review")
            if events_df.empty:
                st.success("No event passed the configured duration thresholds.")
            else:
                st.dataframe(events_df, use_container_width=True, hide_index=True)

            d1, d2, d3, d4 = st.columns(4)
            if annotated.exists():
                d1.download_button("Annotated video", annotated.read_bytes(), annotated.name, "video/mp4", key=f"vid_{idx}")
            d2.download_button("Event CSV", events_path.read_bytes(), events_path.name, "text/csv", key=f"csv_{idx}")
            event_json = Path(artifacts["events_json"])
            d3.download_button("Event JSON", event_json.read_bytes(), event_json.name, "application/json", key=f"json_{idx}")
            report = Path(artifacts["review_report_html"])
            d4.download_button("HTML report", report.read_bytes(), report.name, "text/html", key=f"html_{idx}")

st.divider()
st.subheader("Future Zoom / Google Meet / Microsoft Teams integration")
st.write(
    "The current repository analyzes uploaded recordings. A later production integration can feed consented meeting video or screen-sharing streams into the same `process_video`/event-review layer, then send review events to an examiner dashboard. Platform-specific authentication, permissions, retention policy, and privacy controls must be implemented separately."
)
st.caption(f"Author: {AUTHOR} • Mentor: {MENTOR} • Human review required for all automated flags")
