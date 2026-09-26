# Exam Surveillance AI Review Prototype

**Author:** Hyunjun Yi  
**Mentor:** Dr. Qingyang Xiao

A human-in-the-loop Streamlit prototype derived from the Colab notebook `082226_Hyunjun_Yi_Exam_Surveillance_AI_Proctoring_Prototype_DEBUGGED.ipynb`.

The app accepts short exam surveillance recordings, runs AI-based visual review signals, produces an annotated video, aggregates suspicious-looking time intervals, and generates CSV/JSON/HTML evidence for an authorized examiner to review.

> **Important:** The system does **not** decide whether a student cheated. Its outputs are review flags only and should not be the sole basis for disciplinary action.

## Prototype pipeline

```text
Uploaded exam video
        |
        v
Frame decoding with OpenCV
        |
        +--> YOLO object detection
        |      |- people count
        |      |- cell phone
        |      |- laptop/computer proxy
        |      `- screen/device count
        |
        +--> MediaPipe face landmarks
        |      `- speaking-like mouth-motion heuristic
        |
        +--> Optional scratch-paper ROI
               |- baseline ink density
               `- content-change heuristic
        |
        v
Frame-level review flags + review score
        |
        v
Temporal event aggregation
        |
        +--> annotated MP4
        +--> event CSV / JSON
        +--> frame log CSV
        +--> HTML review report
        `--> optional event clips
```

## What is detected

| Review signal | Prototype method |
|---|---|
| More than allowed people | YOLO `person` count |
| Cellphone | YOLO `cell phone` |
| Laptop / computer proxy | YOLO `laptop` |
| Too many screen-like devices | YOLO `laptop`, `tv`, `cell phone` count compared with policy setting |
| Speaking-like behavior | MediaPipe mouth landmark motion |
| Possible pre-written scratch sheet | Early paper ROI ink-density baseline |
| Scratch-sheet content change | Ink-density increase relative to baseline |
| Online searching | **Not directly observable from a webcam.** Optional hardware/device proxies are provided but are weak evidence. |

The Streamlit UI defaults to **one allowed screen-like device** so an authorized primary exam monitor is not automatically treated as an extra-screen violation. The policy can be changed in the sidebar.

## MediaPipe compatibility fix

The original Colab error occurred when a recent MediaPipe installation imported successfully but did not expose `mp.solutions.face_mesh`. This repository contains the same debug approach as the uploaded notebook:

1. Use legacy `mp.solutions.face_mesh` when available.
2. Otherwise use `mp.tasks.vision.FaceLandmarker`.
3. Download the Face Landmarker task model on first use.
4. If neither API is usable, disable only the speaking-like signal while the YOLO analysis continues.

## Repository structure

```text
.
├── app.py                      # Streamlit Community Cloud entry point
├── proctor_engine.py           # reusable video-analysis engine
├── requirements.txt            # Python packages
├── packages.txt                # Linux package: ffmpeg
├── .streamlit/
│   └── config.toml
├── notebooks/
│   └── 082226_Hyunjun_Yi_Exam_Surveillance_AI_Proctoring_Prototype_DEBUGGED.ipynb
├── tests/
│   └── test_engine.py
├── LICENSE
└── README.md
```

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

You also need `ffmpeg` on the operating system if you want H.264 output and event clips.

## Deploy to Streamlit Community Cloud

1. Extract the ZIP generated for this project.
2. Create a new GitHub repository.
3. Upload **all files and folders** from the extracted repository. `app.py` must remain at the repository root.
4. In Streamlit Community Cloud, create a new app from the GitHub repository.
5. Set **Main file path** to `app.py`.
6. Use Python **3.12** in Advanced settings when available.
7. Deploy.

On the first analysis, Ultralytics may download the selected YOLO model and MediaPipe may download `face_landmarker.task`. Later runs in the same app instance are faster because the YOLO model is cached with `st.cache_resource`.

## Streamlit Community Cloud notes

- Use short test videos first. Free cloud CPU/RAM is limited.
- The default configuration processes at most 900 frames and runs YOLO every third frame.
- Increase limits only after confirming the app works with your videos.
- Uploaded videos and generated outputs live in temporary app storage and can disappear when the Streamlit instance restarts.
- For production, use an approved secure storage and retention design instead of temporary filesystem storage.

## Zoom / Google Meet / Microsoft Teams roadmap

This version analyzes uploaded recordings. A later meeting-platform version can keep `proctor_engine.py` as the visual-analysis layer and replace the file uploader with an authorized stream/recording connector. A production implementation should separately handle:

- user and institution consent;
- platform OAuth/application permissions;
- authorized access to video/screen-sharing streams;
- encryption, storage, audit logs, and retention;
- examiner dashboard notifications;
- human adjudication and appeal workflow.

Actual web-search detection cannot be reliably inferred from the webcam. If exam rules require detecting browser activity, use consented screen-sharing or a browser/desktop component designed for that purpose.

## Responsible-use limitations

This prototype can generate false positives and false negatives. Examples include a family member briefly entering the background, normal talking before an exam starts, a permitted phone stored in view, reflections that resemble a second screen, or pre-existing marks on paper. The final interpretation must remain with an authorized human reviewer.

Do not add face recognition, demographic inference, or covert identity tracking to this prototype.


## App user counter

The Streamlit app now shows **App users** on the main page, in the sidebar, and at the bottom of the page. A new Streamlit browser session adds one count without using a database or external analytics service. See `USER_COUNTER.md` for the implementation and the important Community Cloud persistence limitation.

