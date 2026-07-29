# Repository Audit & Architecture Report

**Repository:** Intelligent Assistive System  
**Branch:** `new-feature-branch`  
**Date:** July 29, 2026  

---

## 1. Complete Directory Tree

```text
repo/
├── REPO_AUDIT.md                     # [THIS FILE] Comprehensive repository audit report
├── LICENSE                           # Apache-2.0 License file
├── README.md                         # Project documentation and user guide
├── SETUP_COMMANDS.md                 # System setup and execution commands cheat-sheet
├── .gitignore                        # Git ignore rules for venv, cache, and temp files
├── requirements.txt                  # Production Python dependencies
├── run_project.bat                   # Windows batch launcher script
│
├── app/
│   └── streamlit_app.py              # Main Web UI application (Streamlit)
│
├── src/                              # Core modular Python package
│   ├── __init__.py                   # Package root marker
│   ├── captioning/
│   │   ├── __init__.py
│   │   └── blip_caption.py           # Salesforce BLIP AI image captioning module
│   ├── capture/
│   │   ├── __init__.py
│   │   └── esp_stream.py             # ESP32-CAM stream fetcher & preprocessor
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_sign_detection.py    # YOLOv11 sign detection & annotation engine
│   ├── translation/
│   │   ├── __init__.py
│   │   └── translator.py             # Deep-translator wrapper module
│   └── tts/
│       ├── __init__.py
│       └── tts_audio.py              # Non-blocking gTTS + Pygame audio engine
│
├── scripts/                          # Standalone CLI tools & scripts
│   ├── Image_captioning_ESP32.py     # CLI live ESP32 captioning & multi-lingual speech
│   ├── Sign_Scan_local.py            # CLI local webcam sign detection with speech
│   ├── Sign_scan_Esp32.py            # CLI ESP32-CAM sign detection with speech
│   ├── live.py                       # Minimal live stream viewer (visual only)
│   ├── main.py                       # Interactive CLI launcher menu
│   └── manual.py                     # Manual webcam/image upload captioning test script
│
├── models/                           # Machine Learning model artifacts
│   ├── best.onnx                     # ONNX exported traffic sign detection model
│   ├── best.pt                       # PyTorch custom YOLO traffic sign detection model
│   └── yolo11n.pt                    # Pre-trained YOLOv11 Nano general object model
│
├── assets/                           # Audio assets
│   └── audio/                        # Pre-recorded alert MP3 files (12 items)
│       ├── Left_turn.mp3
│       ├── Market_in_front.mp3
│       ├── Pedestrian_crossing.mp3
│       ├── Rail_crossing.mp3
│       ├── Rightturn.mp3
│       ├── School_in_front.mp3
│       ├── Speed_breaker.mp3
│       ├── college_in_front.mp3
│       ├── crossroad.mp3
│       ├── side_road_left.mp3
│       ├── side_road_right.mp3
│       └── speed_limit.mp3
│
├── firmware/                         # Hardware micro-controller code
│   └── ESP_CAM.ino                   # Arduino C++ sketch for ESP32-CAM board
│
└── training/                         # Model training & conversion workspace
    ├── YOLOV11 Sign Board Detection.ipynb  # Jupyter notebook for YOLO training
    ├── args (1).yaml                 # Ultralytics training hyperparameters
    ├── best.onnx & best.pt           # Training output weights
    ├── fileonnx.py                   # PyTorch to ONNX export script
    ├── predict.py                    # Standalone ONNX prediction test script
    ├── results (1).csv               # Training metrics CSV
    ├── *.png & *.jpg                 # Training evaluation curves & confusion matrices
    └── Traffic sign detection.v1i.yolov11/ # Roboflow dataset config & test images
```

---

## 2. File-by-File Analysis

### Primary Application Entry Points

#### `app/streamlit_app.py`
- **Purpose**: Main web user interface built with Streamlit. Provides a multi-step user flow (Mode Selection -> Language Selection -> Live Camera & Audio Feedback). Supports Image Captioning Mode, Sign Detection Mode, and Combined Mode with multi-lingual audio output (English, Kannada, Hindi, Tamil, Telugu).
- **Dependencies**: `os`, `sys`, `streamlit`, `time`, `cv2`, `numpy`, `threading`, `queue`, `src.capture.esp_stream`, `src.captioning.blip_caption`, `src.detection.yolo_sign_detection`, `src.translation.translator`, `src.tts.tts_audio`.
- **Referenced By**: Primary web entry point. Executed via `streamlit run app/streamlit_app.py`.
- **Flags/Notes**: Active production Web UI. Uses multi-threading and non-blocking queues to run heavy BLIP captioning without freezing the UI stream.

#### `scripts/main.py`
- **Purpose**: Interactive CLI Launcher. Displays a terminal menu allowing the user to select between Image Captioning (Option 1) and Sign Board Detection (Option 2). Launches sub-scripts via `subprocess.run()`.
- **Dependencies**: `os`, `sys`, `subprocess`.
- **Referenced By**: Primary CLI entry point. Executed via `python scripts/main.py`.
- **Flags/Notes**: Dynamically resolves sub-script paths relative to its own file directory (`SCRIPT_DIR`).

---

### Core Package Modules (`src/`)

#### `src/capture/esp_stream.py`
- **Purpose**: Connects to the ESP32-CAM HTTP endpoint (`http://<ip>/cam-hi.jpg`), fetches raw image bytes, decodes via OpenCV, rotates 180 degrees, resizes to 640x480, and returns a PIL RGB Image.
- **Dependencies**: `requests`, `numpy`, `cv2`, `PIL.Image`.
- **Referenced By**: Imported by `app/streamlit_app.py` (`esp_stream.get_frame(url)`).
- **Flags/Notes**: Cleanly modularized hardware abstraction.

#### `src/captioning/blip_caption.py`
- **Purpose**: Loads the `Salesforce/blip-image-captioning-large` model with automatic GPU (CUDA/MPS) or CPU device detection and FP16 inference optimization. Generates image captions using conditional prompting and beam search, post-processing output to remove prompt artifacts.
- **Dependencies**: `transformers.BlipProcessor`, `transformers.BlipForConditionalGeneration`, `torch`, `re`.
- **Referenced By**: Imported by `app/streamlit_app.py` (`blip_caption.generate_caption(image)`).
- **Flags/Notes**: Core AI vision-to-language component.

#### `src/detection/yolo_sign_detection.py`
- **Purpose**: Loads YOLO models (`best.pt` for traffic signs or `yolo11n.pt` for general objects) with global model caching. Performs detection on image frames, renders bounding box annotations, and extracts the highest-confidence detected class name.
- **Dependencies**: `os`, `ultralytics.YOLO`, `cv2`, `numpy`.
- **Referenced By**: Imported by `app/streamlit_app.py` (`yolo_sign_detection.detect_sign(...)`).
- **Flags/Notes**: Core vision detection module. References `models/` directory using relative paths.

#### `src/translation/translator.py`
- **Purpose**: Wrapper around `deep_translator.GoogleTranslator`. Translates English text strings to target language codes (`kn`, `hi`, `ta`, `te`, `fr`), falling back gracefully to original text if offline or if translation fails.
- **Dependencies**: `deep_translator.GoogleTranslator`.
- **Referenced By**: Imported by `app/streamlit_app.py` (`translator.translate_text(...)`).
- **Flags/Notes**: Clean utility module.

#### `src/tts/tts_audio.py`
- **Purpose**: Generates Text-to-Speech audio using `gTTS` and plays it in a non-blocking background thread via `pygame.mixer`. Temporary `.mp3` files are saved to `tempfile.gettempdir()` and deleted after playback.
- **Dependencies**: `pygame`, `gTTS`, `os`, `tempfile`, `time`, `threading`.
- **Referenced By**: Imported by `app/streamlit_app.py` (`tts_audio.speak(...)`).
- **Flags/Notes**: Core audio feedback component for Web UI.

---

### Scripts Folder (`scripts/`)

#### `scripts/Image_captioning_ESP32.py`
- **Purpose**: Standalone CLI script for live ESP32-CAM image captioning. Features an interactive language choice menu, multi-threaded frame fetching, BLIP captioning, translation, and audio playback.
- **Dependencies**: `os`, `cv2`, `time`, `requests`, `numpy`, `threading`, `re`, `PIL.Image`, `transformers`, `torch`, `gTTS`, `pandas`, `deep_translator`, `pygame`, `tempfile`.
- **Referenced By**: **Reachable via `scripts/main.py` (Option 1)**.
- **Flags/Notes**: CLI parallel implementation of the web app's captioning feature. Loads BLIP model independently inside the script.

#### `scripts/Sign_scan_Esp32.py`
- **Purpose**: Standalone CLI script for live traffic sign board detection from ESP32-CAM stream or webcam fallback. Uses YOLO (`models/best.pt`), `gTTS`, and `pygame` to announce detected signs every 5 seconds.
- **Dependencies**: `os`, `tempfile`, `cv2`, `time`, `pygame`, `requests`, `numpy`, `gTTS`, `ultralytics.YOLO`.
- **Referenced By**: **Reachable via `scripts/main.py` (Option 2)**.
- **Flags/Notes**: CLI parallel implementation of the web app's sign detection feature.

#### `scripts/Sign_Scan_local.py`
- **Purpose**: Standalone CLI script for sign board detection specifically hardcoded for local webcam input (`cv2.VideoCapture(0)`).
- **Dependencies**: `os`, `tempfile`, `cv2`, `gTTS`, `pygame`, `time`, `ultralytics.YOLO`.
- **Referenced By**: **Not currently referenced by any entry point**.
- **Flags/Notes**: Duplicate functionality of `Sign_scan_Esp32.py` (which already supports webcam fallback). Flagged as redundant standalone test script.

#### `scripts/live.py`
- **Purpose**: Standalone lightweight preview script for continuous ESP32 stream visualization using YOLO (`models/best.pt`). Renders OpenCV preview window without audio or TTS.
- **Dependencies**: `os`, `ultralytics.YOLO`, `cv2`, `requests`, `numpy`.
- **Referenced By**: **Not currently referenced by any entry point**.
- **Flags/Notes**: Minimal visual debug script. Flagged as orphaned script.

#### `scripts/manual.py`
- **Purpose**: Interactive CLI test script allowing user to capture a single frame from local webcam (`cv2.VideoCapture(1)`) or upload a local image file path, generate a BLIP caption, translate it, and speak it via Pygame.
- **Dependencies**: `os`, `cv2`, `PIL.Image`, `transformers`, `torch`, `gTTS`, `pygame`, `deep_translator`.
- **Referenced By**: **Not currently referenced by any entry point**.
- **Flags/Notes**: Manual testing / experiment script. Note: hardcodes `VideoCapture(1)` which fails on single-camera systems.

---

### Training & Model Utilities (`training/`)

#### `training/predict.py`
- **Purpose**: Standalone test script used during training to run inference with the exported ONNX model (`best.onnx`) on a webcam or IP camera feed with gTTS alerts.
- **Dependencies**: `ultralytics.YOLO`, `gTTS`, `pygame`, `time`, `os`, `cv2`.
- **Referenced By**: **Not currently referenced by any entry point**.
- **Flags/Notes**: Legacy training verification script.

#### `training/fileonnx.py`
- **Purpose**: Conversion script that loads `best.pt` and exports it to `best.onnx`.
- **Dependencies**: `torch`, `ultralytics.YOLO`.
- **Referenced By**: **Not currently referenced by any entry point**.
- **Flags/Notes**: One-off model export utility script used during training phase.

---

## 3. Script Reachability Matrix (`scripts/`)

| Script File | Reachable via `main.py`? | Reachable via `streamlit_app.py`? | Status |
|:---|:---:|:---:|:---|
| `scripts/main.py` | 🟢 Entry Point | ❌ | **CLI Launcher** |
| `scripts/Image_captioning_ESP32.py` | 🟢 Option 1 | ❌ | **CLI Interactive Mode** |
| `scripts/Sign_scan_Esp32.py` | 🟢 Option 2 | ❌ | **CLI Interactive Mode** |
| `scripts/Sign_Scan_local.py` | 🔴 No | ❌ | **Orphaned** (Webcam variant of `Sign_scan_Esp32.py`) |
| `scripts/live.py` | 🔴 No | ❌ | **Orphaned** (Visual-only debug viewer) |
| `scripts/manual.py` | 🔴 No | ❌ | **Orphaned** (Manual upload test script) |

---

## 4. Identified Code Duplication & Overlaps

1. **Captioning Logic Overlap**:
   - `src/captioning/blip_caption.py` (used by Streamlit Web UI)
   - `scripts/Image_captioning_ESP32.py` (used by CLI launcher)
   - `scripts/manual.py` (standalone manual test)
   *All three duplicate BLIP model loading, prompt construction, beam search parameters, and regex artifact cleaning.*

2. **Sign Detection Overlap**:
   - `src/detection/yolo_sign_detection.py` (used by Streamlit Web UI)
   - `scripts/Sign_scan_Esp32.py` (used by CLI launcher)
   - `scripts/Sign_Scan_local.py` (standalone webcam script)
   - `scripts/live.py` (visual preview)
   - `training/predict.py` (ONNX inference test)
   *Sign detection inference and bounding box plotting are duplicated across 5 files.*

3. **Audio / TTS Overlap**:
   - `src/tts/tts_audio.py` (modular Pygame+gTTS manager)
   - `scripts/Image_captioning_ESP32.py` (inline gTTS + Pygame logic)
   - `scripts/Sign_scan_Esp32.py` (inline gTTS + Pygame logic)
   - `scripts/Sign_Scan_local.py` (inline gTTS + Pygame logic)
   - `scripts/manual.py` (inline gTTS + Pygame logic)
   - `training/predict.py` (inline gTTS + Pygame logic)

---

## 5. Recommendations for Consolidation & Cleanup

> [!NOTE]  
> The items below are recommendations based on the audit. No code has been modified or removed.

1. **Refactor CLI Scripts to Use `src/` Package**:
   - Update `scripts/Image_captioning_ESP32.py` and `scripts/Sign_scan_Esp32.py` to import from `src.captioning.blip_caption`, `src.detection.yolo_sign_detection`, `src.translation.translator`, and `src.tts.tts_audio`. This will eliminate ~400 lines of duplicated model loading and audio logic.

2. **Consolidate Redundant CLI Scripts**:
   - **`scripts/Sign_Scan_local.py`**: Can be removed or merged into `scripts/Sign_scan_Esp32.py` (which already supports both webcam and IP camera sources).
   - **`scripts/live.py`**: Can be integrated into `main.py` as Option 3 ("Visual Stream Preview") or archived.
   - **`scripts/manual.py`**: Can be integrated into `main.py` as Option 4 ("Manual Image Captioning") or moved to `scripts/tools/`.

3. **Archive Training Utilities**:
   - Keep `training/fileonnx.py` and `training/predict.py` inside `training/` as historical artifacts, but add docstrings clarifying they are for model export/validation during training.
