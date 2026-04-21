<div align="center">

<br/>

```
██╗ █████╗ ██╗   ██╗███████╗
██║██╔══██╗██║   ██║██╔════╝
██║███████║██║   ██║███████╗
██║██╔══██║╚██╗ ██╔╝╚════██║
██║██║  ██║ ╚████╔╝ ███████║
╚═╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝
Intelligent Assistive Vision System
```

**Real-time environmental awareness for the visually impaired — powered by YOLOv11, BLIP, and multilingual voice feedback.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-F7931E?style=flat-square)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-8B5CF6?style=flat-square)](CONTRIBUTING.md)

<br/>

[Features](#-features) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Usage](#-usage) · [Structure](#-project-structure) · [Roadmap](#-roadmap) · [Contributing](#-contributing)

</div>

---

## Overview

IAVS is an accessibility-first computer vision system that narrates the world in real time. Point a webcam (or an ESP32-CAM module) at any scene and the system will:

1. **Detect** objects and traffic/hazard signs using a custom-trained YOLOv11 model
2. **Caption** the full scene in natural language via BLIP (Salesforce)
3. **Translate** the output into the user's preferred language
4. **Speak** the result aloud with gTTS audio playback

Designed from the ground up for blind and low-vision users, the interface requires no reading — every interaction is voice-guided.

---

## Features

### Object & Sign Detection
- Custom YOLOv11 model trained on traffic signs and road hazards
- Pretrained YOLO (`yolo11n.pt`) for general everyday objects
- Frame-level real-time inference via OpenCV

### Scene Understanding
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) generates contextual, sentence-level scene descriptions
- Goes beyond bounding boxes — understands *what is happening*, not just *what is there*

### Multilingual Voice Feedback

| Language | Code |
|----------|------|
| English  | `en` |
| Kannada  | `kn` |
| Hindi    | `hi` |
| Tamil    | `ta` |
| Telugu   | `te` |

Translation via `deep-translator` · TTS via `gTTS` · Playback via `pygame`

### Accessibility-First Interface
- Voice-guided menu navigation — no reading required
- High-contrast UI with large tap targets
- Simple 3-step workflow: **Mode → Language → Camera**

### Hardware Support
- Works with any USB webcam
- Optimised for **ESP32-CAM** streams (low-cost portable deployment)

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Input Sources              │
│   Webcam / ESP32-CAM / Static Image     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│           Frame Capture (OpenCV)        │
└──────────────┬──────────────┬───────────┘
               │              │
               ▼              ▼
   ┌───────────────┐  ┌───────────────────┐
   │ YOLOv11       │  │  BLIP Captioning  │
   │ Detection     │  │  (Transformers)   │
   └───────┬───────┘  └────────┬──────────┘
           │                   │
           └─────────┬─────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Translation Layer    │
        │   (deep-translator)    │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Text-to-Speech (gTTS) │
        │  + pygame Playback     │
        └────────────┬───────────┘
                     │
                     ▼
           🔊 Audio Feedback to User
```

---

## Quickstart

### Prerequisites

- Python 3.8+
- A webcam or an ESP32-CAM stream URL
- ~4 GB free disk space (YOLO + Torch downloads)

### 1 — Clone

```bash
git clone https://github.com/Karthik-bhandarkar/Intelligent-Assistive-System.git
cd Intelligent-Assistive-System
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch and Ultralytics will download model weights on first run. This may take a few minutes.

### 3 — Launch

```bash
streamlit run app/streamlit_app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## Usage

The interface walks you through three steps:

```
Step 1 — Select Mode
  ├── [1] Image Captioning only
  ├── [2] Sign & Object Detection only
  └── [3] Combined (Caption + Detection)

Step 2 — Select Language
  ├── English / Kannada / Hindi / Tamil / Telugu

Step 3 — Start Camera
  └── Live feed begins; voice feedback triggers automatically
```

All navigation prompts are spoken aloud — keyboard/touch is optional.

---

## Project Structure

```
intelligent-assistive-vision-system/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit entry point
│
├── core/
│   ├── blip_caption.py           # BLIP scene captioning module
│   ├── yolo_sign_detection.py    # YOLOv11 detection module
│   ├── tts_audio.py              # Text-to-speech engine
│   └── translator.py             # Language translation wrapper
│
├── models/
│   ├── best.pt                   # Custom-trained sign detection model
│   └── yolo11n.pt                # Pretrained general-purpose YOLO
│
├── training/
│   └── YOLOV11_Sign_Board_Detection/   # Training data, configs, scripts
│
├── requirements.txt
├── .gitignore
└── README.md
```

> See [RESTRUCTURE.md](RESTRUCTURE.md) for the automated migration script that moves files from the flat layout to this structure without breaking any imports.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Interface | Streamlit |
| Computer Vision | OpenCV, Ultralytics YOLOv11 |
| Scene Understanding | Hugging Face Transformers (BLIP) |
| Translation | deep-translator |
| Text-to-Speech | gTTS |
| Audio Playback | pygame |
| Language | Python 3.8+ |

---

## Roadmap

- [ ] Edge deployment (ONNX / TensorRT export for Raspberry Pi)
- [ ] Persistent object tracking across frames
- [ ] Navigation guidance ("obstacle 2 metres ahead, move right")
- [ ] Expanded language support (Bengali, Marathi, Malayalam)
- [ ] Offline TTS fallback (pyttsx3) for no-internet environments
- [ ] Battery / power optimisation for ESP32-CAM

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

```bash
# Quick workflow
git checkout -b feature/your-feature-name
# ... make changes, add tests, update docs ...
git commit -m "feat: short description"
git push origin feature/your-feature-name
# Open a Pull Request on GitHub
```

**Good first issues:** language support, UI accessibility improvements, documentation, performance profiling.

Follow [PEP 8](https://peps.python.org/pep-0008/), keep modules focused, and document any public function you add.

---

## Acknowledgements

- [Ultralytics](https://ultralytics.com) for YOLOv11
- [Salesforce / Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base) for BLIP
- [Streamlit](https://streamlit.io) for the rapid UI framework

---

<div align="center">

Built with care for accessibility · Bengaluru, India

*"Technology should work for everyone."*

</div>
