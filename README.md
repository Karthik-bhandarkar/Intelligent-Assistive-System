# 👁️ Intelligent Assistive Vision System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

> An accessibility-focused computer vision application that provides real-time environmental awareness through object detection, scene captioning, translation, and multilingual audio feedback.

This system combines **YOLOv11 (object detection)** and **BLIP (image captioning)** with a clean **Streamlit interface** and **Text-to-Speech (TTS)** integration to help visually impaired users better understand their surroundings.

---

## 🌟 Core Features

### 🚦 Real-Time Sign & Object Detection
- Custom-trained **YOLOv11** model for traffic signs and hazards.
- Pre-trained YOLO model for general object recognition.
- Designed for real-time frame processing.

### 🖼️ Scene Understanding (Image Captioning)
- Uses **BLIP (Hugging Face Transformers)** to generate natural language scene descriptions.
- Goes beyond labels by providing contextual understanding.

### 🗣️ Multilingual Voice Feedback
- Supports **English, Kannada, Hindi, Tamil, and Telugu**
- Real-time translation using `deep-translator`
- Audio output powered by **gTTS**
- Playback handled with `pygame`

### ♿ Accessibility-First Design
- Voice-guided menu instructions
- High-contrast UI with large buttons
- Simple linear workflow:
  ```
  Mode → Language → Camera
  ```

### 📹 Hardware Compatibility
- Optimized for **ESP32-CAM streaming**
- Can function as a portable assistive solution

---

## 🏗 System Architecture

```
Camera / ESP32-CAM
        │
        ▼
Frame Capture (OpenCV)
        │
        ├── YOLOv11 Detection
        │
        ├── BLIP Captioning
        │
        ▼
Translation Layer
        │
        ▼
Text-to-Speech Engine (gTTS)
        │
        ▼
Audio Feedback to User
```

---

## 🛠 Technology Stack

### 🔹 Core Technologies
- Python 3.8+
- Streamlit (Frontend Interface)
- OpenCV (Image Processing)

### 🔹 Computer Vision & AI
- Ultralytics YOLOv11
- Hugging Face Transformers (BLIP)

### 🔹 Audio & Translation
- gTTS (Text-to-Speech)
- pygame (Audio Playback)
- deep-translator (Multilingual Support)

---

## 🚀 Installation & Setup

### 🔧 Prerequisites
- Python 3.8+
- Webcam or ESP32-CAM stream URL

---

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Karthik-bhandarkar/Intelligent-Assistive-System.git
cd Intelligent-Assistive-System
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: YOLO and Torch dependencies may take time to download initially.

---

### 3️⃣ Run Application

```bash
streamlit run streamlit_app.py
```

Application will open in your browser automatically.

---

## 📖 Usage Guide

1. **Select Mode**
   - Option 1: Image Captioning
   - Option 2: Sign Board Detection
   - Option 3: Combined Mode

2. **Select Language**
   - English
   - Kannada
   - Hindi
   - Tamil
   - Telugu

3. **Start Camera**
   - Begin live detection and receive automatic voice feedback.

---

## 📂 Project Structure

```
├── YOLOV11_Sign_Board_Detection/   # YOLO training data & configs
├── best.pt                         # Custom trained YOLO model
├── yolo11n.pt                      # Pretrained YOLO model
├── streamlit_app.py                # Main application
├── blip_caption.py                 # Scene captioning module
├── yolo_sign_detection.py          # Detection module
├── tts_audio.py                    # Text-to-speech module
├── translator.py                   # Language translation module
├── requirements.txt                # Dependencies
└── README.md
```

---

## 🎯 What This Project Demonstrates

✔ Real-time computer vision integration  
✔ Deep learning model usage in practical applications  
✔ Multilingual speech synthesis system  
✔ Streamlit-based UI development  
✔ Modular Python application structure  
✔ Accessibility-oriented product thinking  
✔ Hardware (ESP32-CAM) integration  

---

## 🔮 Potential Enhancements

- Edge deployment optimization
- Improved object tracking stability
- Additional language support
- Navigation guidance system
- Performance optimization for low-resource devices

---

## 🤝 Contributing

Contributions are welcome:

1
