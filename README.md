
# 👁️ Smart Assistive Vision System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

**An intelligent, accessibility-focused application designed to empower the visually impaired with real-time visual understanding.**

This project integrates state-of-the-art computer vision models (**YOLOv11** for object detection and **BLIP** for image captioning) with a user-friendly **Streamlit** interface and **Text-to-Speech (TTS)** capabilities to describe the world in real-time.

---

## 🌟 Key Features

*   **🚦 Sign Board Detection:** Real-time detection of traffic signs and road hazards using a custom-trained **YOLOv11** model.
*   **🖼️ Image Captioning:** Generates descriptive captions for general scenes using the **BLIP** transformer model.
*   **🗣️ Multilingual Voice Feedback:**
    *   Supports **English, Kannada, Hindi, Tamil, and Telugu**.
    *   Automatically translates detections and captions into the selected language.
    *   Uses **gTTS** (Google Text-to-Speech) for clear audio output.
*   **♿ Accessibility First Design:**
    *   **Voice Guidance:** The system reads out menu options and instructions (e.g., "Select Option 1 for Image Captioning").
    *   **High Contrast UI:** Large buttons and clear fonts for low-vision users.
    *   **Simple Navigation:** Linear flow (Mode -> Language -> Camera).
*   **📹 Hardware Integration:** optimized for **ESP32-CAM** streaming, making it a portable assistive device solution.

---

## 🛠️ Technology Stack

*   **Language:** Python
*   **Framework:** Streamlit
*   **Computer Vision:**
    *   Ultralytics YOLOv11 (Object/Sign Detection)
    *   Hugging Face Transformers (BLIP for Captioning)
    *   OpenCV (Image Processing)
*   **Audio & Translation:**
    *   `gTTS` (Text-to-Speech)
    *   `pygame` (Audio Playback)
    *   `deep-translator` (Real-time Translation)

---

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8 or higher installed.
*   A working webcam or ESP32-CAM url.

### 1. Clone the Repository
```bash
git clone https://github.com/Karthik-bhandarkar/Intelligent-Assistive-System.git
cd "Intelligent-Assistive-System"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: For the YOLO and Torch dependencies, it may take a few minutes to download the models.*

### 3. Run the Application
```bash
streamlit run streamlit_app.py
```

---

## 📖 Usage Guide

1.  **Select Mode:**
    *   **Option 1:** Image Captioning (Describes the scene).
    *   **Option 2:** Sign Board Detection (Focuses on road signs).
    *   **Option 3:** Combined Mode (Runs both).
2.  **Select Language:** Choose your preferred language for audio feedback (English, Kannada, Hindi, Tamil, Telugu).
3.  **Start Camera:** Click "Start" to begin the live feed. The system will process frames and read out detections automatically.

---

## 📂 Project Structure

```
├── YOLOV11_Sign_Board_Detection/ # YOLO training data/models
├── best.pt                       # Custom trained YOLOv11 model for signs
├── yolo11n.pt                    # Pre-trained YOLO model for general objects
├── streamlit_app.py              # Main application entry point
├── blip_caption.py               # Image captioning logic
├── yolo_sign_detection.py        # Object detection logic
├── tts_audio.py                  # Text-to-speech module
├── translator.py                 # Language translation module
└── requirements.txt              # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the UI, add more languages, or optimize the models:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
    Made with ❤️ for Accessibility
</div>
