import os
import sys

# Ensure the repo root is importable regardless of the directory this is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import time
import cv2
import numpy as np
import threading
from PIL import Image

# Import backend modules
from src.capture import esp_stream
from src.captioning import blip_caption
from src.detection import yolo_sign_detection
from src.translation import translator
from src.tts import tts_audio

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Assistive Vision",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS for Accessibility (Large Fonts, Big Buttons)
st.markdown("""
<style>
    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        height: 80px;
        font-size: 24px;
        border-radius: 10px;
        background-color: #f0f2f6;
        color: black;
        border: 2px solid #000;
    }
    .stButton>button:hover {
        border-color: #eb4034;
        color: #eb4034;
    }
    /* Fixed Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #262730;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 18px; 
        border-top: 2px solid #eb4034;
        z-index: 100000;
    }
    /* Fixed Header (Matched to Footer Size) */
    .header {
        position: fixed;
        left: 0;
        top: 0;
        width: 100%;
        background-color: #262730;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 18px; 
        font-weight: bold;
        border-bottom: 2px solid #eb4034;
        z-index: 100000;
    }
    /* Adjust main content padding so fixed header/footer don't cover it */
    .block-container {
        padding-top: 80px !important;
        padding-bottom: 60px !important;
    }
</style>
""", unsafe_allow_html=True)

# Fixed Header Injection
st.markdown(
    """
    <div class="header">
        Smart Assistive Vision System
    </div>
    """,
    unsafe_allow_html=True
)

# Footer Injection
st.markdown(
    """
    <div class="footer">
        <p style="margin:0;">Developed for Smart Assistive Vision System | ❤️ Helping the Visually Impaired</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if 'mode' not in st.session_state:
    st.session_state['mode'] = None # 'caption', 'sign', 'both'
if 'language' not in st.session_state:
    st.session_state['language'] = None # Force selection
if 'language_name' not in st.session_state:
    st.session_state['language_name'] = None
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False
if 'last_output' not in st.session_state:
    st.session_state['last_output'] = ""
if 'last_processed_time' not in st.session_state:
    st.session_state['last_processed_time'] = 0
if 'language_welcome_played' not in st.session_state:
    st.session_state['language_welcome_played'] = False
if 'source_type' not in st.session_state:
    st.session_state['source_type'] = 'webcam'
if 'esp32_url' not in st.session_state:
    st.session_state['esp32_url'] = "http://10.219.6.122/cam-hi.jpg"
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

if 'welcome_played' not in st.session_state:
    st.session_state['welcome_played'] = False

# Helper to reset state
def reset_app():
    st.session_state['mode'] = None
    st.session_state['language'] = None
    st.session_state['camera_active'] = False
    st.rerun()

# =========================================================
# STEP 1: MODE SELECTION
# =========================================================
if st.session_state['mode'] is None:
    # Play Welcome Message Once
    if not st.session_state['welcome_played']:
        st.session_state['welcome_played'] = True
        tts_audio.speak("Welcome. System Ready. Select Option 1 for Image Captioning. Select Option 2 for Sign Board Detection. Select Option 3 for Combined Mode.")
    
    st.markdown("### 1️⃣ Step 1: Select a Mode")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("1. 🖼️ Image Captioning"):
            st.session_state['mode'] = 'caption'
            tts_audio.speak("Option 1 Selected. Image Captioning Mode.")
            st.rerun()

    with col2:
        if st.button("2. 🚦 Sign Board Detection"):
            st.session_state['mode'] = 'sign'
            tts_audio.speak("Option 2 Selected. Sign Board Detection Mode.")
            st.rerun()

    with col3:
        if st.button("3. 🔀 Combined Mode"):
            st.session_state['mode'] = 'both'
            tts_audio.speak("Option 3 Selected. Combined Mode.")
            st.rerun()

# =========================================================
# STEP 2: LANGUAGE SELECTION
# =========================================================
elif st.session_state['language'] is None:
    # Reset welcome flag for step 1 so it plays again if they go back
    st.session_state['welcome_played'] = False
    
    # Play Language Welcome Message Once
    if not st.session_state['language_welcome_played']:
        st.session_state['language_welcome_played'] = True
        tts_audio.speak("Step 2. Select Language. Option 1 English. Option 2 Kannada. Option 3 Hindi. Option 4 Tamil. Option 5 Telugu.")

    # Show selected mode
    mode_display = {
        'caption': "🖼️ Image Captioning",
        'sign': "🚦 Sign Board Detection",
        'both': "🔀 Combined Mode"
    }
    st.success(f"Selected Mode: **{mode_display[st.session_state['mode']]}**")
    
    st.markdown("### 2️⃣ Step 2: Select a Language")
    
    l_col1, l_col2, l_col3, l_col4, l_col5 = st.columns(5)

    def set_lang(name, code):
        st.session_state['language'] = code
        st.session_state['language_name'] = name
        tts_audio.speak(f"{name} Selected. Ready to start.")
        st.rerun()

    with l_col1:
        if st.button("1. English"): set_lang("English", "en")
    with l_col2:
        if st.button("2. Kannada"): set_lang("Kannada", "kn")
    with l_col3:
        if st.button("3. Hindi"): set_lang("Hindi", "hi")
    with l_col4:
        if st.button("4. Tamil"): set_lang("Tamil", "ta")
    with l_col5:
        if st.button("5. Telugu"): set_lang("Telugu", "te")
        
    if st.button("🔙 Go Back"):
        st.session_state['language_welcome_played'] = False
        reset_app()

# =========================================================
# STEP 3: CAMERA & INPUT SOURCE CONTROL
# =========================================================
else:
    c_main, c_side = st.columns([0.65, 0.35], gap="large")
    
    with c_main:
        frame_placeholder = st.empty()
        result_placeholder = st.empty()

    with c_side:
        mode_display = {
            'caption': "🖼️ Image Captioning",
            'sign': "🚦 Sign Board Detection",
            'both': "🔀 Combined Mode"
        }
        st.info(f"**Mode:** {mode_display[st.session_state['mode']]}\n\n**Lang:** {st.session_state['language_name']}")
        
        st.markdown("---")
        st.markdown("### 📷 Input Camera Source")
        
        source_idx = 0 if st.session_state['source_type'] == 'webcam' else (1 if st.session_state['source_type'] == 'esp32' else 2)
        source_choice = st.radio(
            "Select Source:",
            ["💻 Laptop Webcam", "📡 ESP32-CAM", "📁 Upload Image"],
            index=source_idx
        )

        if source_choice == "💻 Laptop Webcam":
            st.session_state['source_type'] = 'webcam'
        elif source_choice == "📡 ESP32-CAM":
            st.session_state['source_type'] = 'esp32'
            st.session_state['esp32_url'] = st.text_input("ESP32 Stream URL:", value=st.session_state['esp32_url'])
        elif source_choice == "📁 Upload Image":
            st.session_state['source_type'] = 'upload'
            uploaded = st.file_uploader("Upload Image File:", type=["jpg", "jpeg", "png"])
            if uploaded:
                st.session_state['uploaded_file'] = uploaded

        st.markdown("---")
        
        if not st.session_state['camera_active']:
            if st.button("🟢 Start", use_container_width=True):
                st.session_state['camera_active'] = True
                tts_audio.speak("Starting System")
                st.rerun()
        else:
            if st.button("🔴 Stop", use_container_width=True):
                st.session_state['camera_active'] = False
                tts_audio.speak("Stopping System")
                st.rerun()

        if st.button("🔊 Repeat Audio", use_container_width=True):
            if st.session_state['last_output']:
                translated_text = translator.translate_text(st.session_state['last_output'], st.session_state['language'])
                tts_audio.speak(translated_text, st.session_state['language'])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔙 Reset", use_container_width=True):
             reset_app()

# =========================================================
# MAIN LOOP
# =========================================================
import queue

@st.cache_resource
def get_caption_queue():
    return queue.Queue()

caption_queue = get_caption_queue()

@st.cache_resource
class ThreadState:
    def __init__(self):
        self.blip_active = False

thread_state = ThreadState()

def run_blip_thread(image_copy):
    """Background worker for BLIP"""
    try:
        caption = blip_caption.generate_caption(image_copy)
        if caption:
            caption_queue.put(caption)
    except Exception as e:
        print(f"BLIP Thread Error: {e}")
    finally:
        thread_state.blip_active = False

if 'last_sign_time' not in st.session_state:
    st.session_state['last_sign_time'] = 0
if 'last_caption_time' not in st.session_state:
    st.session_state['last_caption_time'] = 0

if st.session_state['camera_active'] and st.session_state['mode']:
    
    while st.session_state['camera_active']:
        pil_image = None
        
        # 1. Fetch frame based on selected input source
        if st.session_state['source_type'] == 'webcam':
            pil_image = esp_stream.get_frame("webcam")
        elif st.session_state['source_type'] == 'esp32':
            pil_image = esp_stream.get_frame(st.session_state['esp32_url'])
        elif st.session_state['source_type'] == 'upload' and st.session_state['uploaded_file'] is not None:
            try:
                pil_image = Image.open(st.session_state['uploaded_file']).convert("RGB")
            except Exception as e:
                print(f"Image load error: {e}")
        
        if pil_image is not None:
            display_image = pil_image
            
            vis_model_type = 'general' if st.session_state['mode'] == 'caption' else 'sign'
            detected_obj, annotated_bgr = yolo_sign_detection.detect_sign(pil_image, model_type=vis_model_type)
            
            if annotated_bgr is not None:
                display_image = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            current_time = time.time()
            
            # Sign Detection Audio
            if st.session_state['mode'] in ['sign', 'both']:
                if current_time - st.session_state['last_sign_time'] >= 5:
                    if detected_obj:
                         st.session_state['last_output'] = f"Sign Detected: {detected_obj}"
                         st.session_state['last_sign_time'] = current_time
                         translated_text = translator.translate_text(st.session_state['last_output'], st.session_state['language'])
                         tts_audio.speak(translated_text, st.session_state['language'])

            # Image Captioning Audio
            if st.session_state['mode'] in ['caption', 'both']:
                if current_time - st.session_state['last_caption_time'] >= 3:
                    if not thread_state.blip_active:
                        thread_state.blip_active = True
                        st.session_state['last_caption_time'] = current_time
                        img_copy = pil_image.copy()
                        t = threading.Thread(target=run_blip_thread, args=(img_copy,))
                        t.start()
                
                try:
                    result_text = caption_queue.get_nowait()
                    st.session_state['last_output'] = result_text
                    translated_text = translator.translate_text(result_text, st.session_state['language'])
                    tts_audio.speak(translated_text, st.session_state['language'])
                except queue.Empty:
                    pass

            frame_placeholder.image(display_image, channels="RGB", width=640)
            
            if st.session_state['last_output']:
                 result_placeholder.markdown(
                    f"<p style='text-align: center; color: #3498db; font-size: 24px; font-weight: bold; margin-top: 10px;'>{st.session_state['last_output']}</p>", 
                    unsafe_allow_html=True
                 )
        else:
            if st.session_state['source_type'] == 'upload':
                frame_placeholder.info("Please upload an image file using the side panel...")
            else:
                frame_placeholder.error("Waiting for camera frame...")
            time.sleep(1)
            
        time.sleep(0.1)

# =========================================================
# REPEAT BUTTON
# =========================================================
st.markdown("---")
if st.button("🔊 Repeat Last Audio"):
    if st.session_state['last_output']:
         translated_text = translator.translate_text(st.session_state['last_output'], st.session_state['language'])
         tts_audio.speak(translated_text, st.session_state['language'])
