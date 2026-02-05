import streamlit as st
import time
import cv2
import numpy as np
import threading

# Import backend modules
import esp_stream
import blip_caption
import yolo_sign_detection
import translator
import tts_audio

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

# ESP32 URL (Configuration)
ESP32_URL = "http://10.219.6.122/cam-hi.jpg"

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
        # Reset language welcome so it plays again next time
        st.session_state['language_welcome_played'] = False
        reset_app()

# =========================================================
# STEP 3: CAMERA CONTROL
# =========================================================
# =========================================================
# STEP 3: CAMERA CONTROL
# =========================================================
else:
    # Use full width container
    
    # Create Side-by-Side Layout
    # Col 1: Camera (Larger), Col 2: Controls & Text (Smaller)
    c_main, c_side = st.columns([0.7, 0.3], gap="large")
    
    with c_main:
        # Camera Feed
        # st.markdown("**Live Camera Feed:**")
        frame_placeholder = st.empty()
        
        # Caption below Camera (One Line Style)
        result_placeholder = st.empty()

    with c_side:
        # 1. Status Info (Moved up since Result is gone from here)
        mode_display = {
            'caption': "🖼️ Image Captioning",
            'sign': "🚦 Sign Board Detection",
            'both': "🔀 Combined Mode"
        }
        st.info(f"**Mode:** {mode_display[st.session_state['mode']]}\n\n**Lang:** {st.session_state['language_name']}")
        
        st.markdown("---")
        
        # 2. Controls
        if not st.session_state['camera_active']:
            if st.button("🟢 Start", use_container_width=True):
                st.session_state['camera_active'] = True
                tts_audio.speak("Starting Camera")
                st.rerun()
        else:
            if st.button("🔴 Stop", use_container_width=True):
                st.session_state['camera_active'] = False
                tts_audio.speak("Stopping Camera")
                st.rerun()

        # Repeat Audio
        if st.button("🔊 Repeat Audio", use_container_width=True):
            if st.session_state['last_output']:
                translated_text = translator.translate_text(st.session_state['last_output'], st.session_state['language'])
                tts_audio.speak(translated_text, st.session_state['language'])

        # Reset
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔙 Reset", use_container_width=True):
             reset_app()

# =========================================================
# MAIN LOOP
# =========================================================
# (Placeholders are now defined above in Step 3, we check if they exist)
# If we are NOT in Step 3, these won't run, which is fine.

# Global Queue for thread communication (Thread-safe & Persistent)
import queue

@st.cache_resource
def get_caption_queue():
    return queue.Queue()

caption_queue = get_caption_queue()

# Global Thread State (Thread-safe)
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
        # Reset flag safely
        thread_state.blip_active = False

# Session State for Timers
if 'last_sign_time' not in st.session_state:
    st.session_state['last_sign_time'] = 0
if 'last_caption_time' not in st.session_state:
    st.session_state['last_caption_time'] = 0

if st.session_state['camera_active'] and st.session_state['mode']:
    
    # Run loop
    while st.session_state['camera_active']:
        # Fetch Frame
        pil_image = esp_stream.get_frame(ESP32_URL)
        
        if pil_image is not None:
            # Prepare frame for display
            display_image = pil_image
            
            # --- 1. VISUALIZATION (YOLO) ---
            # Determine which model to use for visuals
            # If in 'caption' mode -> Use 'general' model (yolo11n.pt) to see objects like person, chair, etc.
            # If in 'sign' or 'both' mode -> Use 'sign' model (best.pt) to see traffic signs.
            # (In 'both', we prioritize signs because that's the specific safety feature, 
            #  but could argue for general. Let's stick to signs for 'both' to avoid clutter 
            #  or confusion with the specific sign detection audio)
            
            vis_model_type = 'general' if st.session_state['mode'] == 'caption' else 'sign'
            
            # Run detection for visuals
            detected_obj, annotated_bgr = yolo_sign_detection.detect_sign(pil_image, model_type=vis_model_type)
            
            if annotated_bgr is not None:
                # Convert BGR (OpenCV) -> RGB (Streamlit)
                display_image = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            # --- 2. LOGIC HANDLING ---
            current_time = time.time()
            
            # A) Speech for Sign Detection (Independent Timer)
            # Only trigger specific sign audio if we are using the sign model
            if st.session_state['mode'] in ['sign', 'both']:
                # If we are in 'both', we used 'sign' model above, so 'detected_obj' is a sign.
                # If we are in 'sign', 'detected_obj' is a sign.
                
                if current_time - st.session_state['last_sign_time'] >= 5:
                    if detected_obj: # If a sign was found
                         st.session_state['last_output'] = f"Sign Detected: {detected_obj}"
                         st.session_state['last_sign_time'] = current_time
                         
                         # Trigger Speech
                         translated_text = translator.translate_text(st.session_state['last_output'], st.session_state['language'])
                         tts_audio.speak(translated_text, st.session_state['language'])

            # B) Image Captioning (Independent Timer & Thread Logic)
            if st.session_state['mode'] in ['caption', 'both']:
                
                # Check timer (Reduced to 2s for faster response as requested)
                if current_time - st.session_state['last_caption_time'] >= 3:
                    # Check if thread is free
                    if not thread_state.blip_active:
                        thread_state.blip_active = True
                        st.session_state['last_caption_time'] = current_time # Reset timer
                        
                        # Start Thread
                        img_copy = pil_image.copy()
                        t = threading.Thread(target=run_blip_thread, args=(img_copy,))
                        t.start()
                
                # Check for result from Queue (Non-blocking check)
                try:
                    result_text = caption_queue.get_nowait()
                    st.session_state['last_output'] = result_text
                    
                    # Speak
                    translated_text = translator.translate_text(result_text, st.session_state['language'])
                    tts_audio.speak(translated_text, st.session_state['language'])
                except queue.Empty:
                    pass

            # DISPLAY MAIN FRAME
            # Update the placeholder defined in Step 3
            # We don't need columns here, the placeholder is already in a column
            frame_placeholder.image(display_image, channels="RGB", width=640)
            
            # DISPLAY TEXT RESULT
            if st.session_state['last_output']:
                 # Use compact Paragraph style instead of H2 to keep it roughly "one line"
                 result_placeholder.markdown(
                    f"<p style='text-align: center; color: #3498db; font-size: 24px; font-weight: bold; margin-top: 10px;'>{st.session_state['last_output']}</p>", 
                    unsafe_allow_html=True
                 )

        else:
            frame_placeholder.error("Waiting for camera frame...")
            time.sleep(1)
            
        # Recommended Frame Rate: 0.1s delay (~10 FPS) for stability
        time.sleep(0.1)

# =========================================================
# REPEAT BUTTON
# =========================================================
st.markdown("---")
if st.button("🔊 Repeat Last Audio"):
    if st.session_state['last_output']:
         translated_text = translator.translate_text(st.session_state['last_output'], st.session_state['language'])
         tts_audio.speak(translated_text, st.session_state['language'])
