import os
import sys
import cv2
import time
import numpy as np
import threading
from PIL import Image

# Ensure the repo root is importable regardless of launch location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.capture import esp_stream
from src.captioning import blip_caption
from src.translation import translator
from src.tts import tts_audio

# ===========================================================
# CONFIGURATION
# ===========================================================
ESP32_URL = "http://10.219.6.122/cam-hi.jpg"  # ESP32 camera URL
CAPTION_INTERVAL = 5  # seconds between caption generations

# Supported languages
language_map = {
    1: ("Kannada", "kn"),
    2: ("Hindi", "hi"),
    3: ("Tamil", "ta"),
    4: ("Telugu", "te"),
    5: ("French", "fr"),
    6: ("English", "en"),
}

# ===========================================================
# LIVE CAPTIONING LOOP
# ===========================================================
def caption_thread_func(latest_frame_container, lang_code):
    """Worker thread to generate captions periodically without blocking the UI."""
    last_caption_time = 0
    
    while True:
        current_time = time.time()
        if current_time - last_caption_time >= CAPTION_INTERVAL:
            last_caption_time = current_time
            
            if latest_frame_container[0] is not None:
                pil_image = latest_frame_container[0]
                print("Generating caption...")
                caption = blip_caption.generate_caption(pil_image)
                
                if caption:
                    translated_text = translator.translate_text(caption, lang_code)
                    print(f"Caption: {translated_text}")
                    tts_audio.speak(translated_text, lang_code)
            else:
                time.sleep(0.1)
        else:
            time.sleep(0.1)

def live_caption_from_esp32(lang_code):
    print("Starting live ESP32 camera captioning...")
    
    latest_frame_container = [None]
    
    caption_thread = threading.Thread(
        target=caption_thread_func,
        args=(latest_frame_container, lang_code),
        daemon=True
    )
    caption_thread.start()

    print("Press 'q' to exit.\n")

    try:
        while True:
            pil_image = esp_stream.get_frame(ESP32_URL)
            
            if pil_image is not None:
                latest_frame_container[0] = pil_image

                open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                cv2.imshow("ESP32 Live Feed (Press Q to Quit)", open_cv_image)
            else:
                print("No frame received. Retrying...")
                time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
                
    except KeyboardInterrupt:
        print("Stopped manually.")
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        cv2.destroyAllWindows()

# ===========================================================
# MAIN
# ===========================================================
def main():
    print("=== Live ESP32 Image Captioning with Multi-Language Speech ===\n")
    print("Select the target language for speech output:")

    for key, (name, _) in language_map.items():
        print(f"{key}: {name}")

    try:
        choice = int(input("\nEnter your choice: "))
        selected_language_name, lang_code = language_map.get(choice, ("English", "en"))
    except Exception:
        print("Invalid input. Defaulting to English.")
        lang_code = "en"

    print(f"\nSelected language: {selected_language_name}\n")
    live_caption_from_esp32(lang_code)

if __name__ == "__main__":
    main()
