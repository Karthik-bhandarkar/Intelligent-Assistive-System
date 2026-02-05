import os
import cv2
import time
import requests
import numpy as np
import threading
import re
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS

from deep_translator import GoogleTranslator

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
# DEVICE SETUP
# ===========================================================
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ===========================================================
# LOAD BLIP MODEL
# ===========================================================
# ===========================================================
# LOAD BLIP MODEL
# ===========================================================
print("Loading BLIP model (Salesforce/blip-image-captioning-large)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Use float16 for GPU to save memory and increase speed
if device.type != 'cpu':
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large", 
        torch_dtype=torch.float16
    ).to(device)
else:
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

print(f"Model loaded successfully on {device}.\n")

# ===========================================================
# FETCH FRAME FROM ESP32 CAMERA
# ===========================================================
def fetch_frame_from_esp32(url):
    """Fetch a single frame from ESP32 camera, decode safely, and preprocess for BLIP."""
    try:
        response = requests.get(url, stream=True, timeout=5)
        if response.status_code == 200:
            # 1. Safely read bytes
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            
            # 2. Decode using imdecode (handles corrupt frames gracefully most of the time)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Failed to decode frame (corrupt data).")
                return None

            # 3. Convert BGR (OpenCV) -> RGB (PIL/BLIP expectation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 4. Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Pass full 800x600 frame to processor to avoid aspect ratio distortion
            
            return pil_image
        else:
            print(f"Failed to fetch frame. HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching frame: {e}")
        return None

# ===========================================================
# TRANSLATION AND SPEECH
# ===========================================================
import pygame

# Initialize pygame mixer globally
pygame.mixer.init()

# ===========================================================
# TRANSLATION AND SPEECH
# ===========================================================
def translate_and_speak(text, lang_code):
    """Translate and play caption audio using pygame."""
    filename = f"caption_{int(time.time())}.mp3"
    try:
        if lang_code != "en":
            text = GoogleTranslator(source="en", target=lang_code).translate(text)
        print(f"Caption: {text}")
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        
        # Use pygame for playback
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload() # Release file lock
        
    except Exception as e:
        print(f"Speech error: {e}")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except PermissionError:
                print(f"Warning: Could not remove {filename} (file in use).")
            except Exception as e:
                print(f"Warning: Could not remove {filename}: {e}")

# ===========================================================
# CAPTION GENERATION
# ===========================================================
# ===========================================================
# CAPTION GENERATION
# ===========================================================
def generate_caption(image):
    """Generate a caption using BLIP model with conditional prompting and beam search."""
    try:
        # Conditional Prompting - "Describe the image" is more robust than "this scene"
        text = "Describe the image in detail:"
        
        # Preprocess input (image is already PIL RGB from fetch step)
        inputs = processor(image, text, return_tensors="pt").to(device)
        
        # Beam Search for higher quality
        out = model.generate(
            **inputs,
            num_beams=5,
            max_length=40,
            min_length=10,
            repetition_penalty=1.1
        )
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Basic post-processing
        caption = caption.replace("Describe the image in detail: ", "").strip()
        
        return caption
    except Exception as e:
        print(f"Captioning error: {e}")
        return None

# ===========================================================
# LIVE CAPTIONING LOOP
# ===========================================================
import threading

# ===========================================================
# LIVE CAPTIONING LOOP
# ===========================================================
def caption_thread_func(latest_frame_container, lang_code):
    """Worker thread to generate captions periodically without blocking the UI."""
    last_caption_time = 0
    
    while True:
        # Check if we should exit (main thread sets this via a global or shared state if needed, 
        # but for simple scripts daemon threads start/stop with main)
        
        current_time = time.time()
        # Only caption every CAPTION_INTERVAL seconds
        if current_time - last_caption_time >= CAPTION_INTERVAL:
            # Update time immediately so the next cycle counts from NOW, 
            # not after the heavy processing finishes.
            last_caption_time = current_time
            
            if latest_frame_container[0] is not None:
                # Get a copy of the reference to strictly avoid race conditions on the object itself, 
                # though PIL images are immutable-ish.
                pil_image = latest_frame_container[0]
                
                print("Generating caption...")
                caption = generate_caption(pil_image)
                
                if caption:
                    # Fix: Handle cases where model repeats the prompt or adds artifacts
                    prompt = "describe the image in detail"
                    lower_caption = caption.lower()
                    
                    # Remove the prompt if present
                    if prompt in lower_caption:
                        caption = caption[lower_caption.index(prompt) + len(prompt):].strip()
                    
                    # Remove leading punctuation/artifacts like ": ", "ly," using regex
                    # This handles ": ly,", "ly,", ": ", etc.
                    caption = re.sub(r'^[:,\s]*(ly|ly,|ly\.)[:,\s]*', '', caption, flags=re.IGNORECASE).strip()
                    
                    # Capitalize first letter
                    caption = caption.capitalize()
                    
                    translate_and_speak(caption, lang_code)
            else:
                time.sleep(0.1)
        else:
            time.sleep(0.1)

def live_caption_from_esp32(lang_code):
    print("Starting live ESP32 camera captioning...")
    
    # Shared container for the latest frame [pil_image or None]
    # We use a list to make it mutable and sharable between threads
    latest_frame_container = [None]
    
    # Start the captioning thread
    caption_thread = threading.Thread(target=caption_thread_func, args=(latest_frame_container, lang_code), daemon=True)
    caption_thread.start()

    print("Press 'q' to exit.\n")

    try:
        while True:
            # Fetches PIL Image directly
            pil_image = fetch_frame_from_esp32(ESP32_URL)
            
            if pil_image is not None:
                # Update the shared frame for the caption thread
                latest_frame_container[0] = pil_image

                # Convert back to OpenCV BGR for display
                open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                cv2.imshow("ESP32 Live Feed (Press Q to Quit)", open_cv_image)
            else:
                # If fetch fails, just wait a bit and retry loop
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

# ===========================================================
if __name__ == "__main__":
    main()
