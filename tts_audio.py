import pygame
from gtts import gTTS
import os
import time
import threading

# Initialize mixer once
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Audio init check: {e}")

_audio_lock = threading.Lock()

def play_audio_file(filename):
    """Worker function to play audio safely."""
    with _audio_lock:
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            # Clean up file
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass

def speak(text, lang_code='en'):
    """
    Generate TTS and play audio in a non-blocking thread.
    """
    if not text:
        return

    def _generate_and_play():
        try:
            filename = f"tts_{int(time.time())}_{threading.get_ident()}.mp3"
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(filename)
            play_audio_file(filename)
        except Exception as e:
            print(f"TTS error: {e}")

    # Run in separate thread to not block Streamlit UI
    thread = threading.Thread(target=_generate_and_play)
    thread.start()

def repeat_last_output(text, lang_code='en'):
    """Repeat the last spoken text."""
    speak(text, lang_code)
