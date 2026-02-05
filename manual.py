import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS
from playsound import playsound
from deep_translator import GoogleTranslator

# Check if MPS (Metal) is available and set the device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Language mapping for text-to-speech and translation
language_map = {
    1: ("Kannada", "kn"),
    2: ("Hindi", "hi"),
    3: ("Tamil", "ta"),
    4: ("Telugu", "te"),
    5: ("French", "fr"),
    6: ("English", "en"),
}

# Function to translate text to the selected language and convert to speech
def translate_text_to_speech(text, target_language_code):
    try:
        print("Translating the text...")
        translation = GoogleTranslator(source="en", target=target_language_code).translate(text)
        print(f"Translated text: {translation}")

        # Convert translated text to speech
        print("Converting translated text to speech...")
        tts = gTTS(text=translation, lang=target_language_code)
        audio_file = "output.mp3"
        tts.save(audio_file)

        # Play the audio
        playsound(audio_file)

        # Remove the audio file after playback
        os.remove(audio_file)

    except Exception as e:
        print(f"An error occurred: {e}")

# Function to generate caption for an image
def generate_caption(image, selected_language_code):
    # Generate a caption for the image
    text = "You are seeing a"
    inputs = processor(image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Display the caption
    print(f"Caption: {caption}")

    # Translate and convert caption to speech in the selected language
    translate_text_to_speech(caption, selected_language_code)

# Function to capture an image using webcam
def capture_from_webcam(selected_language_code):
    print("Opening webcam...")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    print("Press 'c' to capture an image or 'q' to quit.")

    try:
        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture an image.")
                break

            # Display the live video feed in an OpenCV window
            cv2.imshow("Live Webcam Feed - Press 'c' to Capture or 'q' to Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Capture the image when 'c' is pressed
                print("Capturing image for caption generation...")
                cv2.imwrite("images/captured_frame.jpg", frame)
                image = Image.open("images/captured_frame.jpg")
                generate_caption(image, selected_language_code)
            elif key == ord('q'):  # Quit when 'q' is pressed
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

# Main function
def main():
    print("Image Captioning with Multi-Language TTS")
    print("Welcome to the Multi-Language Image Captioning App")
    print("This application generates captions for images, translates them into a selected language, and plays the translated text as audio.")

    # Language Selection
    print("Select the target language for translation and speech:")
    for key, (name, _) in language_map.items():
        print(f"{key}: {name}")
    language_choice = int(input("Enter your choice: "))
    selected_language_name, selected_language_code = language_map.get(language_choice, ("English", "en"))

    # Image Source Selection
    print("Choose Image Source:")
    print("1: Use Webcam")
    print("2: Upload Image")
    source_choice = int(input("Enter your choice: "))

    if source_choice == 1:
        capture_from_webcam(selected_language_code)
    elif source_choice == 2:
        image_path = input("Enter the path of the image: ")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            generate_caption(image, selected_language_code)
        else:
            print("Invalid file path.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
