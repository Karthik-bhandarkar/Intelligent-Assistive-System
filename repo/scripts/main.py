import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=========================================================")
    print("  Welcome to Assistive Vision & Sign Detection Launcher")
    print("=========================================================")
    print("Press 1 for Live ESP32 Image Captioning & Multi-lingual Speech")
    print("Press 2 for Live Sign Board Detection (ESP32-CAM or Webcam)")
    print("Press 3 for Live Video Stream Preview (Visual Detection Only)")
    print("Press 4 for Manual Image Captioning (Webcam Capture / File Upload)")
    print("=========================================================")

    try:
        choice = int(input("Enter your choice (1-4): ").strip())

        if choice == 1:
            print("\nLaunching Image Captioning System...\n")
            script_path = os.path.join(SCRIPT_DIR, "Image_captioning_ESP32.py")
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            subprocess.run([sys.executable, script_path])

        elif choice == 2:
            print("\nLaunching Sign Board Detection System...\n")
            script_path = os.path.join(SCRIPT_DIR, "Sign_scan_Esp32.py")
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            subprocess.run([sys.executable, script_path])

        elif choice == 3:
            print("\nLaunching Live Video Stream Viewer (Visual Only)...\n")
            script_path = os.path.join(SCRIPT_DIR, "live.py")
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            subprocess.run([sys.executable, script_path])

        elif choice == 4:
            print("\nLaunching Manual Image Captioning Utility...\n")
            script_path = os.path.join(SCRIPT_DIR, "manual.py")
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            subprocess.run([sys.executable, script_path])

        else:
            print("Invalid input. Please select a number between 1 and 4.")

    except ValueError:
        print("Invalid input. Please enter a valid number (1 to 4).")

if __name__ == "__main__":
    main()
