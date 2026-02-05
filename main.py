import os
import sys
import subprocess

def main():
    print("===============================================")
    print("     Welcome to Image Captioning and Sign Board Detection")
    print("===============================================")
    print("Press 1 for Image Captioning")
    print("Press 2 for Sign Board Detection")
    print("===============================================")

    try:
        choice = int(input("Enter your choice: ").strip())

        if choice == 1:
            print("\nLaunching Image Captioning System...\n")
            script_path = os.path.join(os.getcwd(), "Image_captioning_ESP32.py")
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            subprocess.run([sys.executable, script_path])

        elif choice == 2:
            print("\nLaunching Sign Board Detection System...\n")
            script_path = os.path.join(os.getcwd(), "Sign_scan_Esp32.py")
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                sys.exit(1)
            subprocess.run([sys.executable, script_path])

        else:
            print("Invalid input. Please select 1 or 2.")

    except ValueError:
        print("Invalid input. Please enter a valid number (1 or 2).")

if __name__ == "__main__":
    main()
