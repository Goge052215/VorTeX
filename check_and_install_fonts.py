#!/usr/bin/env python3
import os
import platform
import subprocess
import sys

def check_and_install_fonts():
    """Check the operating system and run the appropriate font installation script"""
    print("Checking for required fonts...")
    
    system = platform.system()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if system == "Darwin":  # macOS
        print("Detected macOS system")
        font_script = os.path.join(script_dir, "fonts", "fonts_download.bash")
        if os.path.exists(font_script):
            print(f"Running font installation script: {font_script}")
            try:
                subprocess.run(["bash", font_script], check=True)
                print("Font installation script completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Font installation script returned error code {e.returncode}")
        else:
            print(f"Warning: Font installation script not found at {font_script}")
    
    elif system == "Windows":
        print("Detected Windows system")
        font_script = os.path.join(script_dir, "fonts", "fonts_download.ps1")
        if os.path.exists(font_script):
            print(f"Running font installation script: {font_script}")
            try:
                # Use PowerShell to execute the script
                subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", font_script], check=True)
                print("Font installation script completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Font installation script returned error code {e.returncode}")
        else:
            print(f"Warning: Font installation script not found at {font_script}")
    
    else:
        print(f"Unsupported operating system: {system}")
        print("Font installation will be skipped")

def main():
    """Main function to check fonts and launch the application"""
    check_and_install_fonts()
    
    # Launch the main application
    print("Starting VorTeX Calculator...")
    main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    if os.path.exists(main_script):
        try:
            subprocess.run([sys.executable, main_script])
        except Exception as e:
            print(f"Error launching application: {e}")
            return 1
    else:
        print(f"Error: Main application script not found at {main_script}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 