import os
import subprocess
import sys

def compile_suite():
    print("[*] Bootstrapping Standalone Executable Compilation Suite...")
    print("[*] Checking dependencies...")
    
    try:
        import PyInstaller
    except ImportError:
        print("[!] PyInstaller not detected. Installing automatically via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[+] PyInstaller installed successfully.")

    # Target file
    target_script = "blood_cell_gui.py"
    if not os.path.exists(target_script):
        print(f"[X] Target script '{target_script}' not found in current directory.")
        sys.exit(1)

    print(f"[*] Compiling '{target_script}' into a optimized, lightweight executable...")
    
    # Construct robust build command excluding heavy, unused global ML packages
    # --noconsole hides terminal boxes on end-user desktops
    # --onefile packages everything cleanly into a portable executable
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconsole",
        "--onefile",
        "--clean",
        "--name=BloodCellAnalyzerSuite",
        "--exclude-module=torch",
        "--exclude-module=ultralytics",
        "--exclude-module=tensorboard",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=torchvision",
        target_script
    ]

    try:
        subprocess.check_call(cmd)
        print("\n[+] Compilation Successful!")
        print("[*] Executable is ready inside the './dist/' folder:")
        exe_path = os.path.abspath(os.path.join("dist", "BloodCellAnalyzerSuite.exe"))
        print(f"    -->  {exe_path}")
        print("\nYou can distribute this standalone file safely to end-users.")
    except subprocess.CalledProcessError as e:
        print(f"\n[X] Compilation process failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    compile_suite()
