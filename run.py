import os
import platform
import subprocess
import sys
from pathlib import Path

APP_FILE = sys.argv[1] if len(sys.argv) > 1 else "app.py"
VENV_DIR = Path(".venv")
REQ_FILE = Path("requirements.txt")


def run(cmd, **kwargs):
    print("\n>", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, **kwargs)


def venv_python():
    if platform.system().lower().startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main():
    if not REQ_FILE.exists():
        raise SystemExit("requirements.txt not found in current folder.")

    # 1) Create venv if missing
    if not VENV_DIR.exists():
        print(f"Creating virtual environment at: {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"Using existing virtual environment at: {VENV_DIR}")

    py = venv_python()
    if not py.exists():
        raise SystemExit(f"Venv python not found at: {py}")

    # 2) Install requirements inside venv
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])

    # 3) Run streamlit using the venv python
    app_path = Path(APP_FILE)
    if not app_path.exists():
        raise SystemExit(f"Streamlit app file not found: {APP_FILE}")

    # Streamlit runs until you stop it, so we use check_call (blocks)
    run([str(py), "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
