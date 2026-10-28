"""Start/reuse the local app and open Microsoft Edge on Windows."""
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def healthy(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except Exception:
        return False


def main():
    edge = next((path for path in [
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ] if path.is_file()), None)
    if edge is None:
        raise RuntimeError("Microsoft Edge was not found. Install Edge and try again.")
    port = 8501
    if not healthy(port):
        for candidate in range(8501, 8511):
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", candidate))
                    port = candidate
                    break
                except OSError:
                    continue
        else:
            raise RuntimeError("Ports 8501–8510 are occupied. Close another local app and retry.")
        logs = ROOT / "verification"
        logs.mkdir(exist_ok=True)
        with (logs / "edge_launcher.log").open("ab") as output:
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
                "--browser.gatherUsageStats=false"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(30):
            if healthy(port):
                break
            time.sleep(1)
        else:
            raise RuntimeError("App startup failed. See verification/edge_launcher.log.")
    url = f"http://127.0.0.1:{port}/"
    subprocess.Popen([str(edge), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Rewilding Vision opened in Edge: {url}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Could not launch Rewilding Vision: {error}")
        sys.exit(1)
