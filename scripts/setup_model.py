"""Fetch the recorded base checkpoint and verify each file before replacing it."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def download_verified(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256:
        return False
    with urlopen(url, timeout=120) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Checksum mismatch for {destination.name}; no file was replaced")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".download")
    pending.write_bytes(data)
    pending.replace(destination)
    return True


def main():
    destination = ROOT / "model/ade20k"
    manifest = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
    print("Model use is subject to licenses/NVIDIA-SegFormer.txt (research/evaluation).")
    print(f"Pinned revision: {manifest['revision']}")
    for filename, digest in manifest["sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("Invalid model filename in provenance")
        url = f"https://huggingface.co/{manifest['model_id']}/resolve/{manifest['revision']}/{filename}"
        changed = download_verified(url, destination / filename, digest)
        print(f"{'Downloaded and verified' if changed else 'Verified cached'}: {filename}", flush=True)
    print("ML checkpoint is ready. Run: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
