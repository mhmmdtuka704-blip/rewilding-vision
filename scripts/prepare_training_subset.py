"""Fetch only the recorded RUGD subset, pinned by revision and SHA-256 hashes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    manifest = json.loads((ROOT / "data/rugd_subset/provenance.json").read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"] if args.split == "all" or r["split"] == args.split]
    repo = manifest["mirror"].removeprefix("https://huggingface.co/datasets/")
    print("Review original RUGD terms before acquisition/use; see docs/data-card.md.")
    print(f"Pinned revision: {manifest['revision']}; selected {len(rows)} image/annotation pairs", flush=True)
    tasks = [(r, kind) for r in rows for kind in ["image", "mask"]]
    def fetch(task):
        row, kind = task
        target = (ROOT / row[f"{kind}_path"]).resolve()
        if not target.is_relative_to((ROOT / "data/rugd_subset").resolve()):
            raise ValueError("Dataset path leaves the expected data folder")
        digest = row[f"{kind}_sha256"]
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            return target.stat().st_size
        url = f"https://huggingface.co/datasets/{repo}/resolve/{manifest['revision']}/{row[f'remote_{kind}']}"
        response = requests.get(url, timeout=(15, 120))
        response.raise_for_status()
        data = response.content
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {target.name}; no file replaced")
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name + ".download")
        pending.write_bytes(data)
        pending.replace(target)
        return len(data)
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, size in enumerate(pool.map(fetch, tasks), 1):
            total += size
            if index % 40 == 0 or index == len(tasks):
                print(f"Verified {index}/{len(tasks)} files; {total/1e6:.1f} MB", flush=True)
    print("Recorded subset ready. No new split or source revision was selected.")

if __name__ == "__main__":
    main()
