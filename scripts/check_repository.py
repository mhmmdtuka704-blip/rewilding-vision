"""Check tracked publication files, local links and recorded metric consistency."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def files_to_check():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if result.returncode == 0 and result.stdout:
        return [ROOT / p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "verification", "training"}
    return [p for p in ROOT.rglob("*") if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts)
            and p.name not in {"model.safetensors", "pytorch_model.bin"}]


def check():
    errors = []
    files = files_to_check()
    patterns = [r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
                r"AKIA[A-Z0-9]{16}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                r"[A-Za-z]:[\\/]Users[\\/][^\s\"']+"]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 10_000_000:
            errors.append(f"Oversized tracked file: {relative}")
        if path.suffix.lower() not in {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".bat", ".svg", ".cff"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if any(re.search(pattern, text) for pattern in patterns):
            errors.append(f"Potential credential or personal machine path: {relative}")
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
                if target.startswith(("https:", "http:", "mailto:", "#", "data:")):
                    continue
                local = target.split("#", 1)[0]
                if local and not (path.parent / local).exists():
                    errors.append(f"Broken local link in {relative}: {target}")
    metrics = json.loads((ROOT / "benchmarks/test_metrics.json").read_text())
    counts = metrics["confusion_counts"]
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    expected = {"vegetation_iou": tp/(tp+fp+fn), "dice_f1": 2*tp/(2*tp+fp+fn),
                "pixel_accuracy": (tp+tn)/(tp+fp+fn+tn)}
    for key, value in expected.items():
        if abs(metrics[key]-value) > 1e-12:
            errors.append(f"Recorded metric does not match confusion counts: {key}")
    manifest = ROOT / "data/rugd_subset/manifest.csv"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != metrics["manifest_sha256"]:
        errors.append("Dataset manifest hash differs from the recorded evaluation")
    return errors, len(files)


if __name__ == "__main__":
    errors, count = check()
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"PASS: {count} publication files; local links, size/path/secret scan, metric counts and manifest hash checked.")
