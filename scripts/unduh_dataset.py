"""Unduh hanya CROPPED/*.bmp dari mirror HuggingFace SIPaKMeD."""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

TUJUAN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sipakmed_raw")

p = snapshot_download(
    repo_id="rjaditya/sipakmed-dataset",
    repo_type="dataset",
    allow_patterns=["images/*/*/CROPPED/*.bmp"],
    local_dir=str(TUJUAN),
    max_workers=16,
)
print("SELESAI:", p)

akar = Path(p) / "images"
total = 0
for kelas in sorted(akar.iterdir()):
    n = len(list((kelas / kelas.name / "CROPPED").glob("*.bmp")))
    total += n
    print(f"  {kelas.name:30s} {n:5d}")
print("TOTAL:", total)
