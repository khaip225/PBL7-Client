"""Phê duyệt nhanh các file pending vào fl_data để build retrieval database.

Labels được lấy từ tên file (do người dùng đặt khi upload) — đây là ground-truth.
Chỉ dùng cho mục đích demo/POC.

Sử dụng:
    python scripts/approve_pending.py                     # Approve tất cả pending
    python scripts/approve_pending.py --dry-run           # Xem trước, không thực thi
    python scripts/approve_pending.py --batch 1           # Chỉ định batch number
    python scripts/approve_pending.py --auto              # Approve + tự động build database
    python scripts/approve_pending.py --auto --rebuild    # Approve + build lại toàn bộ database
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PENDING_DIR = os.path.join(BASE_DIR, "Local_Data", "pending")
FL_BASE = os.path.join(BASE_DIR, "Local_Data", "fl_data")

# Regex parse filename: {Label1_Label2_...}_{YYYYMMDD_HHMMSS}.{ext}
FILENAME_RE = re.compile(r"^(.+)_(\d{8}_\d{6})\.(\w+)$")

KNOWN_LABELS = {"Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis", "Crackle", "Wheeze"}


def parse_labels_from_filename(filename: str) -> tuple[list[str], str | None]:
    """Parse multi-label từ tên file.

    Ví dụ: "Fibrosis_Crackle_Wheeze_20260605_190353.jpeg"
    → (["Fibrosis", "Crackle", "Wheeze"], "20260605_190353")
    """
    m = FILENAME_RE.match(filename)
    if not m:
        return [], None

    label_str = m.group(1)
    timestamp = m.group(2)

    # Greedy parse: match known labels (longest first)
    known_sorted = sorted(KNOWN_LABELS, key=len, reverse=True)
    result = []
    remaining = label_str
    for known in known_sorted:
        idx = remaining.find(known)
        while idx != -1:
            left_ok = idx == 0 or remaining[idx - 1] == "_"
            right_end = idx + len(known)
            right_ok = right_end == len(remaining) or remaining[right_end] == "_"
            if left_ok and right_ok:
                result.append(known)
                before = remaining[:idx]
                after = remaining[right_end:]
                remaining = (before + after).replace("__", "_").strip("_")
                break
            idx = remaining.find(known, idx + 1)

    if not result:
        result = ["Normal"]

    return result, timestamp


def create_batch_dir(batch_idx: int) -> str:
    """Tạo cấu trúc thư mục fl_data_X."""
    batch_dir = os.path.join(FL_BASE, f"fl_data_{batch_idx}")
    os.makedirs(os.path.join(batch_dir, "fl_image"), exist_ok=True)
    os.makedirs(os.path.join(batch_dir, "fl_audio"), exist_ok=True)
    os.makedirs(os.path.join(batch_dir, "metadata", "image_fl"), exist_ok=True)
    os.makedirs(os.path.join(batch_dir, "metadata", "audio_fl"), exist_ok=True)
    return batch_dir


def approve_images(image_files: list[str], batch_dir: str, dry_run: bool) -> int:
    """Copy ảnh vào fl_image/ và ghi CSV labels."""
    if not image_files:
        return 0

    img_src_dir = os.path.join(PENDING_DIR, "image_files")
    img_dst_dir = os.path.join(batch_dir, "fl_image")
    csv_path = os.path.join(batch_dir, "metadata", "image_fl", "client_1_train.csv")

    count = 0
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a" if file_exists else "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["path", "Pneumonia", "COPD_Emphysema", "Fibrosis"])

        for fname in sorted(set(image_files)):
            labels, _ = parse_labels_from_filename(fname)
            if not labels:
                print(f"  [SKIP] Không parse được labels: {fname}")
                continue

            src = os.path.join(img_src_dir, fname)
            dst = os.path.join(img_dst_dir, fname)

            if not dry_run:
                if not os.path.exists(src):
                    print(f"  [SKIP] File không tồn tại: {src}")
                    continue
                shutil.copy2(src, dst)
                writer.writerow([
                    fname,
                    1 if "Pneumonia" in labels else 0,
                    1 if "COPD_Emphysema" in labels else 0,
                    1 if "Fibrosis" in labels else 0,
                ])

            print(f"  [{'DRY-RUN' if dry_run else 'OK'}] Ảnh: {fname} → labels={labels}")
            count += 1

    return count


def approve_audios(audio_files: list[str], batch_dir: str, dry_run: bool) -> int:
    """Copy audio vào fl_audio/ và ghi CSV labels."""
    if not audio_files:
        return 0

    aud_src_dir = os.path.join(PENDING_DIR, "audio_files")
    aud_dst_dir = os.path.join(batch_dir, "fl_audio")
    csv_path = os.path.join(batch_dir, "metadata", "audio_fl", "client_1_train.csv")

    count = 0
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a" if file_exists else "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["path", "crackle", "wheeze"])

        for fname in sorted(set(audio_files)):
            labels, _ = parse_labels_from_filename(fname)
            if not labels:
                print(f"  [SKIP] Không parse được labels: {fname}")
                continue

            src = os.path.join(aud_src_dir, fname)
            dst = os.path.join(aud_dst_dir, fname)

            if not dry_run:
                if not os.path.exists(src):
                    print(f"  [SKIP] File không tồn tại: {src}")
                    continue
                shutil.copy2(src, dst)
                writer.writerow([
                    fname,
                    1 if "Crackle" in labels else 0,
                    1 if "Wheeze" in labels else 0,
                ])

            print(f"  [{'DRY-RUN' if dry_run else 'OK'}] Audio: {fname} → labels={labels}")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Phê duyệt nhanh pending files vào fl_data")
    parser.add_argument("--dry-run", action="store_true", help="Xem trước, không thực thi")
    parser.add_argument("--batch", type=int, default=1, help="Batch number (default: 1)")
    parser.add_argument("--auto", action="store_true",
                        help="Tự động chạy build_database_fl.py ngay sau khi approve")
    parser.add_argument("--rebuild", action="store_true",
                        help="(Dùng với --auto) Build lại toàn bộ database thay vì append")
    args = parser.parse_args()

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    AUD_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    image_dir = os.path.join(PENDING_DIR, "image_files")
    audio_dir = os.path.join(PENDING_DIR, "audio_files")

    images = sorted(
        [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    ) if os.path.isdir(image_dir) else []
    audios = sorted(
        [f for f in os.listdir(audio_dir) if os.path.splitext(f)[1].lower() in AUD_EXTS]
    ) if os.path.isdir(audio_dir) else []

    print(f"\n{'='*60}")
    print(f"APPROVE PENDING → fl_data_{args.batch}")
    print(f"{'='*60}")
    print(f"Pending ảnh:  {len(images)}")
    print(f"Pending audio: {len(audios)}")
    print(f"Mode: {'DRY-RUN (xem trước)' if args.dry_run else 'THỰC THI'}")
    print()

    if not images and not audios:
        print("Không có file pending nào!")
        return

    batch_dir = create_batch_dir(args.batch)
    print(f"Batch dir: {batch_dir}\n")

    # Approve từng loại
    img_count = approve_images(images, batch_dir, args.dry_run)
    aud_count = approve_audios(audios, batch_dir, args.dry_run)

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ: {img_count} ảnh + {aud_count} audio")
    if args.dry_run:
        print("DRY-RUN — chưa có gì được thực thi.")
        print("Chạy không có --dry-run để thực thi.")
    else:
        print(f"Đã approve vào {batch_dir}")

        # ── Tự động build database nếu có --auto ──────────────────────
        if args.auto and (img_count > 0 or aud_count > 0):
            print(f"\n>>> Tự động build database (append)...")
            build_script = os.path.join(BASE_DIR, "scripts", "build_database_fl.py")
            cmd = [sys.executable, build_script, "--fl-dir", batch_dir]
            if args.rebuild:
                cmd.append("--rebuild")
            import subprocess
            subprocess.run(cmd)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
