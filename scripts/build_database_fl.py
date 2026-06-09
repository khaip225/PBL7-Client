"""Build retrieval database (.npy) từ fl_data đã được bác sĩ duyệt.

Mặc định dùng APPEND mode: chỉ trích xuất embedding cho file MỚI (chưa có trong DB),
rồi nối vào database hiện có. Không extract lại file cũ.

Sử dụng:
    python scripts/build_database_fl.py --fl-dir Local_Data/fl_data/fl_data_1
    python scripts/build_database_fl.py --fl-dir Local_Data/fl_data/fl_data_1 --rebuild  # Build lại từ đầu
    python scripts/build_database_fl.py --fl-dir Local_Data/fl_data/fl_data_1 --dry-run   # Xem trước
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_engines.pipeline_engine import PipelineEngine

STAGE4_PATH = os.path.join(BASE_DIR, "models", "stage4_best_model.pth")
IMAGE_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Image_best.pth")
AUDIO_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Audio_best.pth")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_existing_db(db_path: str) -> dict | None:
    """Load database .npy hiện có, hoặc None nếu chưa tồn tại."""
    if os.path.exists(db_path):
        db = np.load(db_path, allow_pickle=True).item()
        return db
    return None


def _known_case_ids(db: dict | None) -> set:
    """Trả về set các case_id đã có trong database."""
    if db is None:
        return set()
    return set(db.get("case_ids", []))


def _load_image_gt(fl_dir: str) -> dict[str, dict]:
    """Đọc ground-truth image labels từ tất cả CSV metadata.

    Returns: {filename: {Normal, Pneumonia, COPD_Emphysema, Fibrosis}}
    """
    gt = {}
    for csv_path in sorted(Path(fl_dir).glob("metadata/image_fl/*.csv")):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                fname = row.get("path", "")
                if not fname:
                    continue
                gt[fname] = {
                    "Normal": 1 if (row.get("Pneumonia", 0) == 0
                                    and row.get("COPD_Emphysema", 0) == 0
                                    and row.get("Fibrosis", 0) == 0) else 0,
                    "Pneumonia": int(row.get("Pneumonia", 0)),
                    "COPD_Emphysema": int(row.get("COPD_Emphysema", 0)),
                    "Fibrosis": int(row.get("Fibrosis", 0)),
                }
        except Exception as e:
            print(f"  [WARN] Lỗi CSV {csv_path}: {e}")
    return gt


def _load_audio_gt(fl_dir: str) -> dict[str, dict]:
    """Đọc ground-truth audio labels từ tất cả CSV metadata.

    Returns: {filename: {crackle, wheeze}}
    """
    gt = {}
    for csv_path in sorted(Path(fl_dir).glob("metadata/audio_fl/*.csv")):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                fname = row.get("path", "")
                if not fname:
                    continue
                gt[fname] = {
                    "crackle": int(row.get("crackle", 0)),
                    "wheeze": int(row.get("wheeze", 0)),
                }
        except Exception as e:
            print(f"  [WARN] Lỗi CSV {csv_path}: {e}")
    return gt


# ── Build / Append ───────────────────────────────────────────────────────────

DB_DIR = os.path.join(BASE_DIR, "Local_Data", "databases")

def process_image_db(fl_dir: str, engine: PipelineEngine, output_dir: str,
                     rebuild: bool = False, dry_run: bool = False) -> str | None:
    """Xử lý image database: append (default) hoặc rebuild từ đầu."""
    img_dir = os.path.join(fl_dir, "fl_image")
    out_path = os.path.join(DB_DIR, "image_database.npy")

    if not os.path.isdir(img_dir):
        print("[ImageDB] Không có fl_image/ — bỏ qua")
        return None

    # ── Load ground-truth từ CSV ────────────────────────────────────────
    gt_labels = _load_image_gt(fl_dir)
    if not gt_labels:
        print("[ImageDB] Không có ground-truth labels từ CSV — bỏ qua")
        return None

    # ── Load database hiện có ───────────────────────────────────────────
    existing_db = None if rebuild else _load_existing_db(out_path)
    known = _known_case_ids(existing_db)
    if existing_db is not None:
        print(f"[ImageDB] Database hiện có: {len(known)} file")

    # ── Tìm file mới ────────────────────────────────────────────────────
    new_files = []
    skipped = 0
    for fname, labels in gt_labels.items():
        if fname in known:
            skipped += 1
            continue
        fpath = os.path.join(img_dir, fname)
        if os.path.exists(fpath):
            new_files.append((fpath, fname, labels))
        else:
            print(f"  [SKIP] Ảnh không tồn tại: {fpath}")

    print(f"[ImageDB] File mới cần xử lý: {len(new_files)}  |  Đã có: {skipped} (bỏ qua)")

    if not new_files:
        print("[ImageDB] Không có file mới — database đã cập nhật nhất")
        return out_path if existing_db is not None else None

    if dry_run:
        print("[ImageDB] DRY-RUN — hiển thị file mới sẽ được thêm:")
        for _, fname, labels in new_files:
            disease = [k for k, v in labels.items() if v == 1 and k != "Normal"] or ["Normal"]
            print(f"  + {fname}  →  {disease}")
        return out_path

    # ── Trích xuất embedding cho file mới ───────────────────────────────
    new_embeddings = []
    new_paths = []
    new_labels_list = []
    new_case_ids = []

    for i, (fpath, fname, labels) in enumerate(new_files):
        try:
            tensor = engine.preprocess_image(fpath)
            emb = engine.get_image_embedding(tensor)
            new_embeddings.append(emb)
            new_paths.append(fpath)
            new_labels_list.append(labels)
            new_case_ids.append(fname)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(new_files)}")

    if not new_embeddings:
        print("[ImageDB] Không extract được embedding mới nào!")
        return out_path if existing_db is not None else None

    # ── Merge với database cũ ───────────────────────────────────────────
    if existing_db is not None:
        all_emb = np.concatenate([existing_db["embeddings"], np.array(new_embeddings, dtype=np.float32)], axis=0)
        all_files = existing_db["files"] + new_paths
        all_labels = existing_db["labels"] + new_labels_list
        all_cases = existing_db["case_ids"] + new_case_ids
    else:
        all_emb = np.array(new_embeddings, dtype=np.float32)
        all_files = new_paths
        all_labels = new_labels_list
        all_cases = new_case_ids

    db = {
        "embeddings": all_emb,
        "files": all_files,
        "labels": all_labels,
        "case_ids": all_cases,
    }
    np.save(out_path, db, allow_pickle=True)
    print(f"[ImageDB] Đã lưu: {len(all_cases)} total ({len(new_embeddings)} mới + {len(known)} cũ) → {out_path}")
    return out_path


def process_audio_db(fl_dir: str, engine: PipelineEngine, output_dir: str,
                     rebuild: bool = False, dry_run: bool = False) -> str | None:
    """Xử lý audio database: append (default) hoặc rebuild từ đầu."""
    aud_dir = os.path.join(fl_dir, "fl_audio")
    out_path = os.path.join(DB_DIR, "audio_database.npy")

    if not os.path.isdir(aud_dir):
        print("[AudioDB] Không có fl_audio/ — bỏ qua")
        return None

    # ── Load ground-truth từ CSV ────────────────────────────────────────
    gt_labels = _load_audio_gt(fl_dir)
    if not gt_labels:
        print("[AudioDB] Không có ground-truth labels từ CSV — bỏ qua")
        return None

    # ── Load database hiện có ───────────────────────────────────────────
    existing_db = None if rebuild else _load_existing_db(out_path)
    known = _known_case_ids(existing_db)
    if existing_db is not None:
        print(f"[AudioDB] Database hiện có: {len(known)} file")

    # ── Tìm file mới ────────────────────────────────────────────────────
    new_files = []
    skipped = 0
    for fname, labels in gt_labels.items():
        if fname in known:
            skipped += 1
            continue
        fpath = os.path.join(aud_dir, fname)
        if os.path.exists(fpath):
            new_files.append((fpath, fname, labels))
        else:
            print(f"  [SKIP] Audio không tồn tại: {fpath}")

    print(f"[AudioDB] File mới cần xử lý: {len(new_files)}  |  Đã có: {skipped} (bỏ qua)")

    if not new_files:
        print("[AudioDB] Không có file mới — database đã cập nhật nhất")
        return out_path if existing_db is not None else None

    if dry_run:
        print("[AudioDB] DRY-RUN — hiển thị file mới sẽ được thêm:")
        for _, fname, labels in new_files:
            acoustic = "Normal"
            if labels.get("crackle") and labels.get("wheeze"):
                acoustic = "Crackle+Wheeze"
            elif labels.get("crackle"):
                acoustic = "Crackle"
            elif labels.get("wheeze"):
                acoustic = "Wheeze"
            print(f"  + {fname}  →  {acoustic}")
        return out_path

    # ── Trích xuất embedding cho file mới ───────────────────────────────
    new_embeddings = []
    new_paths = []
    new_labels_list = []
    new_case_ids = []

    for i, (fpath, fname, labels) in enumerate(new_files):
        try:
            tensor = engine.preprocess_audio(fpath)
            emb = engine.get_audio_embedding(tensor)
            new_embeddings.append(emb)
            new_paths.append(fpath)
            # Xác định acoustic label
            if labels.get("crackle", 0) == 1 and labels.get("wheeze", 0) == 1:
                acoustic = "Crackle_Wheeze"
            elif labels.get("crackle", 0) == 1:
                acoustic = "Crackle"
            elif labels.get("wheeze", 0) == 1:
                acoustic = "Wheeze"
            else:
                acoustic = "Normal"
            new_labels_list.append({"acoustic": acoustic, **labels})
            new_case_ids.append(fname)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(new_files)}")

    if not new_embeddings:
        print("[AudioDB] Không extract được embedding mới nào!")
        return out_path if existing_db is not None else None

    # ── Merge với database cũ ───────────────────────────────────────────
    if existing_db is not None:
        all_emb = np.concatenate([existing_db["embeddings"], np.array(new_embeddings, dtype=np.float32)], axis=0)
        all_files = existing_db["files"] + new_paths
        all_labels = existing_db["labels"] + new_labels_list
        all_cases = existing_db["case_ids"] + new_case_ids
    else:
        all_emb = np.array(new_embeddings, dtype=np.float32)
        all_files = new_paths
        all_labels = new_labels_list
        all_cases = new_case_ids

    db = {
        "embeddings": all_emb,
        "files": all_files,
        "labels": all_labels,
        "case_ids": all_cases,
    }
    np.save(out_path, db, allow_pickle=True)
    print(f"[AudioDB] Đã lưu: {len(all_cases)} total ({len(new_embeddings)} mới + {len(known)} cũ) → {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build/Append retrieval database từ fl_data")
    parser.add_argument("--fl-dir", required=True,
                        help="Đường dẫn fl_data_X (vd: Local_Data/fl_data/fl_data_1)")
    parser.add_argument("--output-db-dir", default=None,
                        help="Thư mục output file .npy (default: BASE_DIR)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Build lại TOÀN BỘ từ đầu (mặc định: append — chỉ thêm file mới)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Xem trước file mới sẽ được thêm, không thực thi")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    fl_dir = os.path.abspath(args.fl_dir)
    if not os.path.isdir(fl_dir):
        print(f"ERROR: Thư mục không tồn tại: {fl_dir}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_db_dir) if args.output_db_dir else DB_DIR
    os.makedirs(output_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    mode_str = "REBUILD" if args.rebuild else ("DRY-RUN" if args.dry_run else "APPEND")
    print(f"{'='*60}")
    print(f"BUILD RETRIEVAL DATABASE — {mode_str}")
    print(f"{'='*60}")
    print(f"FL dir:   {fl_dir}")
    print(f"Output:   {output_dir}")
    print(f"Device:   {device}")
    if not args.rebuild:
        print(f"Mode:     Chỉ xử lý file MỚI (dùng --rebuild để build lại toàn bộ)")
    print()

    # Khởi tạo PipelineEngine
    print("Khởi tạo PipelineEngine...")
    engine = PipelineEngine(
        stage4_path=STAGE4_PATH,
        image_head_path=IMAGE_HEAD_PATH,
        audio_head_path=AUDIO_HEAD_PATH,
        device=device,
    )
    print("Done.\n")

    # Xử lý
    img_db = process_image_db(fl_dir, engine, output_dir,
                              rebuild=args.rebuild, dry_run=args.dry_run)
    aud_db = process_audio_db(fl_dir, engine, output_dir,
                              rebuild=args.rebuild, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("HOÀN THÀNH")
    print(f"  Image DB: {img_db or 'N/A'}")
    print(f"  Audio DB: {aud_db or 'N/A'}")
    if args.dry_run:
        print("  DRY-RUN — chưa có gì được thay đổi.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
