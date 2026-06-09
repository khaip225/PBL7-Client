"""Build embedding database (.npy) cho cross-modal retrieval.

Sử dụng:
    python scripts/build_database.py \
        --data-dir Local_Data/pending/images \
        --output image_database.npy \
        --mode image

    python scripts/build_database.py \
        --data-dir Local_Data/audio_files \
        --output audio_database.npy \
        --mode audio

Output .npy có cấu trúc dict:
    {
        "embeddings": np.array (N, 256),
        "files": [str],
        "labels": [dict],
        "case_ids": [str],
    }
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_engines.pipeline_engine import PipelineEngine

# ── Config ──────────────────────────────────────────────────────────────────
STAGE4_PATH = os.path.join(BASE_DIR, "models", "stage4_best_model.pth")
IMAGE_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Image_best.pth")
AUDIO_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Audio_best.pth")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def build_database(data_dir: str, output_path: str, mode: str, engine: PipelineEngine):
    """Duyệt thư mục, trích xuất embedding 256-d, lưu thành .npy.

    Args:
        data_dir: Thư mục chứa file ảnh hoặc audio
        output_path: Đường dẫn file .npy output
        mode: "image" hoặc "audio"
        engine: PipelineEngine đã khởi tạo
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_dir}")

    # ── Thu thập files ──────────────────────────────────────────────────
    exts = IMAGE_EXTS if mode == "image" else AUDIO_EXTS
    files = []
    for ext in exts:
        files.extend(data_dir.glob(f"**/*{ext}"))
        files.extend(data_dir.glob(f"**/*{ext.upper()}"))

    files = sorted(set(files))
    if not files:
        print(f"[BuildDB] WARNING: Không tìm thấy file nào trong {data_dir}")
        return

    print(f"[BuildDB] Tìm thấy {len(files)} files trong {data_dir}")

    # ── Trích xuất embeddings ────────────────────────────────────────────
    embeddings = []
    file_paths = []
    labels_list = []
    case_ids = []
    seen_hashes = set()     # dedup bằng MD5
    dup_count = 0

    for i, file_path in enumerate(files):
        file_str = str(file_path)

        # Dedup: bỏ qua file có nội dung trùng
        try:
            with open(file_str, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(file_hash)
        except OSError:
            pass

        case_id = f"case_{i+1:04d}"

        try:
            if mode == "image":
                tensor = engine.preprocess_image(file_str)
                emb = engine.get_image_embedding(tensor)
                # Chạy classification để lấy label
                probs = engine.classify_image(tensor)
                label = {"disease": max(probs, key=probs.get)}
            else:
                tensor = engine.preprocess_audio(file_str)
                emb = engine.get_audio_embedding(tensor)
                probs = engine.classify_audio(tensor)
                label = {"acoustic": max(probs, key=probs.get)}

            embeddings.append(emb)
            file_paths.append(file_str)
            labels_list.append(label)
            case_ids.append(case_id)

            if (i + 1) % 100 == 0:
                print(f"[BuildDB]  ... {i+1}/{len(files)}")

        except Exception as e:
            print(f"[BuildDB] Bỏ qua {file_str}: {e}")

    if not embeddings:
        print("[BuildDB] Không extract được embedding nào!")
        return

    # ── Lưu database ────────────────────────────────────────────────────
    embeddings_arr = np.array(embeddings, dtype=np.float32)  # (N, 256)
    database = {
        "embeddings": embeddings_arr,
        "files": file_paths,
        "labels": labels_list,
        "case_ids": case_ids,
    }

    np.save(output_path, database, allow_pickle=True)
    print(f"[BuildDB] Đã lưu {len(embeddings)} embeddings vào {output_path}")
    print(f"[BuildDB] Kích thước: {embeddings_arr.shape}")
    if dup_count > 0:
        print(f"[BuildDB] Đã bỏ qua {dup_count} file trùng lặp (MD5)")


def main():
    parser = argparse.ArgumentParser(description="Build embedding database for cross-modal retrieval")
    parser.add_argument("--data-dir", required=True, help="Thư mục chứa file ảnh/audio")
    parser.add_argument("--output", required=True, help="Đường dẫn file .npy output")
    parser.add_argument("--mode", required=True, choices=["image", "audio"],
                        help="Loại dữ liệu: image hoặc audio")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BuildDB] Device: {device}")

    # Khởi tạo PipelineEngine
    engine = PipelineEngine(
        stage4_path=STAGE4_PATH,
        image_head_path=IMAGE_HEAD_PATH,
        audio_head_path=AUDIO_HEAD_PATH,
        device=device,
    )

    build_database(args.data_dir, args.output, args.mode, engine)
    print("[BuildDB] Hoàn thành!")


if __name__ == "__main__":
    main()
