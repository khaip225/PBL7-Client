"""Setup retrieval database tu file Kaggle da tai ve.

Su dung:
   python scripts/setup_retrieval_db.py --zip-path ../forretrival/retrieval_test.zip
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

FORRETRIVAL = os.path.join(os.path.dirname(BASE_DIR), "forretrival")

NIH_TEST_CSV = os.path.join(FORRETRIVAL, "nih_cxr_test_balanced.csv")
ICBHI_TEST_CSV = os.path.join(FORRETRIVAL, "icbhi_test_balanced.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", help="Duong dan file retrieval_test.zip")
    parser.add_argument("--extract-dir", default=None,
                        help="Thu muc giai nen (default: Local_Data/retrieval_db)")
    args = parser.parse_args()

    # ── Xac dinh thu muc giai nen ────────────────────────────────────
    extract_dir = args.extract_dir or os.path.join(BASE_DIR, "Local_Data", "retrieval_db")
    nih_img_dir = os.path.join(extract_dir, "images")
    mel_spec_dir = os.path.join(extract_dir, "mel_specs")

    # ── Giai nen zip ──────────────────────────────────────────────────
    if args.zip_path and os.path.exists(args.zip_path):
        print(f"Giai nen {args.zip_path} -> {extract_dir}")
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("Done.")
    else:
        print(f"Khong tim thay zip. Gia su da giai nen san vao {extract_dir}")

    # ── Kiem tra ──────────────────────────────────────────────────────
    nih_imgs = list(Path(nih_img_dir).glob("*.png"))
    mel_specs = list(Path(mel_spec_dir).glob("*.npy"))
    print(f"Anh NIH: {len(nih_imgs)}")
    print(f"Mel-spec ICBHI: {len(mel_specs)}")

    if len(nih_imgs) == 0 and len(mel_specs) == 0:
        print("\nERROR: Chua co file nao!")
        print("Hay tai retrieval_test.zip tu Kaggle va chay:")
        print(f"  python scripts/setup_retrieval_db.py --zip-path path/to/retrieval_test.zip")
        return

    # ── Tao CSV voi path local ────────────────────────────────────────
    # Remap path trong CSV tu /kaggle/working/stage1_final/ -> extract_dir
    print("\nTao CSV local path...")

    # NIH: path column <- basename -> nih_img_dir
    if os.path.exists(NIH_TEST_CSV) and len(nih_imgs) > 0:
        nih_df = pd.read_csv(NIH_TEST_CSV)
        nih_df["local_path"] = nih_df["path"].apply(
            lambda p: os.path.join(nih_img_dir, os.path.basename(str(p)))
        )
        valid_nih = nih_df[nih_df["local_path"].apply(os.path.exists)]
        print(f"NIH valid files: {len(valid_nih)}/{len(nih_df)}")
        local_nih_csv = os.path.join(extract_dir, "nih_test_local.csv")
        valid_nih.to_csv(local_nih_csv, index=False)
    else:
        print("SKIP NIH (khong co anh hoac CSV)")
        local_nih_csv = None

    # ICBHI: spec_path column <- basename -> mel_spec_dir
    if os.path.exists(ICBHI_TEST_CSV) and len(mel_specs) > 0:
        icbhi_df = pd.read_csv(ICBHI_TEST_CSV)
        icbhi_df["local_spec_path"] = icbhi_df["spec_path"].apply(
            lambda p: os.path.join(mel_spec_dir, os.path.basename(str(p)))
        )
        valid_icbhi = icbhi_df[icbhi_df["local_spec_path"].apply(os.path.exists)]
        print(f"ICBHI valid files: {len(valid_icbhi)}/{len(icbhi_df)}")
        local_icbhi_csv = os.path.join(extract_dir, "icbhi_test_local.csv")
        valid_icbhi.to_csv(local_icbhi_csv, index=False)
    else:
        print("SKIP ICBHI (khong co spec hoac CSV)")
        local_icbhi_csv = None

    print("\n=== Build database ===")

    from ai_engines.pipeline_engine import PipelineEngine

    STAGE4_PATH = os.path.join(BASE_DIR, "models", "stage4_best_model.pth")
    IMAGE_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Image_best.pth")
    AUDIO_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Audio_best.pth")

    engine = PipelineEngine(
        stage4_path=STAGE4_PATH,
        image_head_path=IMAGE_HEAD_PATH,
        audio_head_path=AUDIO_HEAD_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ── Build image DB ────────────────────────────────────────────────
    if local_nih_csv and os.path.exists(local_nih_csv):
        df = pd.read_csv(local_nih_csv)
        embeddings = []
        files = []
        labels_list = []
        case_ids = []

        for i, row in df.iterrows():
            img_path = row["local_path"]
            case_id = row.get("Image Index", f"case_{i:04d}")
            try:
                tensor = engine.preprocess_image(img_path)
                emb = engine.get_image_embedding(tensor)
                embeddings.append(emb)
                files.append(img_path)
                labels_list.append({
                    "disease": row.get("manual_class", "Unknown"),
                    "Normal": int(row.get("Normal", 0)),
                    "Pneumonia": int(row.get("Pneumonia", 0)),
                    "COPD_Emphysema": int(row.get("Emphysema", 0)),
                    "Fibrosis": int(row.get("Fibrosis", 0)),
                })
                case_ids.append(str(case_id))
            except Exception as e:
                print(f"  SKIP {img_path}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(df)}")

        db = {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "files": files,
            "labels": labels_list,
            "case_ids": case_ids,
        }
        image_db_path = os.path.join(BASE_DIR, "Local_Data", "databases", "image_database.npy")
        np.save(image_db_path, db, allow_pickle=True)
        print(f"Image DB: {len(embeddings)} embeddings -> {image_db_path}")

    # ── Build audio DB ────────────────────────────────────────────────
    if local_icbhi_csv and os.path.exists(local_icbhi_csv):
        df = pd.read_csv(local_icbhi_csv)
        embeddings = []
        files = []
        labels_list = []
        case_ids = []

        for i, row in df.iterrows():
            spec_path = row["local_spec_path"]
            case_id = f"{row.get('patient_id', '?')}_{i:04d}"
            try:
                # Load mel-spectrogram .npy (128, T)
                spec = np.load(spec_path).astype(np.float32)
                # Chuẩn hóa z-score (giống notebook)
                spec = (spec - spec.mean()) / (spec.std() + 1e-6)
                # Pad/crop time dimension den MAX_FRAMES=384
                if spec.shape[1] < 384:
                    spec = np.pad(spec, ((0, 0), (0, 384 - spec.shape[1])), mode="constant")
                else:
                    spec = spec[:, :384]
                # Tensor (1, 128, 384) — KHOP voi PipelineEngine.preprocess_audio
                spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).to(engine.device)
                with torch.no_grad():
                    emb = engine.audio_encoder(spec_tensor)
                    emb = emb.squeeze(0).cpu().numpy()

                embeddings.append(emb)
                files.append(spec_path)
                labels_list.append({
                    "acoustic": row.get("combo", "Unknown"),
                    "crackle": int(row.get("crackle", 0)),
                    "wheeze": int(row.get("wheeze", 0)),
                })
                case_ids.append(str(case_id))
            except Exception as e:
                print(f"  SKIP {spec_path}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(df)}")

        db = {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "files": files,
            "labels": labels_list,
            "case_ids": case_ids,
        }
        audio_db_path = os.path.join(BASE_DIR, "Local_Data", "databases", "audio_database.npy")
        np.save(audio_db_path, db, allow_pickle=True)
        print(f"Audio DB: {len(embeddings)} embeddings -> {audio_db_path}")

    print("\n=== HOAN THANH ===")
    print(f"Database nam trong {BASE_DIR}/")
    print("Khoi dong lai server de su dung database moi.")


if __name__ == "__main__":
    main()
