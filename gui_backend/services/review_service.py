import os
import numpy as np
from local_managers.data_sync_manager import DataSyncManager
from gui_backend.services.history_service import HistoryService

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ReviewService:
    def __init__(self, history_service: HistoryService, data_sync_manager: DataSyncManager,
                 pipeline_engine=None):
        self.history_service = history_service
        self.data_sync_manager = data_sync_manager
        self.pipeline = pipeline_engine  # PipelineEngine instance (có thể None nếu chưa load)

    def list_pending(self, page: int = 1, page_size: int = 20) -> dict:
        return self.history_service.list_records(page=page, page_size=page_size)

    def approve(self, record_id: str, labels: dict[str, bool]) -> dict:
        record = self.history_service.get_record(record_id)
        if not record:
            raise ValueError("Record not found")

        result = self.data_sync_manager.sync_record(
            record.get("audio_path"),
            record.get("image_path"),
            labels,
        )

        # ── Tự động append embedding vào retrieval database ─────────
        db_results = {}
        if self.pipeline is not None:
            if result.image_dest:
                db_results["image"] = self._append_image_to_db(
                    result.image_dest, labels, result.batch_dir)
            if result.audio_dest:
                db_results["audio"] = self._append_audio_to_db(
                    result.audio_dest, labels, result.batch_dir)

        return {
            "record_id": record_id,
            "labels": labels,
            "audio_dest": result.audio_dest,
            "image_dest": result.image_dest,
            "batch_dir": result.batch_dir,
            "csv_path": result.csv_path,
            "database_updated": db_results,
        }

    def _append_image_to_db(self, image_dest: str, labels: dict[str, bool], batch_dir: str) -> dict:
        """Trích xuất embedding ảnh và append vào image_database.npy."""
        db_path = os.path.join(BASE_DIR, "Local_Data", "databases", "image_database.npy")
        try:
            tensor = self.pipeline.preprocess_image(image_dest)
            emb = self.pipeline.get_image_embedding(tensor)  # (256,) numpy

            # Label dict cho database
            db_label = {
                "Normal": 1 if not any(labels.get(k, False) for k in ["Pneumonia", "COPD_Emphysema", "Fibrosis"]) else 0,
                "Pneumonia": 1 if labels.get("Pneumonia", False) else 0,
                "COPD_Emphysema": 1 if labels.get("COPD_Emphysema", False) else 0,
                "Fibrosis": 1 if labels.get("Fibrosis", False) else 0,
            }

            return self._append_to_db(db_path, emb, image_dest, db_label, os.path.basename(image_dest))
        except Exception as e:
            return {"error": str(e)}

    def _append_audio_to_db(self, audio_dest: str, labels: dict[str, bool], batch_dir: str) -> dict:
        """Trích xuất embedding audio và append vào audio_database.npy."""
        db_path = os.path.join(BASE_DIR, "Local_Data", "databases", "audio_database.npy")
        try:
            tensor = self.pipeline.preprocess_audio(audio_dest)
            emb = self.pipeline.get_audio_embedding(tensor)  # (256,) numpy

            # Xác định acoustic label
            crackle = 1 if labels.get("Crackle", False) else 0
            wheeze = 1 if labels.get("Wheeze", False) else 0
            if crackle and wheeze:
                acoustic = "Crackle_Wheeze"
            elif crackle:
                acoustic = "Crackle"
            elif wheeze:
                acoustic = "Wheeze"
            else:
                acoustic = "Normal"

            db_label = {"acoustic": acoustic, "crackle": crackle, "wheeze": wheeze}

            return self._append_to_db(db_path, emb, audio_dest, db_label, os.path.basename(audio_dest))
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _append_to_db(db_path: str, embedding: np.ndarray, file_path: str,
                      label: dict, case_id: str) -> dict:
        """Append 1 embedding vào file .npy database. Tự động skip nếu case_id đã tồn tại."""
        # Load database hiện có
        if os.path.exists(db_path):
            db = np.load(db_path, allow_pickle=True).item()
            # Skip nếu case_id đã tồn tại
            if case_id in db.get("case_ids", []):
                return {"status": "skipped", "reason": f"case_id '{case_id}' đã tồn tại", "total": len(db["case_ids"])}
        else:
            db = {"embeddings": np.empty((0, 256), dtype=np.float32), "files": [],
                  "labels": [], "case_ids": []}

        # Append
        new_emb = np.array([embedding], dtype=np.float32)  # (1, 256)
        db["embeddings"] = np.concatenate([db["embeddings"], new_emb], axis=0)
        db["files"].append(file_path)
        db["labels"].append(label)
        db["case_ids"].append(case_id)

        np.save(db_path, db, allow_pickle=True)
        return {"status": "appended", "case_id": case_id, "total": len(db["case_ids"])}

    def get_state(self) -> dict:
        return self.data_sync_manager.load_state()

    def reject(self, record_id: str) -> dict:
        """Chuyển dữ liệu pending vào thùng rác."""
        return self.history_service.delete_record(record_id)

    def list_trash(self) -> list[dict]:
        """Liệt kê dữ liệu trong thùng rác."""
        return self.history_service.list_trash()

    def restore_from_trash(self, record_id: str) -> dict:
        """Khôi phục dữ liệu từ thùng rác về pending."""
        return self.history_service.restore_from_trash(record_id)

    def delete_permanently(self, record_id: str) -> dict:
        """Xóa vĩnh viễn dữ liệu khỏi thùng rác."""
        return self.history_service.delete_permanently(record_id)

    def advance_batch(self) -> dict:
        return self.data_sync_manager.advance_batch()
