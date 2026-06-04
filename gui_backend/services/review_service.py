from local_managers.data_sync_manager import DataSyncManager
from gui_backend.services.history_service import HistoryService


class ReviewService:
    def __init__(self, history_service: HistoryService, data_sync_manager: DataSyncManager):
        self.history_service = history_service
        self.data_sync_manager = data_sync_manager

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

        return {
            "record_id": record_id,
            "labels": labels,
            "audio_dest": result.audio_dest,
            "image_dest": result.image_dest,
            "batch_dir": result.batch_dir,
            "csv_path": result.csv_path,
        }

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
