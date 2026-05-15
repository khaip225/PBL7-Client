import os
import tempfile
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from ..models.schemas import DiagnosisResponse

router = APIRouter()

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXT = {".wav"}


def _save_upload(upload: UploadFile, ext: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name


@router.post("/api/diagnosis", response_model=DiagnosisResponse)
async def diagnose(
    request: Request,
    mode: str = Form(...),
    audio_file: UploadFile | None = File(None),
    image_file: UploadFile | None = File(None),
):
    if mode not in ("fusion", "audio", "image"):
        raise HTTPException(400, "Invalid mode. Use: fusion, audio, image")
    if mode == "fusion" and (not audio_file or not image_file):
        raise HTTPException(400, "Fusion mode requires both audio_file and image_file")
    if mode == "audio" and not audio_file:
        raise HTTPException(400, "Audio mode requires audio_file")
    if mode == "image" and not image_file:
        raise HTTPException(400, "Image mode requires image_file")

    audio_path = None
    image_path = None

    if audio_file:
        ext = os.path.splitext(audio_file.filename or ".wav")[1].lower()
        if ext not in ALLOWED_AUDIO_EXT:
            raise HTTPException(400, f"Audio must be .wav, got {ext}")
        audio_path = _save_upload(audio_file, ext)

    if image_file:
        ext = os.path.splitext(image_file.filename or ".png")[1].lower()
        if ext not in ALLOWED_IMAGE_EXT:
            raise HTTPException(400, f"Image must be .png/.jpg/.jpeg, got {ext}")
        image_path = _save_upload(image_file, ext)

    service = request.app.state.diagnosis_service
    try:
        result = service.run(mode, audio_path, image_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return DiagnosisResponse(**result)
