from fastapi import HTTPException, UploadFile, status

ALLOWED_CONTENT_TYPES = {"video/mp4", "application/mp4"}
ALLOWED_FTYP_BRANDS = {b"mp41", b"mp42", b"isom", b"avc1", b"M4V ", b"M4A ", b"iso2", b"iso5", b"qt  "}


async def read_valid_mp4(file: UploadFile, max_bytes: int) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only MP4 videos are supported.")
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Video must be 50 MB or smaller.",
        )
    if len(data) < 12 or data[4:8] != b"ftyp" or data[8:12] not in ALLOWED_FTYP_BRANDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file does not look like a valid MP4 video.",
        )
    return data
