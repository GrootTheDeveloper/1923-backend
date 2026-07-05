from __future__ import annotations

import logging

from bson import ObjectId
from fastapi import Depends, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import ENABLE_DEMO_MODE, JWT_ALGORITHM, JWT_SECRET_KEY, MAX_UPLOAD_BYTES
from app.database import users_collection

optional_security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def demo_user() -> dict:
    return {"id": "demo-user", "username": "Demo", "email": ""}


def auth_required_error() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication is required.")


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_security),
) -> dict:
    if credentials is None:
        if ENABLE_DEMO_MODE:
            return demo_user()
        raise auth_required_error()

    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id or not ObjectId.is_valid(user_id):
            logger.warning("Rejected bearer token with an invalid subject.")
            raise auth_required_error()
    except JWTError as exc:
        logger.warning("Rejected invalid or expired bearer token.")
        raise auth_required_error() from exc

    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        logger.warning("Rejected bearer token for a user that no longer exists.")
        raise auth_required_error()

    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
    }


def validate_pdf_upload(file: UploadFile, file_bytes: bytes) -> None:
    filename = file.filename or ""
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")

    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded PDF file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF file is too large. Maximum allowed size is {max_mb:.0f} MB.",
        )

    if not file_bytes.lstrip()[:5] == b"%PDF-":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is not a valid PDF.")


def object_id_or_404(value: str, label: str = "Resource") -> ObjectId:
    if not ObjectId.is_valid(value):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{label} not found.")
    return ObjectId(value)


def model_payload(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=True)
    return model.dict(exclude_unset=True)
