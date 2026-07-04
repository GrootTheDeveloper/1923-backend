from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, File, HTTPException, UploadFile, status, Depends
from pymongo.errors import PyMongoError

from app.config import MAX_PDF_PAGES
from app.routes.cvmatch_common import get_optional_user, validate_pdf_upload

from app.database import documents_collection
from app.services.pdf_extractor import PDFExtractionError, extract_pdf_text

router = APIRouter()


def serialize_document(document: dict, include_full_text: bool = True) -> dict:
    serialized = {
        "id": str(document["_id"]) if document.get("_id") else None,
        "filename": document["filename"],
        "page_count": document["page_count"],
        "char_count": document["char_count"],
        "library_used": document["library_used"],
        "created_at": document["created_at"],
        "pages": document.get("pages", []),
        "saved_to_mongodb": bool(document.get("_id")),
    }

    if document.get("storage_error"):
        serialized["storage_error"] = document["storage_error"]

    if include_full_text:
        serialized["full_text"] = document.get("full_text", "")
    else:
        serialized["text_preview"] = document.get("full_text", "")[:420]

    return serialized


@router.post("/extract", status_code=status.HTTP_201_CREATED)
async def extract_document(file: UploadFile = File(...), current_user: dict = Depends(get_optional_user)):
    file_bytes = await file.read()
    validate_pdf_upload(file, file_bytes)

    try:
        extracted = extract_pdf_text(file_bytes, file.filename, max_pages=MAX_PDF_PAGES)
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc) or "Could not extract text from this PDF.",
        ) from exc

    document = {
        **extracted,
        "owner_id": current_user["id"],
        "created_at": datetime.now(timezone.utc),
    }

    try:
        result = await documents_collection.insert_one(document)
        document["_id"] = result.inserted_id
    except PyMongoError:
        document["_id"] = None
        document["storage_error"] = "MongoDB is unavailable, so the extracted text was not saved."

    return serialize_document(document)


@router.get("")
async def list_documents(current_user: dict = Depends(get_optional_user)):
    try:
        cursor = documents_collection.find({"owner_id": current_user["id"]}).sort("created_at", -1).limit(20)
        documents = await cursor.to_list(length=20)
    except PyMongoError:
        return []

    return [serialize_document(document, include_full_text=False) for document in documents]


@router.get("/{document_id}")
async def get_document(document_id: str, current_user: dict = Depends(get_optional_user)):
    if not ObjectId.is_valid(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    document = await documents_collection.find_one({"_id": ObjectId(document_id), "owner_id": current_user["id"]})
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    return serialize_document(document)
