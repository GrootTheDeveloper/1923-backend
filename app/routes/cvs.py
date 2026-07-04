from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pymongo import ReturnDocument

from app.config import MAX_PDF_PAGES
from app.database import candidates_collection, cv_documents_collection, match_results_collection
from app.models.cvmatch import CVExtractedDataUpdate
from app.routes.cvmatch_common import get_optional_user, model_payload, object_id_or_404, validate_pdf_upload
from app.services.gemini_extraction import extract_cv_data_hybrid
from app.services.object_storage import cv_object_key, get_cv_pdf, put_cv_pdf, remove_cv_pdf
from app.services.pdf_extractor import PDFExtractionError, extract_pdf_text
from app.services.pii_service import mask_profile
from app.services.skill_service import normalize_skills

router = APIRouter()

PARSER_VERSION = "mvp-1"


def serialize_candidate(document: dict | None) -> dict | None:
    if not document:
        return None
    return {
        "id": str(document["_id"]),
        "name": document.get("name", "Unnamed Candidate"),
        "email": document.get("email", ""),
        "phone": document.get("phone", ""),
        "links": document.get("links", []),
        "owner_id": document.get("owner_id", ""),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
    }


def serialize_cv(document: dict, candidate: dict | None = None, include_full_text: bool = True) -> dict:
    payload = {
        "id": str(document["_id"]),
        "candidate_id": str(document["candidate_id"]) if document.get("candidate_id") else None,
        "candidate": serialize_candidate(candidate),
        "filename": document.get("filename", ""),
        "page_count": document.get("page_count", 0),
        "char_count": document.get("char_count", 0),
        "status": document.get("status", "Ready"),
        "extraction_method": document.get("extraction_method", "rule_based"),
        "extraction_error": document.get("extraction_error", ""),
        "parser_version": document.get("parser_version", "mvp-1"),
        "raw_object_stored": document.get("raw_object_stored", False),
        "extracted_data": document.get("extracted_data", {}),
        "masked_data": document.get("masked_data", {}),
        "pii_masking": document.get("pii_masking", {"status": "pending"}),
        "owner_id": document.get("owner_id", ""),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
    }
    if include_full_text:
        payload["raw_text"] = document.get("raw_text", "")
        payload["pages"] = document.get("pages", [])
    else:
        payload["text_preview"] = document.get("raw_text", "")[:420]
    return payload


@router.post("/upload", status_code=status.HTTP_201_CREATED)
@router.post("/import", status_code=status.HTTP_201_CREATED)
async def upload_cv(file: UploadFile = File(...), current_user: dict = Depends(get_optional_user)):
    file_bytes = await file.read()
    validate_pdf_upload(file, file_bytes)

    # Idempotency: the same file (by content hash) for the same owner is not
    # re-parsed. Saves LLM cost and dedupes accidental re-imports.
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = await cv_documents_collection.find_one(
        {"owner_id": current_user["id"], "file_hash": file_hash}
    )
    if existing:
        candidate = (
            await candidates_collection.find_one({"_id": existing["candidate_id"]})
            if existing.get("candidate_id")
            else None
        )
        return serialize_cv(existing, candidate)

    try:
        extracted_pdf = extract_pdf_text(file_bytes, file.filename, max_pages=MAX_PDF_PAGES)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc) or "Could not read this PDF.") from exc

    extracted_data, method, extraction_error = await extract_cv_data_hybrid(extracted_pdf.get("full_text", ""))
    masked_data, pii_masking = mask_profile(extracted_data)
    now = datetime.now(timezone.utc)
    candidate = await upsert_candidate(extracted_data, current_user["id"], now)

    # Persist the raw PDF to object storage so it can be re-parsed later.
    object_key = await put_cv_pdf(cv_object_key(current_user["id"], file_hash), file_bytes)

    cv_document = {
        "candidate_id": candidate["_id"],
        "filename": file.filename,
        "file_hash": file_hash,
        "raw_object_key": object_key,
        "raw_object_stored": object_key is not None,
        "raw_text": extracted_pdf.get("full_text", ""),
        "pages": extracted_pdf.get("pages", []),
        "page_count": extracted_pdf.get("page_count", 0),
        "char_count": extracted_pdf.get("char_count", 0),
        "library_used": extracted_pdf.get("library_used", "PyMuPDF"),
        "extracted_data": extracted_data,
        "masked_data": masked_data,
        "pii_masking": pii_masking,
        "status": "Ready",
        "extraction_method": method,
        "extraction_error": extraction_error,
        "parser_version": PARSER_VERSION,
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now,
    }
    result = await cv_documents_collection.insert_one(cv_document)
    cv_document["_id"] = result.inserted_id
    return serialize_cv(cv_document, candidate)


@router.get("")
async def list_cvs(current_user: dict = Depends(get_optional_user)):
    cursor = cv_documents_collection.find({"owner_id": current_user["id"]}).sort("created_at", -1)
    documents = await cursor.to_list(length=100)
    candidate_ids = [document.get("candidate_id") for document in documents if document.get("candidate_id")]
    candidates = await load_candidates(candidate_ids)
    return [
        serialize_cv(document, candidates.get(document.get("candidate_id")), include_full_text=False)
        for document in documents
    ]


@router.get("/{cv_id}")
async def get_cv(cv_id: str, current_user: dict = Depends(get_optional_user)):
    document = await cv_documents_collection.find_one(
        {"_id": object_id_or_404(cv_id, "CV"), "owner_id": current_user["id"]}
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CV not found.")
    candidate = None
    if document.get("candidate_id"):
        candidate = await candidates_collection.find_one({"_id": document["candidate_id"]})
    return serialize_cv(document, candidate)


@router.put("/{cv_id}/extracted-data")
async def update_cv_extracted_data(
    cv_id: str,
    data: CVExtractedDataUpdate,
    current_user: dict = Depends(get_optional_user),
):
    object_id = object_id_or_404(cv_id, "CV")
    payload = model_payload(data)
    if "skills" in payload:
        payload["skills"] = normalize_skills(payload["skills"])
    masked_data, pii_masking = mask_profile(payload)

    document = await cv_documents_collection.find_one_and_update(
        {"_id": object_id, "owner_id": current_user["id"]},
        {"$set": {
            "extracted_data": payload,
            "masked_data": masked_data,
            "pii_masking": pii_masking,
            "status": "Ready",
            "updated_at": datetime.now(timezone.utc),
        }},
        return_document=ReturnDocument.AFTER,
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CV not found.")

    if document.get("candidate_id"):
        await candidates_collection.update_one(
            {"_id": document["candidate_id"], "owner_id": current_user["id"]},
            {
                "$set": {
                    "name": payload.get("candidate_name", "Unnamed Candidate"),
                    "email": payload.get("email", ""),
                    "phone": payload.get("phone", ""),
                    "links": payload.get("links", []),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
    candidate = await candidates_collection.find_one({"_id": document.get("candidate_id")})
    return serialize_cv(document, candidate)


@router.get("/{cv_id}/parsed")
async def get_parsed_cv(cv_id: str, current_user: dict = Depends(get_optional_user)):
    document = await cv_documents_collection.find_one(
        {"_id": object_id_or_404(cv_id, "CV"), "owner_id": current_user["id"]}
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CV not found.")
    return {
        "cv_id": cv_id,
        "status": document.get("status", "Ready"),
        "parsed": document.get("extracted_data", {}),
        "ranking_profile": document.get("masked_data", {}),
        "pii_masking": document.get("pii_masking", {}),
        "parser": {
            "method": document.get("extraction_method", "rule_based"),
            "version": document.get("parser_version", "mvp-1"),
            "error": document.get("extraction_error", ""),
        },
    }


@router.post("/{cv_id}/reparse")
async def reparse_cv(cv_id: str, current_user: dict = Depends(get_optional_user)):
    """Re-run extraction from the stored raw PDF (e.g. after a parser upgrade),
    without asking the user to re-upload."""
    object_id = object_id_or_404(cv_id, "CV")
    document = await cv_documents_collection.find_one({"_id": object_id, "owner_id": current_user["id"]})
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CV not found.")

    file_bytes = await get_cv_pdf(document.get("raw_object_key"))
    if file_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Raw PDF is not available for re-parse (object storage disabled or file missing).",
        )

    try:
        extracted_pdf = extract_pdf_text(file_bytes, document.get("filename", ""), max_pages=MAX_PDF_PAGES)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc) or "Could not read this PDF.") from exc

    extracted_data, method, extraction_error = await extract_cv_data_hybrid(extracted_pdf.get("full_text", ""))
    masked_data, pii_masking = mask_profile(extracted_data)
    now = datetime.now(timezone.utc)

    updated = await cv_documents_collection.find_one_and_update(
        {"_id": object_id, "owner_id": current_user["id"]},
        {"$set": {
            "raw_text": extracted_pdf.get("full_text", ""),
            "pages": extracted_pdf.get("pages", []),
            "page_count": extracted_pdf.get("page_count", 0),
            "char_count": extracted_pdf.get("char_count", 0),
            "extracted_data": extracted_data,
            "masked_data": masked_data,
            "pii_masking": pii_masking,
            "status": "Ready",
            "extraction_method": method,
            "extraction_error": extraction_error,
            "parser_version": PARSER_VERSION,
            "updated_at": now,
        }},
        return_document=ReturnDocument.AFTER,
    )
    candidate = await candidates_collection.find_one({"_id": updated.get("candidate_id")}) if updated.get("candidate_id") else None
    return serialize_cv(updated, candidate)


@router.delete("/{cv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cv(cv_id: str, current_user: dict = Depends(get_optional_user)):
    object_id = object_id_or_404(cv_id, "CV")
    cv = await cv_documents_collection.find_one({"_id": object_id, "owner_id": current_user["id"]})
    if not cv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CV not found.")

    await cv_documents_collection.delete_one({"_id": object_id})
    await match_results_collection.delete_many({"cv_id": object_id})
    # Best-effort cleanup of the raw object; only if no other CV references the same hash.
    object_key = cv.get("raw_object_key")
    if object_key and not await cv_documents_collection.find_one(
        {"owner_id": current_user["id"], "file_hash": cv.get("file_hash")}
    ):
        await remove_cv_pdf(object_key)
    return


async def upsert_candidate(extracted_data: dict, owner_id: str, now: datetime) -> dict:
    email = extracted_data.get("email")
    candidate_payload = {
        "name": extracted_data.get("candidate_name") or "Unnamed Candidate",
        "email": email or "",
        "phone": extracted_data.get("phone") or "",
        "links": extracted_data.get("links") or [],
        "owner_id": owner_id,
        "updated_at": now,
    }
    if email:
        candidate = await candidates_collection.find_one_and_update(
            {"email": email, "owner_id": owner_id},
            {"$set": candidate_payload, "$setOnInsert": {"created_at": now}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return candidate

    candidate_payload["created_at"] = now
    result = await candidates_collection.insert_one(candidate_payload)
    candidate_payload["_id"] = result.inserted_id
    return candidate_payload


async def load_candidates(candidate_ids: list[ObjectId]) -> dict:
    if not candidate_ids:
        return {}
    cursor = candidates_collection.find({"_id": {"$in": candidate_ids}})
    documents = await cursor.to_list(length=len(candidate_ids))
    return {document["_id"]: document for document in documents}
