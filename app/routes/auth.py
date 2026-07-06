from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
from app.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    GUEST_SESSION_COOKIE_NAME,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    RATE_LIMIT_AUTH,
)
from app.database import (
    audit_logs_collection,
    candidates_collection,
    cv_documents_collection,
    fairness_attributes_collection,
    jobs_collection,
    match_feedback_collection,
    match_jobs_collection,
    match_results_collection,
    ranking_models_collection,
    users_collection,
)
from app.models.user import UserRegister, UserLogin, UserResponse, Token
from app.rate_limit import limiter
from app.services.guest_session import decode_guest_token

router = APIRouter()
security = HTTPBearer()


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pwd_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pwd_bytes, hashed_bytes)


def create_access_token(data: dict):
    """Tạo JWT access token với thời hạn."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Dependency: xác thực JWT token và trả về user hiện tại."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None or not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=401, detail="Token không hợp lệ")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")

    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise HTTPException(status_code=401, detail="User không tồn tại")

    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
    }


@router.post("/register", response_model=UserResponse)
@limiter.limit(RATE_LIMIT_AUTH)
async def register(request: Request, user: UserRegister):
    """Đăng ký tài khoản mới."""
    # Check email trùng
    existing = await users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email đã tồn tại")

    # Hash password
    hashed = hash_password(user.password)

    # Insert
    result = await users_collection.insert_one(
        {
            "username": user.username,
            "email": user.email,
            "password": hashed,
            "created_at": datetime.now(timezone.utc),
        }
    )

    return UserResponse(
        id=str(result.inserted_id),
        username=user.username,
        email=user.email,
    )


@router.post("/login", response_model=Token)
@limiter.limit(RATE_LIMIT_AUTH)
async def login(request: Request, user: UserLogin):
    """Đăng nhập và nhận JWT token."""
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(
            status_code=401, detail="Email hoặc mật khẩu không đúng"
        )

    token = create_access_token({"sub": str(db_user["_id"])})
    return Token(access_token=token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Lấy thông tin user hiện tại từ token."""
    return UserResponse(**current_user)


# Every Mongo collection that stores an owner_id created during a guest session.
# Rewriting owner_id on all of them lets the user see their pre-signup work.
_GUEST_OWNED_COLLECTIONS = [
    candidates_collection,
    jobs_collection,
    match_results_collection,
    match_jobs_collection,
    match_feedback_collection,
    fairness_attributes_collection,
    ranking_models_collection,
    audit_logs_collection,
]


@router.post("/claim-guest-session")
@limiter.limit(RATE_LIMIT_AUTH)
async def claim_guest_session(
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    """Rewrite guest-owned records to the authenticated user.

    The frontend calls this immediately after register/login so anonymous
    uploads (CVs, JDs, matches, feedback) are not lost when a user signs in.
    Idempotent: once the guest cookie is cleared or the guest_id owns nothing,
    subsequent calls become no-ops.
    """
    token = request.cookies.get(GUEST_SESSION_COOKIE_NAME)
    if not token:
        return {"claimed": False, "reason": "no_guest_session"}
    session_id = decode_guest_token(token)
    if not session_id:
        response.delete_cookie(GUEST_SESSION_COOKIE_NAME)
        return {"claimed": False, "reason": "invalid_guest_session"}

    guest_owner_id = f"guest:{session_id}"
    user_owner_id = current_user["id"]

    # CVs: the (owner_id, file_hash) unique index would reject the update if the
    # user has already uploaded a matching file. Drop guest duplicates first so
    # the surviving user copy keeps its history.
    user_hashes_cursor = cv_documents_collection.find(
        {"owner_id": user_owner_id, "file_hash": {"$type": "string"}},
        {"file_hash": 1},
    )
    user_hashes = {doc["file_hash"] async for doc in user_hashes_cursor if doc.get("file_hash")}
    dropped_cvs = 0
    if user_hashes:
        drop_result = await cv_documents_collection.delete_many(
            {"owner_id": guest_owner_id, "file_hash": {"$in": list(user_hashes)}}
        )
        dropped_cvs = drop_result.deleted_count

    try:
        cv_update = await cv_documents_collection.update_many(
            {"owner_id": guest_owner_id},
            {"$set": {"owner_id": user_owner_id}},
        )
        cv_moved = cv_update.modified_count
    except DuplicateKeyError:
        # A concurrent user upload landed the same hash between the dedupe scan
        # and this update. The guest record now can't merge — drop it.
        await cv_documents_collection.delete_many({"owner_id": guest_owner_id})
        cv_moved = 0

    counts: dict[str, int] = {"cv_documents": cv_moved}
    for coll in _GUEST_OWNED_COLLECTIONS:
        result = await coll.update_many(
            {"owner_id": guest_owner_id},
            {"$set": {"owner_id": user_owner_id}},
        )
        counts[coll.name] = result.modified_count

    response.delete_cookie(GUEST_SESSION_COOKIE_NAME)
    total = sum(counts.values()) + dropped_cvs
    return {
        "claimed": total > 0,
        "moved": counts,
        "dropped_duplicate_cvs": dropped_cvs,
    }
