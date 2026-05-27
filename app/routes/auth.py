from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from bson import ObjectId
from app.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.database import users_collection
from app.models.user import UserRegister, UserLogin, UserResponse, Token

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
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
        if user_id is None:
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
async def register(user: UserRegister):
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
            "created_at": datetime.utcnow(),
        }
    )

    return UserResponse(
        id=str(result.inserted_id),
        username=user.username,
        email=user.email,
    )


@router.post("/login", response_model=Token)
async def login(user: UserLogin):
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
