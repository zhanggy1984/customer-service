"""注册 / 登录认证接口（JWT）。

- POST /api/v1/auth/register  注册（role 固定 user，由 DB 默认值保证，不可自选）
- POST /api/v1/auth/login     登录，返回 JWT（payload 含 sub/username/role）
"""
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.infrastructure.mysql import mysql_pool
from app.utils.logger import logger

router = APIRouter(prefix="/auth", tags=["auth"])

ALGORITHM = "HS256"


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=64)
    phone: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def _hash_password(raw: str) -> str:
    return bcrypt.hashpw(raw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(raw.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # 存量数据 hash 格式不合法（如非 bcrypt）时视为密码错误
        return False


def _create_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expire_hours),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


@router.post("/register", status_code=201)
async def register(req: RegisterRequest) -> dict:
    exist = await mysql_pool.fetchone(
        "SELECT id FROM users WHERE username=%s", (req.username,)
    )
    if exist:
        logger.info("auth_register_fail", extra={"username": req.username, "reason": "username_exists"})
        raise HTTPException(status_code=409, detail="用户名已存在")

    hashed = _hash_password(req.password)
    await mysql_pool.execute(
        "INSERT INTO users (username, password_hash, phone) VALUES (%s, %s, %s)",
        (req.username, hashed, req.phone),
    )
    logger.info("auth_register_ok", extra={"username": req.username})
    return {"msg": "注册成功"}


@router.post("/login")
async def login(req: LoginRequest) -> TokenResponse:
    row = await mysql_pool.fetchone(
        "SELECT id, username, password_hash, role, phone FROM users WHERE username=%s",
        (req.username,),
    )
    if not row or not _verify_password(req.password, row["password_hash"]):
        logger.info("auth_login_fail", extra={"username": req.username, "reason": "bad_credentials"})
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = _create_token(row)
    logger.info("auth_login_ok", extra={"username": req.username, "role": row["role"]})
    return TokenResponse(
        access_token=token,
        user={
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "phone": row["phone"],
        },
    )
