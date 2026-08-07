"""依赖注入：JWT 解析 + 当前用户 / 管理员校验。

用法:
    async def endpoint(user: dict = Depends(get_current_user)): ...
    async def admin_endpoint(user: dict = Depends(require_admin)): ...
"""
import jwt
from fastapi import Depends, Header, HTTPException

from app.config import settings

ALGORITHM = "HS256"


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的登录凭证")


def get_current_user(authorization: str = Header(default="")) -> dict:
    """解析 Bearer token，返回 payload dict（含 sub/username/role）。"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少认证信息")
    return _decode_token(authorization[7:])


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """校验当前用户 role == admin。"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
