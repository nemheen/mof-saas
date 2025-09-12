# auth.py
import os, time
from datetime import timedelta, datetime, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User

SECRET_KEY = os.getenv("SECRET_KEY", "2a186f83bce5a618718f7b6d50eb6d2d8837077b6f0b9ed43cefb6fed4f6ba44")
JWT_ALG = "HS256"
ACCESS_TTL_MIN = int(os.getenv("ACCESS_TTL_MIN", "20"))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", "7"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def hash_pw(p: str) -> str:
    return pwd_ctx.hash(p)

def verify_pw(p: str, hp: str) -> bool:
    return pwd_ctx.verify(p, hp)

def _create_jwt(sub: str, ttl: timedelta, scope: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "scope": scope, "iat": int(now.timestamp()), "exp": int((now + ttl).timestamp())}
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALG)

def create_access_token(sub: str) -> str:
    return _create_jwt(sub, timedelta(minutes=ACCESS_TTL_MIN), scope="access")

def create_refresh_token(sub: str) -> str:
    return _create_jwt(sub, timedelta(days=REFRESH_TTL_DAYS), scope="refresh")

def set_refresh_cookie(resp: Response, token: str):
    # In prod set secure=True and samesite="lax" or "none" (with HTTPS)
    resp.set_cookie(
        key="refresh_token",
        value=token,
        httponly=True,
        secure=bool(os.getenv("COOKIE_SECURE", "0") == "1"),
        samesite=os.getenv("COOKIE_SAMESITE", "lax"),
        path="/auth/refresh",
        max_age=REFRESH_TTL_DAYS * 24 * 3600,
    )

def clear_refresh_cookie(resp: Response):
    resp.delete_cookie("refresh_token", path="/auth/refresh")

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALG])

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = decode_token(token)
        if payload.get("scope") != "access":
            raise cred_exc
        email = payload.get("sub")
    except JWTError:
        raise cred_exc
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise cred_exc
    return user
