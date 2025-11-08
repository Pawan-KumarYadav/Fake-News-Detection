from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
import os
from dotenv import load_dotenv

load_dotenv()
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALG = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

def get_password_hash(password: str) -> str:
    return PWD_CONTEXT.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return PWD_CONTEXT.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: int = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=(expires_delta or ACCESS_EXPIRE_MIN))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)
    return encoded_jwt

def decode_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except Exception:
        return None
