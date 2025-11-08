from pydantic import BaseModel, EmailStr
from typing import Optional

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class VerifyRequest(BaseModel):
    text: str

class VerifyResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
