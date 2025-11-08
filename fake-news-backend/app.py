from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from models import RegisterRequest, LoginRequest, TokenResponse, VerifyRequest
from db import users_col, verifications_col
from auth import get_password_hash, verify_password, create_access_token, decode_token
from predict_service import predict_text
import datetime


# ----------------------------------------------------
# Initialize FastAPI
# ----------------------------------------------------
app = FastAPI(title="🧠 Fake News Detection Backend", version="1.0")


# ----------------------------------------------------
# Allow frontend origins (React runs at port 3000)
# ----------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# Get current user from token
# ----------------------------------------------------
def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid token scheme")

    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = users_col.find_one({"email": payload.get("sub")}, {"password": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ----------------------------------------------------
# Register endpoint
# ----------------------------------------------------
@app.post("/auth/register", response_model=dict)
def register(req: RegisterRequest):
    """Registers a new user."""
    if users_col.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = get_password_hash(req.password)
    user = {
        "username": req.username,
        "email": req.email,
        "password": hashed,
        "created_at": datetime.datetime.utcnow()
    }
    users_col.insert_one(user)
    return {"msg": "✅ User registered successfully"}


# ----------------------------------------------------
# Login endpoint
# ----------------------------------------------------
@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """Authenticates the user and returns JWT token."""
    user = users_col.find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": req.email})
    return {"access_token": token}


# ----------------------------------------------------
# Verify news text (Model Prediction)
# ----------------------------------------------------
@app.post("/verify/text", response_model=dict)
def verify_news(req: VerifyRequest, current_user: dict = Depends(get_current_user)):
    """Verifies whether the provided news text is REAL or FAKE."""
    try:
        result = predict_text(req.text)  # Call prediction model

        record = {
            "user_email": current_user["email"],
            "text": req.text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "timestamp": datetime.datetime.utcnow()
        }
        verifications_col.insert_one(record)

        return {
            "msg": "✅ Verification complete",
            "data": record
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ----------------------------------------------------
# Fetch user verification history
# ----------------------------------------------------
@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    """Returns user’s previous verification results."""
    docs = list(verifications_col.find(
        {"user_email": current_user["email"]},
        {"_id": 0}
    ).sort("timestamp", -1))

    return {"count": len(docs), "history": docs}


# ----------------------------------------------------
# Root Endpoint
# ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "🚀 Fake News Detection Backend is running successfully!"}
