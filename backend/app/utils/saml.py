"""
SAML Authentication Module
Fully isolated FastAPI router
"""

import os
import datetime
from pathlib import Path
from typing import Dict, Any

import jwt
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from jwt import ExpiredSignatureError, InvalidTokenError

# ---------------------------
# Router
# ---------------------------

router = APIRouter(prefix="/saml", tags=["SAML"])

# ---------------------------
# Config
# ---------------------------

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "VectorIQ#Dev")  # change in prod

# Resolve app/saml path correctly
BASE_DIR = Path(__file__).resolve().parent.parent  # -> app/
SAML_PATH = str(BASE_DIR / "saml")


# ---------------------------
# Helpers
# ---------------------------

def prepare_fastapi_request(request: Request) -> Dict[str, Any]:
    """
    Converts FastAPI request into format required by python3-saml
    """
    form = {}

    # Only parse form for POST requests
    if request.method == "POST":
        # We must read form asynchronously later in route
        pass

    return {
        "https": "on" if request.url.scheme == "https" else "off",
        "http_host": request.headers.get("host"),
        "script_name": request.url.path,
        "server_port": request.url.port or ("443" if request.url.scheme == "https" else "80"),
        "get_data": dict(request.query_params),
        "post_data": form,
    }


def init_saml_auth(req: Dict[str, Any]) -> OneLogin_Saml2_Auth:
    """
    Initialize SAML Auth object
    """
    return OneLogin_Saml2_Auth(req, custom_base_path=SAML_PATH)


def create_jwt_token(user_data: Dict[str, Any]) -> str:
    """
    Create JWT token valid for 1 hour
    """
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=1)

    payload = {
        "user_data": user_data,
        "exp": expiration,
    }

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
    return token


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT
    """
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])


# ---------------------------
# Route 1: Login
# ---------------------------

@router.get("/login")
async def saml_login(request: Request):
    """
    Initiates SAML login
    """
    req = prepare_fastapi_request(request)
    auth = init_saml_auth(req)

    login_url = auth.login()
    return RedirectResponse(login_url)


# ---------------------------
# Route 2: Callback (ACS)
# ---------------------------

@router.post("/callback")
async def saml_callback(request: Request):
    """
    Handles SAML Assertion Consumer Service (ACS)
    """
    # Read form data
    form_data = await request.form()

    req = prepare_fastapi_request(request)
    req["post_data"] = dict(form_data)

    auth = init_saml_auth(req)

    auth.process_response()
    errors = auth.get_errors()

    if errors:
        raise HTTPException(status_code=400, detail=f"SAML Error: {errors}")

    if not auth.is_authenticated():
        raise HTTPException(status_code=401, detail="SAML authentication failed")

    # Extract attributes
    attributes = auth.get_attributes()

    email_list = attributes.get(
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
        []
    )
    email = email_list[0] if email_list else "no-email@example.com"

    job_title_list = attributes.get(
        "http://schemas.microsoft.com/identity/claims/jobtitle",
        []
    )
    job_title = job_title_list[0] if job_title_list else "No Job Title"

    user_data = {
        "name": auth.get_nameid() or "Unknown User",
        "group": "user",
        "email": email,
        "job_title": job_title,
    }

    token = create_jwt_token(user_data)

    # Redirect to frontend
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
    return RedirectResponse(f"{frontend_url}/dashboard?token={token}")


# ---------------------------
# Route 3: Validate Token
# ---------------------------

@router.get("/validate")
async def validate_token(token: str):
    """
    Validates JWT and returns user data
    """
    try:
        decoded = decode_jwt_token(token)
        return JSONResponse(content={"user_data": decoded.get("user_data")})

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")

    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
