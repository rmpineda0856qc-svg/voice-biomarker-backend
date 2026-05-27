"""Authentication endpoints: register and login."""
from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends

from app import database, schemas
from app.services.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=schemas.Token)
def register(body: schemas.UserRegister):
    """Create a new user account and return a JWT token."""
    existing = database.get_user_by_email(body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    try:
        user_id = database.create_user(
            email=body.email,
            password_hash=hash_password(body.password),
            age=body.age,
            gender=body.gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token = create_access_token(user_id)
    return schemas.Token(access_token=token, user_id=user_id)


@router.post("/login", response_model=schemas.Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Login with email (sent as 'username') and password. Returns a JWT token."""
    user = database.get_user_by_email(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token(user["id"])
    return schemas.Token(access_token=token, user_id=user["id"])