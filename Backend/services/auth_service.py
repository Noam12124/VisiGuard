from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET
from database.models import User
from repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, db: Session) -> None:
        self._users = UserRepository(db)

    @staticmethod
    def hash_password(password: str) -> str:
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )

    @staticmethod
    def create_access_token(user_id: int, username: str) -> str:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
        payload = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> dict:
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except JWTError as exc:
            raise ValueError("Invalid or expired token") from exc

    def signup(self, username: str, email: str, password: str) -> tuple[User, str]:
        username = username.strip()
        email = email.strip().lower()

        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")

        if self._users.get_by_username(username):
            raise ValueError("Username already registered.")
        if self._users.get_by_email(email):
            raise ValueError("Email already registered.")

        user = self._users.create(
            username=username,
            email=email,
            password_hash=self.hash_password(password),
        )
        token = self.create_access_token(user.id, user.username)
        logger.info("User registered: %s (id=%s)", user.username, user.id)
        return user, token

    def login(self, username: str, password: str) -> tuple[User, str]:
        user = self._users.get_by_username(username.strip())
        if user is None or not self.verify_password(password, user.password_hash):
            raise ValueError("Invalid username or password.")

        token = self.create_access_token(user.id, user.username)
        logger.info("User logged in: %s", user.username)
        return user, token

    def get_user_from_token(self, token: str) -> User:
        payload = self.decode_token(token)
        user_id = int(payload["sub"])
        user = self._users.get_by_id(user_id)
        if user is None:
            raise ValueError("User not found.")
        return user
