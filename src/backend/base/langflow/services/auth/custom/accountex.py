"""AccountexAI Custom Auth Service for LangFlow.

This auth service integrates LangFlow with AccountexAI's JWT-based authentication.
It validates JWTs from our auth system and performs JIT provisioning of users.

Configure via lfx.toml:
    auth_service = "langflow.services.auth.custom.accountex:AccountexAuthService"

Environment variables required:
    ACCOUNTEX_JWT_SECRET: Secret for validating AccountexAI JWTs
    ACCOUNTEX_JWT_ISSUER: Expected issuer claim (e.g., https://accountexai-api-production.up.railway.app)
    LANGFLOW_AUTO_LOGIN: Must be true for this to work
    LANGFLOW_SUPERUSER: Default superuser to use (e.g., accountexai@system.local)
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

import jwt
from jwt import InvalidTokenError
from lfx.log.logger import logger
from sqlalchemy.exc import IntegrityError

from langflow.helpers.user import get_user_by_flow_id_or_endpoint_name
from langflow.services.auth.base import BaseAuthService
from langflow.services.auth.constants import AUTO_LOGIN_ERROR, AUTO_LOGIN_WARNING
from langflow.services.auth.exceptions import (
    InactiveUserError,
    InvalidCredentialsError,
    MissingCredentialsError,
    TokenExpiredError,
)
from langflow.services.auth.exceptions import (
    InvalidTokenError as AuthInvalidTokenError,
)
from langflow.services.database.models.api_key.crud import check_key
from langflow.services.database.models.user.crud import (
    get_user_by_id,
    get_user_by_username,
    update_user_last_login_at,
)
from langflow.services.database.models.user.model import User, UserRead
from langflow.services.deps import session_scope
from langflow.services.schema import ServiceType

if TYPE_CHECKING:
    from lfx.services.settings.service import SettingsService
    from sqlmodel.ext.asyncio.session import AsyncSession

    from langflow.services.database.models.api_key.model import ApiKey


class AccountexAuthService(BaseAuthService):
    """Custom auth service for AccountexAI integration.

    Validates JWTs from AccountexAI's auth system and creates LangFlow users
    on-demand (JIT provisioning).
    """

    name = ServiceType.AUTH_SERVICE.value

    def __init__(self, settings_service: SettingsService):
        self.settings_service = settings_service
        self.jwt_secret = os.getenv("ACCOUNTEX_JWT_SECRET", "accountexai-jwt-secret-32chars-minimum")
        self.jwt_issuer = os.getenv("ACCOUNTEX_JWT_ISSUER", "https://accountexai-api-production.up.railway.app")
        self.jwt_algorithm = "HS256"
        self.set_ready()
        logger.info("AccountexAuthService initialized with issuer: %s", self.jwt_issuer)

    @property
    def settings(self) -> SettingsService:
        return self.settings_service

    async def authenticate_with_credentials(
        self,
        token: str | None,
        api_key: str | None,
        db: AsyncSession,
    ) -> User | UserRead:
        """Authenticate using AccountexAI JWT or API key.

        Args:
            token: AccountexAI JWT access token
            api_key: LangFlow API key (fallback)
            db: Database session

        Returns:
            User or UserRead object

        Raises:
            MissingCredentialsError: No credentials provided
            InvalidCredentialsError: Invalid credentials
            InvalidTokenError: Invalid JWT
            TokenExpiredError: JWT expired
            InactiveUserError: User inactive
        """
        if token:
            try:
                return await self._authenticate_with_accountex_token(token, db)
            except (AuthInvalidTokenError, TokenExpiredError, InactiveUserError, InvalidCredentialsError):
                raise
            except Exception as e:
                logger.error("AccountexAI token auth failed: %s", str(e))
                if api_key:
                    try:
                        user = await self._authenticate_with_api_key(api_key, db)
                        if user:
                            return user
                        raise InvalidCredentialsError("Invalid API key")
                    except InvalidCredentialsError:
                        raise
                    except Exception as api_key_err:
                        logger.error("API key auth also failed: %s", str(api_key_err))
                        raise InvalidCredentialsError("Authentication failed") from api_key_err
                raise AuthInvalidTokenError("Token authentication failed") from e

        if api_key:
            try:
                user = await self._authenticate_with_api_key(api_key, db)
                if user:
                    return user
                raise InvalidCredentialsError("Invalid API key")
            except InvalidCredentialsError:
                raise
            except Exception as e:
                logger.error("API key auth failed: %s", str(e))
                raise InvalidCredentialsError("API key authentication failed") from e

        msg = "No authentication credentials provided"
        raise MissingCredentialsError(msg)

    async def _authenticate_with_accountex_token(self, token: str, db: AsyncSession) -> User:
        """Validate AccountexAI JWT and return/create user.

        Args:
            token: JWT from AccountexAI
            db: Database session

        Returns:
            User object

        Raises:
            InvalidTokenError: Invalid JWT
            TokenExpiredError: JWT expired
            InvalidCredentialsError: Token valid but user creation failed
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                payload = jwt.decode(
                    token,
                    self.jwt_secret,
                    algorithms=[self.jwt_algorithm],
                    options={"verify_iss": True, "verify_exp": True},
                    issuer=self.jwt_issuer,
                )
        except jwt.ExpiredSignatureError:
            logger.info("AccountexAI token has expired")
            msg = "Token has expired"
            raise TokenExpiredError(msg)
        except InvalidTokenError as e:
            logger.debug("AccountexAI JWT validation failed: %s", str(e))
            msg = "Invalid token"
            raise AuthInvalidTokenError(msg) from e
        except Exception as e:
            logger.error("Unexpected error decoding AccountexAI token: %s", str(e))
            msg = "Token validation failed"
            raise AuthInvalidTokenError(msg) from e

        user_id: str | None = payload.get("sub")
        email: str | None = payload.get("email")
        name: str | None = payload.get("name")
        role: str | None = payload.get("role", "VIEWER")

        if not user_id or not email:
            logger.info("Invalid token payload: missing sub or email")
            msg = "Invalid token payload"
            raise AuthInvalidTokenError(msg)

        is_superuser = role in ("ADMIN", "MANAGER")

        user = await self._get_or_create_user(
            user_id=user_id,
            email=email,
            name=name,
            is_superuser=is_superuser,
            db=db,
        )

        return user

    async def _get_or_create_user(
        self,
        user_id: str,
        email: str,
        name: str | None,
        is_superuser: bool,
        db: AsyncSession,
    ) -> User:
        """Get existing user or create new one (JIT provisioning).

        Args:
            user_id: AccountexAI user ID
            email: User email
            name: User display name
            is_superuser: Whether user has admin/manager role
            db: Database session

        Returns:
            User object
        """
        existing_user = await get_user_by_username(db, email)
        if existing_user:
            if not existing_user.is_active:
                msg = "User account is inactive"
                raise InactiveUserError(msg)
            logger.debug("Found existing AccountexAI user: %s", email)
            return existing_user

        logger.info("Creating new LangFlow user for AccountexAI user: %s", email)
        new_user = User(
            username=email,
            email=email,
            password="",  # No password - auth via JWT only
            name=name or email.split("@")[0],
            is_superuser=is_superuser,
            is_active=True,
            last_login_at=datetime.now(timezone.utc),
        )

        db.add(new_user)
        try:
            await db.commit()
            await db.refresh(new_user)
            logger.info("Created new LangFlow user: %s (id=%s)", email, str(new_user.id))
            return new_user
        except IntegrityError:
            await db.rollback()
            existing_user = await get_user_by_username(db, email)
            if existing_user:
                return existing_user
            msg = "Failed to create user"
            raise InvalidCredentialsError(msg)
        except Exception as e:
            await db.rollback()
            logger.error("Error creating AccountexAI user: %s", str(e))
            msg = "Failed to create user"
            raise InvalidCredentialsError(msg) from e

    async def _authenticate_with_api_key(self, api_key: str, db: AsyncSession) -> UserRead | None:
        """Authenticate using LangFlow API key."""
        result = await check_key(db, api_key)
        if not result:
            return None

        if isinstance(result, User):
            user_read = UserRead.model_validate(result, from_attributes=True)
            if not user_read.is_active:
                msg = "User account is inactive"
                raise InactiveUserError(msg)
            return user_read

        return None

    async def api_key_security(
        self, query_param: str | None, header_param: str | None, db: AsyncSession | None = None
    ) -> UserRead | None:
        """API key validation with AccountexAI auto-login support."""
        settings_service = self.settings

        if db is not None:
            return await self._api_key_security_impl(query_param, header_param, db, settings_service)

        async with session_scope() as new_db:
            return await self._api_key_security_impl(query_param, header_param, new_db, settings_service)

    async def _api_key_security_impl(
        self,
        query_param: str | None,
        header_param: str | None,
        db: AsyncSession,
        settings_service,
    ) -> UserRead | None:
        """Internal API key validation."""
        if settings_service.auth_settings.AUTO_LOGIN:
            if not query_param and not header_param:
                if settings_service.auth_settings.skip_auth_auto_login:
                    result = await get_user_by_username(db, settings_service.auth_settings.SUPERUSER)
                    if result:
                        logger.warning(AUTO_LOGIN_WARNING)
                        return UserRead.model_validate(result, from_attributes=True)

            api_key = query_param or header_param
            if api_key:
                result = await check_key(db, api_key)
                if isinstance(result, User):
                    return UserRead.model_validate(result, from_attributes=True)

        elif query_param or header_param:
            api_key = query_param or header_param
            if api_key:
                result = await check_key(db, api_key)
                if isinstance(result, User):
                    return UserRead.model_validate(result, from_attributes=True)

        return None

    async def ws_api_key_security(self, api_key: str | None) -> UserRead:
        """WebSocket API key validation."""
        settings = self.settings
        async with session_scope() as db:
            if settings.auth_settings.AUTO_LOGIN:
                if not api_key:
                    if settings.auth_settings.skip_auth_auto_login:
                        result = await get_user_by_username(db, settings.auth_settings.SUPERUSER)
                        if result:
                            logger.warning(AUTO_LOGIN_WARNING)
                            return UserRead.model_validate(result, from_attributes=True)

            if api_key:
                result = await check_key(db, api_key)
                if isinstance(result, User):
                    return UserRead.model_validate(result, from_attributes=True)

        from fastapi import WebSocketException, status
        raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR, reason="Authentication failed")

    async def get_current_user(
        self,
        token: str | Coroutine | None,
        query_param: str | None,
        header_param: str | None,
        db: AsyncSession,
    ) -> User | UserRead:
        """Get current user from token or API key."""
        resolved_token: str | None = None
        if isinstance(token, Coroutine):
            resolved_token = await token
        elif isinstance(token, str):
            resolved_token = token

        api_key = query_param or header_param
        return await self.authenticate_with_credentials(resolved_token, api_key, db)

    async def get_current_user_from_access_token(
        self, token: str | Coroutine | None, db: AsyncSession
    ) -> User:
        """Get user from access token."""
        if token is None:
            msg = "Missing authentication token"
            raise MissingCredentialsError(msg)

        resolved_token: str
        if isinstance(token, Coroutine):
            resolved_token = await token
        elif isinstance(token, str):
            resolved_token = token
        else:
            msg = "Invalid token format"
            raise AuthInvalidTokenError(msg)

        return await self._authenticate_with_accountex_token(resolved_token, db)

    async def get_current_user_for_websocket(
        self, token: str | None, api_key: str | None, db: AsyncSession
    ) -> User | UserRead:
        """Get current user for WebSocket."""
        return await self.authenticate_with_credentials(token, api_key, db)

    async def get_current_user_for_sse(
        self, token: str | None, api_key: str | None, db: AsyncSession
    ) -> User | UserRead:
        """Get current user for SSE."""
        return await self.authenticate_with_credentials(token, api_key, db)

    async def get_current_active_user(self, current_user: User | UserRead) -> User | UserRead | None:
        """Check if user is active."""
        if not current_user.is_active:
            return None
        return current_user

    async def get_current_active_superuser(self, current_user: User | UserRead) -> User | UserRead | None:
        """Check if user is active superuser."""
        if not current_user.is_active or not current_user.is_superuser:
            return None
        return current_user

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Password verification not used with JWT auth."""
        msg = "Password verification not supported with AccountexAI JWT auth"
        raise NotImplementedError(msg)

    def get_password_hash(self, password: str) -> str:
        """Password hashing not used with JWT auth."""
        msg = "Password hashing not supported with AccountexAI JWT auth"
        raise NotImplementedError(msg)

    def create_token(self, data: dict[str, Any], expires_delta: timedelta) -> str:
        """Create JWT token (delegates to LangFlow's signing)."""
        from langflow.services.auth.utils import get_jwt_signing_key

        settings_service = self.settings
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + expires_delta
        to_encode["exp"] = expire

        signing_key = get_jwt_signing_key(settings_service)
        return jwt.encode(
            to_encode,
            signing_key,
            algorithm=settings_service.auth_settings.ALGORITHM,
        )

    async def create_user_tokens(self, user_id: UUID, db: AsyncSession, *, update_last_login: bool = False) -> dict[str, Any]:
        """Create auth tokens for user."""
        settings_service = self.settings

        access_token_expires = timedelta(seconds=settings_service.auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS)
        access_token = self.create_token(
            data={"sub": str(user_id), "type": "access"},
            expires_delta=access_token_expires,
        )

        refresh_token_expires = timedelta(seconds=settings_service.auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS)
        refresh_token = self.create_token(
            data={"sub": str(user_id), "type": "refresh"},
            expires_delta=refresh_token_expires,
        )

        if update_last_login:
            await update_user_last_login_at(user_id, db)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }

    async def create_refresh_token(self, refresh_token: str, db: AsyncSession):
        """Create new tokens from refresh token."""
        from langflow.services.auth.utils import get_jwt_verification_key

        settings_service = self.settings
        algorithm = settings_service.auth_settings.ALGORITHM
        verification_key = get_jwt_verification_key(settings_service)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                payload = jwt.decode(
                    refresh_token,
                    verification_key,
                    algorithms=[algorithm],
                )
            user_id: UUID = payload.get("sub")  # type: ignore[assignment]
            token_type: str = payload.get("type")  # type: ignore[assignment]

            if user_id is None or token_type != "refresh":
                raise InvalidCredentialsError("Invalid refresh token")

            user_exists = await get_user_by_id(db, user_id)
            if user_exists is None:
                raise InvalidCredentialsError("Invalid refresh token")

            if not user_exists.is_active:
                raise InvalidCredentialsError("User account inactive")

            return await self.create_user_tokens(user_id, db)

        except InvalidTokenError as e:
            logger.exception("JWT decoding error")
            raise InvalidCredentialsError("Invalid refresh token") from e

    async def authenticate_user(self, username: str, password: str, db: AsyncSession) -> User | None:
        """Username/password auth not used with JWT."""
        logger.debug("Username/password auth not supported with AccountexAI JWT")
        return None

    async def create_super_user(self, username: str, password: str, db: AsyncSession) -> User:
        """Create superuser (for initial setup)."""
        super_user = await get_user_by_username(db, username)

        if not super_user:
            password_hash = self.get_password_hash(password) if password else ""
            super_user = User(
                username=username,
                email=username,
                password=password_hash,
                is_superuser=True,
                is_active=True,
            )
            db.add(super_user)
            try:
                await db.commit()
                await db.refresh(super_user)
            except IntegrityError:
                await db.rollback()
                super_user = await get_user_by_username(db, username)
                if not super_user:
                    raise

        return super_user

    async def create_user_longterm_token(self, db: AsyncSession) -> tuple[UUID, dict]:
        """Create long-term token for auto-login."""
        settings_service = self.settings
        if not settings_service.auth_settings.AUTO_LOGIN:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Auto login required to create long-term token",
            )

        username = settings_service.auth_settings.SUPERUSER
        super_user = await get_user_by_username(db, username)
        if not super_user:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Super user hasn't been created")

        access_token_expires_longterm = timedelta(days=365)
        access_token = self.create_token(
            data={"sub": str(super_user.id), "type": "access"},
            expires_delta=access_token_expires_longterm,
        )

        await update_user_last_login_at(super_user.id, db)

        return super_user.id, {
            "access_token": access_token,
            "refresh_token": None,
            "token_type": "bearer",
        }

    def create_user_api_key(self, user_id: UUID) -> dict:
        """Create API key for user."""
        access_token = self.create_token(
            data={"sub": str(user_id), "type": "api_key"},
            expires_delta=timedelta(days=365 * 2),
        )
        return {"api_key": access_token}

    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for storage."""
        from langflow.services.auth.utils import ensure_fernet_key
        from cryptography.fernet import Fernet

        secret_key: str = self.settings.auth_settings.SECRET_KEY.get_secret_value()
        fernet = Fernet(ensure_fernet_key(secret_key))
        return fernet.encrypt(api_key.encode()).decode()

    def decrypt_api_key(self, encrypted_api_key: str) -> str:
        """Decrypt stored API key."""
        from langflow.services.auth.utils import ensure_fernet_key
        from cryptography.fernet import Fernet

        if not encrypted_api_key or not encrypted_api_key.startswith("gAAAAA"):
            return encrypted_api_key

        secret_key: str = self.settings.auth_settings.SECRET_KEY.get_secret_value()
        fernet = Fernet(ensure_fernet_key(secret_key))
        try:
            return fernet.decrypt(encrypted_api_key.encode()).decode()
        except Exception:
            return ""

    async def get_webhook_user(self, flow_id: str, request: Any) -> UserRead:
        """Get user for webhook execution."""
        settings_service = self.settings

        if not settings_service.auth_settings.WEBHOOK_AUTH_ENABLE:
            try:
                flow_owner = await get_user_by_flow_id_or_endpoint_name(flow_id)
                if flow_owner is None:
                    from fastapi import HTTPException, status
                    raise HTTPException(status_code=404, detail="Flow not found")
                return flow_owner
            except Exception:
                from fastapi import HTTPException, status
                raise HTTPException(status_code=404, detail="Flow not found")

        api_key_header_val = request.headers.get("x-api-key")
        api_key_query_val = request.query_params.get("x-api-key")
        api_key = api_key_header_val or api_key_query_val

        if not api_key:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=403, detail="API key required")

        try:
            async with session_scope() as db:
                result = await check_key(db, api_key)
                if not result:
                    logger.warning("Invalid API key for webhook")
                    from fastapi import HTTPException, status
                    raise HTTPException(status_code=403, detail="Invalid API key")

                authenticated_user = UserRead.model_validate(result, from_attributes=True)
        except Exception as e:
            logger.error("Webhook auth error: %s", str(e))
            from fastapi import HTTPException, status
            raise HTTPException(status_code=403, detail="Authentication failed") from e

        try:
            flow_owner = await get_user_by_flow_id_or_endpoint_name(flow_id)
            if flow_owner is None:
                from fastapi import HTTPException, status
                raise HTTPException(status_code=404, detail="Flow not found")
        except Exception:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=404, detail="Flow not found")

        if flow_owner.id != authenticated_user.id:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=403, detail="Access denied")

        return authenticated_user

    async def get_current_user_mcp(
        self,
        token: str | Coroutine | None,
        query_param: str | None,
        header_param: str | None,
        db: AsyncSession,
    ) -> User | UserRead:
        """Get current user for MCP."""
        if token:
            return await self.get_current_user_from_access_token(token, db)

        settings_service = self.settings
        if settings_service.auth_settings.AUTO_LOGIN:
            if not query_param and not header_param:
                result = await get_user_by_username(db, settings_service.auth_settings.SUPERUSER)
                if result:
                    logger.warning(AUTO_LOGIN_WARNING)
                    return result

            api_key = query_param or header_param
            if api_key:
                result = await check_key(db, api_key)
                if isinstance(result, User):
                    return result

        from fastapi import HTTPException, status
        raise HTTPException(status_code=403, detail="Authentication required")

    async def get_current_active_user_mcp(self, current_user: User | UserRead) -> User | UserRead:
        """Validate MCP user is active."""
        if not current_user.is_active:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=401, detail="Inactive user")
        return current_user

    async def teardown(self) -> None:
        """Teardown (no-op for JWT auth)."""
        logger.debug("AccountexAuthService teardown")
