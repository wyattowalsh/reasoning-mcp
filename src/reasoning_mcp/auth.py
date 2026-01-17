"""JWT Bearer token and API key authentication for reasoning-mcp server.

This module provides authentication middleware and helper functions
for the MCPaaS (MCP as a Service) feature set. It supports both JWT Bearer
token authentication and API key authentication.

Features:
- JWT token generation with configurable expiration
- JWT token validation with signature and expiration checks
- Bearer token extraction from Authorization header
- API key generation, validation, and revocation
- User context injection into requests
- FastMCP middleware integration

Example:
    >>> from reasoning_mcp.auth import JWTAuthMiddleware, create_token
    >>> # Generate a JWT token
    >>> token = create_token(
    ...     user_id="user123",
    ...     secret_key="your-secret-key",
    ...     expire_minutes=30
    ... )
    >>> # Generate an API key
    >>> api_key = generate_api_key()
    >>> register_api_key(api_key, user_id="user123", permissions=["read", "write"])
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import jwt
from jwt import DecodeError, ExpiredSignatureError, InvalidTokenError
from mcp.types import CallToolRequest  # noqa: TCH002
from pydantic import SecretStr

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp import FastMCP

    from reasoning_mcp.config import Settings


def _get_api_key_value(api_key: str | SecretStr) -> str:
    """Extract the raw string value from an API key.

    Args:
        api_key: API key as string or SecretStr

    Returns:
        Raw string value of the API key
    """
    if isinstance(api_key, SecretStr):
        return api_key.get_secret_value()
    return api_key

# Type stub for middleware result
MiddlewareResult = Any  # noqa: PGH003

logger = logging.getLogger(__name__)


class JWTAuthError(Exception):
    """Base exception for JWT authentication errors."""

    pass


class TokenMissingError(JWTAuthError):
    """Raised when no token is provided in the Authorization header."""

    pass


class TokenInvalidError(JWTAuthError):
    """Raised when a token is invalid or expired."""

    pass


def create_token(
    user_id: str,
    secret_key: str,
    algorithm: str = "HS256",
    expire_minutes: int = 30,
    issuer: str | None = None,
    audience: str | None = None,
    **extra_claims: Any,
) -> str:
    """Create a JWT token for a user.

    Args:
        user_id: The user ID to encode in the token
        secret_key: Secret key for signing the token
        algorithm: JWT algorithm to use (default: HS256)
        expire_minutes: Token expiration time in minutes (default: 30)
        issuer: Optional issuer claim
        audience: Optional audience claim
        **extra_claims: Additional claims to include in the token

    Returns:
        Encoded JWT token string

    Raises:
        ValueError: If secret_key is empty or invalid

    Example:
        >>> token = create_token(
        ...     user_id="user123",
        ...     secret_key="my-secret",
        ...     expire_minutes=60,
        ...     role="admin"
        ... )
    """
    if not secret_key:
        raise ValueError("secret_key must not be empty")

    # Calculate expiration time
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=expire_minutes)

    # Build payload with standard and custom claims
    payload: dict[str, Any] = {
        "sub": user_id,  # Subject (user ID)
        "iat": now,  # Issued at
        "exp": expire,  # Expiration
        **extra_claims,
    }

    # Add optional claims
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience

    # Encode and return token
    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    logger.debug(f"Created JWT token for user {user_id} (expires in {expire_minutes}m)")
    return token


def validate_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
    issuer: str | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    """Validate and decode a JWT token.

    Args:
        token: JWT token string to validate
        secret_key: Secret key for verifying the signature
        algorithm: JWT algorithm to use (default: HS256)
        issuer: Optional issuer to verify
        audience: Optional audience to verify

    Returns:
        Decoded token payload dictionary

    Raises:
        TokenInvalidError: If the token is invalid, expired, or signature fails

    Example:
        >>> payload = validate_token(token, secret_key="my-secret")
        >>> user_id = payload["sub"]
    """
    if not secret_key:
        raise ValueError("secret_key must not be empty")

    try:
        # Build decode options
        decode_options: dict[str, Any] = {
            "verify_signature": True,
            "verify_exp": True,
        }

        # Decode and verify token
        payload: dict[str, Any] = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            issuer=issuer,
            audience=audience,
            options=decode_options,
        )

        logger.debug(f"Validated JWT token for user {payload.get('sub')}")
        return payload

    except ExpiredSignatureError as e:
        logger.warning(f"JWT token expired: {e}")
        raise TokenInvalidError("Token has expired") from e
    except DecodeError as e:
        logger.warning(f"JWT token decode error: {e}")
        raise TokenInvalidError("Invalid token format") from e
    except InvalidTokenError as e:
        logger.warning(f"JWT token validation failed: {e}")
        raise TokenInvalidError(f"Token validation failed: {e}") from e


def extract_bearer_token(authorization_header: str | None) -> str:
    """Extract Bearer token from Authorization header.

    Args:
        authorization_header: The Authorization header value

    Returns:
        The extracted token string

    Raises:
        TokenMissingError: If no Authorization header or invalid format

    Example:
        >>> token = extract_bearer_token("Bearer abc123...")
        >>> # Returns: "abc123..."
    """
    if not authorization_header:
        raise TokenMissingError("No Authorization header provided")

    # Check for Bearer scheme
    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise TokenMissingError("Invalid Authorization header format. Expected: 'Bearer <token>'")

    return parts[1]


# ============================================================================
# API Key Management
# ============================================================================

# In-memory storage for API keys (production would use a database)
_api_keys: dict[str, Any] = {}


def generate_api_key(length: int = 32) -> str:
    """Generate a new secure API key.

    Args:
        length: Length of the API key in bytes (default 32)

    Returns:
        A URL-safe base64-encoded API key string with rmcp_ prefix

    Example:
        >>> key = generate_api_key()
        >>> key.startswith("rmcp_")
        True
    """
    # Generate cryptographically secure random bytes and convert to URL-safe string
    key = secrets.token_urlsafe(length)
    # Add prefix for identification
    return f"rmcp_{key}"


def hash_api_key(api_key: str | SecretStr) -> str:
    """Hash an API key for secure storage.

    Args:
        api_key: The API key to hash (string or SecretStr)

    Returns:
        SHA-256 hash of the API key

    Example:
        >>> key = "rmcp_test123"
        >>> hashed = hash_api_key(key)
        >>> len(hashed)
        64
    """
    key_value = _get_api_key_value(api_key)
    return hashlib.sha256(key_value.encode()).hexdigest()


def register_api_key(
    api_key: str | SecretStr,
    user_id: str,
    description: str = "",
    permissions: list[str] | None = None,
    expires_at: datetime | None = None,
) -> Any:
    """Register a new API key in the system.

    Args:
        api_key: The API key to register (string or SecretStr)
        user_id: User ID associated with this key
        description: Human-readable description
        permissions: List of permissions (default ["read"])
        expires_at: Expiration time (None for no expiration)

    Returns:
        APIKeyInfo object for the registered key

    Example:
        >>> key = generate_api_key()
        >>> info = register_api_key(key, "user123", "Test key")
        >>> info.user_id
        'user123'
    """
    from reasoning_mcp.config import APIKeyInfo

    key_hash = hash_api_key(_get_api_key_value(api_key))

    if permissions is None:
        permissions = ["read"]

    info = APIKeyInfo(
        key_hash=key_hash,
        user_id=user_id,
        description=description,
        permissions=permissions,
        expires_at=expires_at,
    )

    _api_keys[key_hash] = info
    logger.info(f"Registered API key for user {user_id}: {description}")
    return info


def validate_api_key(api_key: str | SecretStr) -> Any | None:
    """Validate an API key and return its info if valid.

    Args:
        api_key: The API key to validate (string or SecretStr)

    Returns:
        APIKeyInfo if the key is valid, None otherwise

    Example:
        >>> key = generate_api_key()
        >>> register_api_key(key, "user123")
        >>> info = validate_api_key(key)
        >>> info is not None
        True
    """
    key_hash = hash_api_key(_get_api_key_value(api_key))
    info = _api_keys.get(key_hash)

    if info is None:
        logger.debug(f"API key not found: {key_hash[:16]}...")
        return None

    if not info.is_valid():
        logger.debug(f"API key invalid or expired: {key_hash[:16]}...")
        return None

    # Update last_used_at
    info.last_used_at = datetime.now()
    logger.debug(f"Validated API key for user {info.user_id}")
    return info


def revoke_api_key(api_key: str | SecretStr) -> bool:
    """Revoke an API key.

    Args:
        api_key: The API key to revoke (string or SecretStr)

    Returns:
        True if the key was revoked, False if not found

    Example:
        >>> key = generate_api_key()
        >>> register_api_key(key, "user123")
        >>> revoke_api_key(key)
        True
    """
    key_hash = hash_api_key(_get_api_key_value(api_key))
    info = _api_keys.get(key_hash)

    if info is None:
        logger.warning(f"Attempted to revoke non-existent API key: {key_hash[:16]}...")
        return False

    info.revoked = True
    info.revoked_at = datetime.now()
    logger.info(f"Revoked API key for user {info.user_id}")
    return True


def list_api_keys() -> list[Any]:
    """List all registered API keys.

    Returns:
        List of all APIKeyInfo objects

    Example:
        >>> keys = list_api_keys()
        >>> isinstance(keys, list)
        True
    """
    return list(_api_keys.values())


def clear_api_keys() -> int:
    """Clear all API keys from memory.

    Returns:
        Number of keys cleared

    Example:
        >>> count = clear_api_keys()
        >>> count >= 0
        True
    """
    count = len(_api_keys)
    _api_keys.clear()
    logger.info(f"Cleared {count} API keys from memory")
    return count


class JWTAuthMiddleware:
    """JWT authentication middleware for FastMCP server.

    This middleware intercepts tool calls and validates JWT tokens from
    the Authorization header. It injects user context into the request
    for downstream tools to use.

    Features:
    - Bearer token extraction and validation
    - User context injection into MCP Context
    - Configurable error handling
    - Request logging and metrics

    Example:
        >>> middleware = JWTAuthMiddleware(
        ...     secret_key="your-secret-key",
        ...     algorithm="HS256",
        ...     issuer="reasoning-mcp",
        ... )
        >>> middleware.register(mcp)
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: str | None = None,
        audience: str | None = None,
    ) -> None:
        """Initialize JWT authentication middleware.

        Args:
            secret_key: Secret key for JWT verification
            algorithm: JWT algorithm (default: HS256)
            issuer: Optional issuer to verify
            audience: Optional audience to verify

        Raises:
            ValueError: If secret_key is empty
        """
        if not secret_key:
            raise ValueError("secret_key must not be empty")

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience

    def register(self, mcp: FastMCP) -> None:
        """Register JWT authentication middleware with FastMCP server.

        This adds a tool call interceptor that validates JWT tokens
        before allowing requests to proceed.

        Args:
            mcp: The FastMCP server instance

        Example:
            >>> middleware = JWTAuthMiddleware(secret_key="secret")
            >>> middleware.register(mcp)
        """

        @mcp.on_tool_call  # type: ignore[attr-defined,untyped-decorator]
        async def jwt_auth_middleware(
            request: CallToolRequest,
            call_next: Callable[[], Any],
        ) -> Any:
            """Intercept tool calls for JWT authentication."""
            # Extract Authorization header from request metadata
            # Note: In MCP, authentication metadata should be provided by the client
            # This is a placeholder - actual implementation depends on MCP transport
            authorization_header = None

            # Try to get from request params if available
            if hasattr(request, "params") and hasattr(request.params, "_meta"):
                authorization_header = getattr(request.params._meta, "authorization", None)

            # If no header, check environment or session
            # This is a fallback mechanism
            if not authorization_header:
                logger.warning(
                    "No Authorization header found in request. "
                    "JWT authentication requires clients to provide Authorization metadata."
                )
                raise TokenMissingError("Authorization required. Please provide a Bearer token.")

            try:
                # Extract and validate token
                token = extract_bearer_token(authorization_header)
                payload = validate_token(
                    token,
                    self.secret_key,
                    self.algorithm,
                    self.issuer,
                    self.audience,
                )

                # Inject user context into request
                # Store in a way that downstream tools can access
                user_id = payload.get("sub")
                logger.info(f"Authenticated request for user: {user_id}")

                # TODO: Store user context in MCP Context for tool access
                # This requires Context state management support

                # Proceed with the request
                result = await call_next()
                return result

            except (TokenMissingError, TokenInvalidError) as e:
                logger.error(f"JWT authentication failed: {e}")
                # Raise the error to reject the request
                raise

        logger.info("JWT authentication middleware registered with FastMCP server")


def create_middleware_from_settings(
    settings: Settings,
) -> JWTAuthMiddleware | None:
    """Create JWT auth middleware from settings.

    Args:
        settings: Server settings with JWT configuration

    Returns:
        Configured JWTAuthMiddleware or None if JWT is disabled

    Raises:
        ValueError: If JWT is enabled but secret_key is missing or empty

    Example:
        >>> from reasoning_mcp.config import get_settings
        >>> settings = get_settings()
        >>> middleware = create_middleware_from_settings(settings)
        >>> if middleware:
        ...     middleware.register(mcp)
    """
    if not settings.jwt_enabled:
        logger.info("JWT authentication is disabled")
        return None

    # Extract secret value from SecretStr (handles None case)
    if settings.jwt_secret_key is None:
        raise ValueError(
            "SECURITY ERROR: JWT authentication is enabled but jwt_secret_key is not configured. "
            "Please set REASONING_MCP_JWT_SECRET_KEY environment variable to a secure secret key."
        )

    secret_value = settings.jwt_secret_key.get_secret_value()
    if not secret_value or not secret_value.strip():
        raise ValueError(
            "SECURITY ERROR: JWT authentication is enabled but jwt_secret_key is empty. "
            "Please set REASONING_MCP_JWT_SECRET_KEY environment variable to a secure secret key."
        )

    middleware = JWTAuthMiddleware(
        secret_key=secret_value,
        algorithm=settings.jwt_algorithm,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )

    logger.info(
        f"JWT authentication middleware created "
        f"(algorithm={settings.jwt_algorithm}, "
        f"issuer={settings.jwt_issuer or 'None'}, "
        f"audience={settings.jwt_audience or 'None'})"
    )

    return middleware


class APIKeyAuthError(Exception):
    """Base exception for API key authentication errors."""

    pass


class APIKeyMissingError(APIKeyAuthError):
    """Raised when no API key is provided."""

    pass


class APIKeyInvalidError(APIKeyAuthError):
    """Raised when an API key is invalid or revoked."""

    pass


class APIKeyAuthMiddleware:
    """API key authentication middleware for FastMCP server.

    This middleware intercepts tool calls and validates API keys from
    a configurable header (default: X-API-Key). It injects user context
    into the request for downstream tools to use.

    Features:
    - API key extraction and validation
    - User context injection into MCP Context
    - Configurable header name
    - Request logging and metrics

    Example:
        >>> middleware = APIKeyAuthMiddleware(header_name="X-API-Key")
        >>> middleware.register(mcp)
    """

    def __init__(
        self,
        header_name: str = "X-API-Key",
    ) -> None:
        """Initialize API key authentication middleware.

        Args:
            header_name: HTTP header name for API key (default: X-API-Key)
        """
        self.header_name = header_name

    def register(self, mcp: FastMCP) -> None:
        """Register API key authentication middleware with FastMCP server.

        This adds a tool call interceptor that validates API keys
        before allowing requests to proceed.

        Args:
            mcp: The FastMCP server instance

        Example:
            >>> middleware = APIKeyAuthMiddleware()
            >>> middleware.register(mcp)
        """

        @mcp.on_tool_call  # type: ignore[attr-defined,untyped-decorator]
        async def api_key_auth_middleware(
            request: CallToolRequest,
            call_next: Callable[[], Any],
        ) -> Any:
            """Intercept tool calls for API key authentication."""
            # Extract API key header from request metadata
            # Note: In MCP, authentication metadata should be provided by the client
            api_key = None

            # Try to get from request params if available
            if hasattr(request, "params") and hasattr(request.params, "_meta"):
                api_key = getattr(
                    request.params._meta, self.header_name.lower().replace("-", "_"), None
                )

            # If no header, check environment or session
            if not api_key:
                logger.warning(
                    f"No {self.header_name} header found in request. "
                    "API key authentication requires clients to provide the API key."
                )
                raise APIKeyMissingError(
                    f"API key required. Please provide {self.header_name} header."
                )

            try:
                # Validate API key
                info = validate_api_key(api_key)

                if info is None:
                    raise APIKeyInvalidError("Invalid or expired API key")

                # Inject user context into request
                user_id = info.user_id
                permissions = info.permissions
                logger.info(
                    f"Authenticated request for user: {user_id} (permissions: {permissions})"
                )

                # TODO: Store user context in MCP Context for tool access
                # This requires Context state management support

                # Proceed with the request
                result = await call_next()
                return result

            except APIKeyInvalidError as e:
                logger.error(f"API key authentication failed: {e}")
                # Raise the error to reject the request
                raise

        logger.info(
            f"API key authentication middleware registered with FastMCP server "
            f"(header: {self.header_name})"
        )


def create_api_key_middleware_from_settings(
    settings: Settings,
) -> APIKeyAuthMiddleware | None:
    """Create API key auth middleware from settings.

    Args:
        settings: Server settings with API key configuration

    Returns:
        Configured APIKeyAuthMiddleware or None if API key auth is disabled

    Example:
        >>> from reasoning_mcp.config import get_settings
        >>> settings = get_settings()
        >>> middleware = create_api_key_middleware_from_settings(settings)
        >>> if middleware:
        ...     middleware.register(mcp)
    """
    if not settings.api_key_enabled:
        logger.info("API key authentication is disabled")
        return None

    middleware = APIKeyAuthMiddleware(
        header_name=settings.api_key_header,
    )

    logger.info(f"API key authentication middleware created (header={settings.api_key_header})")

    return middleware


__all__ = [
    "JWTAuthError",
    "TokenMissingError",
    "TokenInvalidError",
    "create_token",
    "validate_token",
    "extract_bearer_token",
    "JWTAuthMiddleware",
    "create_middleware_from_settings",
    "APIKeyAuthError",
    "APIKeyMissingError",
    "APIKeyInvalidError",
    "generate_api_key",
    "hash_api_key",
    "register_api_key",
    "validate_api_key",
    "revoke_api_key",
    "list_api_keys",
    "clear_api_keys",
    "APIKeyAuthMiddleware",
    "create_api_key_middleware_from_settings",
]
