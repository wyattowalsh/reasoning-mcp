"""Unit tests for JWT authentication.

Tests the JWT Bearer token authentication middleware and helper functions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from reasoning_mcp.auth import (
    APIKeyAuthMiddleware,
    APIKeyInvalidError,
    APIKeyMissingError,
    JWTAuthError,
    JWTAuthMiddleware,
    TokenInvalidError,
    TokenMissingError,
    clear_api_keys,
    create_api_key_middleware_from_settings,
    create_middleware_from_settings,
    create_token,
    extract_bearer_token,
    generate_api_key,
    hash_api_key,
    list_api_keys,
    register_api_key,
    revoke_api_key,
    validate_api_key,
    validate_token,
)
from reasoning_mcp.config import Settings


class TestCreateToken:
    """Test JWT token creation."""

    def test_create_token_basic(self) -> None:
        """Test basic token creation."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
        )
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_token_with_extra_claims(self) -> None:
        """Test token creation with extra claims."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
            role="admin",
            permissions=["read", "write"],
        )
        assert isinstance(token, str)

        # Validate and check claims
        payload = validate_token(token, secret_key="test-secret")
        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"
        assert payload["permissions"] == ["read", "write"]

    def test_create_token_with_issuer_audience(self) -> None:
        """Test token creation with issuer and audience."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
            issuer="reasoning-mcp",
            audience="mcp-client",
        )

        payload = validate_token(
            token,
            secret_key="test-secret",
            issuer="reasoning-mcp",
            audience="mcp-client",
        )
        assert payload["sub"] == "user123"
        assert payload["iss"] == "reasoning-mcp"
        assert payload["aud"] == "mcp-client"

    def test_create_token_empty_secret_raises_error(self) -> None:
        """Test that empty secret key raises ValueError."""
        with pytest.raises(ValueError, match="secret_key must not be empty"):
            create_token(
                user_id="user123",
                secret_key="",
                expire_minutes=30,
            )

    def test_create_token_different_algorithms(self) -> None:
        """Test token creation with different algorithms."""
        algorithms = ["HS256", "HS384", "HS512"]
        for algorithm in algorithms:
            token = create_token(
                user_id="user123",
                secret_key="test-secret",
                algorithm=algorithm,
                expire_minutes=30,
            )
            payload = validate_token(token, secret_key="test-secret", algorithm=algorithm)
            assert payload["sub"] == "user123"


class TestValidateToken:
    """Test JWT token validation."""

    def test_validate_token_success(self) -> None:
        """Test successful token validation."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
        )
        payload = validate_token(token, secret_key="test-secret")
        assert payload["sub"] == "user123"
        assert "exp" in payload
        assert "iat" in payload

    def test_validate_token_wrong_secret_raises_error(self) -> None:
        """Test that wrong secret key raises TokenInvalidError."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
        )
        with pytest.raises(TokenInvalidError):
            validate_token(token, secret_key="wrong-secret")

    def test_validate_token_expired_raises_error(self) -> None:
        """Test that expired token raises TokenInvalidError."""
        # Create token that expires immediately
        import jwt

        now = datetime.now(UTC)
        expired = now - timedelta(minutes=1)
        payload = {
            "sub": "user123",
            "iat": now,
            "exp": expired,
        }
        token = jwt.encode(payload, "test-secret", algorithm="HS256")

        with pytest.raises(TokenInvalidError, match="Token has expired"):
            validate_token(token, secret_key="test-secret")

    def test_validate_token_invalid_format_raises_error(self) -> None:
        """Test that invalid token format raises TokenInvalidError."""
        with pytest.raises(TokenInvalidError, match="Invalid token format"):
            validate_token("not-a-valid-token", secret_key="test-secret")

    def test_validate_token_empty_secret_raises_error(self) -> None:
        """Test that empty secret key raises ValueError."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
        )
        with pytest.raises(ValueError, match="secret_key must not be empty"):
            validate_token(token, secret_key="")

    def test_validate_token_wrong_algorithm_raises_error(self) -> None:
        """Test that wrong algorithm raises TokenInvalidError."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            algorithm="HS256",
            expire_minutes=30,
        )
        with pytest.raises(TokenInvalidError):
            validate_token(token, secret_key="test-secret", algorithm="HS512")

    def test_validate_token_issuer_mismatch_raises_error(self) -> None:
        """Test that issuer mismatch raises TokenInvalidError."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
            issuer="reasoning-mcp",
        )
        with pytest.raises(TokenInvalidError):
            validate_token(
                token,
                secret_key="test-secret",
                issuer="wrong-issuer",
            )

    def test_validate_token_audience_mismatch_raises_error(self) -> None:
        """Test that audience mismatch raises TokenInvalidError."""
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
            audience="mcp-client",
        )
        with pytest.raises(TokenInvalidError):
            validate_token(
                token,
                secret_key="test-secret",
                audience="wrong-audience",
            )


class TestExtractBearerToken:
    """Test Bearer token extraction."""

    def test_extract_bearer_token_success(self) -> None:
        """Test successful Bearer token extraction."""
        token = extract_bearer_token("Bearer abc123xyz")
        assert token == "abc123xyz"

    def test_extract_bearer_token_case_insensitive(self) -> None:
        """Test that Bearer scheme is case insensitive."""
        token = extract_bearer_token("bearer abc123xyz")
        assert token == "abc123xyz"

        token = extract_bearer_token("BEARER abc123xyz")
        assert token == "abc123xyz"

    def test_extract_bearer_token_none_raises_error(self) -> None:
        """Test that None header raises TokenMissingError."""
        with pytest.raises(TokenMissingError, match="No Authorization header provided"):
            extract_bearer_token(None)

    def test_extract_bearer_token_empty_raises_error(self) -> None:
        """Test that empty header raises TokenMissingError."""
        with pytest.raises(TokenMissingError, match="No Authorization header provided"):
            extract_bearer_token("")

    def test_extract_bearer_token_wrong_scheme_raises_error(self) -> None:
        """Test that wrong scheme raises TokenMissingError."""
        with pytest.raises(TokenMissingError, match="Invalid Authorization header format"):
            extract_bearer_token("Basic abc123xyz")

    def test_extract_bearer_token_no_token_raises_error(self) -> None:
        """Test that missing token raises TokenMissingError."""
        with pytest.raises(TokenMissingError, match="Invalid Authorization header format"):
            extract_bearer_token("Bearer")

    def test_extract_bearer_token_multiple_parts_raises_error(self) -> None:
        """Test that multiple parts raises TokenMissingError."""
        with pytest.raises(TokenMissingError, match="Invalid Authorization header format"):
            extract_bearer_token("Bearer abc123 extra")


class TestJWTAuthMiddleware:
    """Test JWT authentication middleware."""

    def test_middleware_initialization_success(self) -> None:
        """Test successful middleware initialization."""
        middleware = JWTAuthMiddleware(
            secret_key="test-secret",
            algorithm="HS256",
        )
        assert middleware.secret_key == "test-secret"
        assert middleware.algorithm == "HS256"
        assert middleware.issuer is None
        assert middleware.audience is None

    def test_middleware_initialization_with_issuer_audience(self) -> None:
        """Test middleware initialization with issuer and audience."""
        middleware = JWTAuthMiddleware(
            secret_key="test-secret",
            algorithm="HS256",
            issuer="reasoning-mcp",
            audience="mcp-client",
        )
        assert middleware.issuer == "reasoning-mcp"
        assert middleware.audience == "mcp-client"

    def test_middleware_initialization_empty_secret_raises_error(self) -> None:
        """Test that empty secret key raises ValueError."""
        with pytest.raises(ValueError, match="secret_key must not be empty"):
            JWTAuthMiddleware(secret_key="")


class TestCreateMiddlewareFromSettings:
    """Test middleware creation from settings."""

    def test_create_middleware_jwt_disabled(self) -> None:
        """Test that None is returned when JWT is disabled."""
        settings = Settings(jwt_enabled=False)
        middleware = create_middleware_from_settings(settings)
        assert middleware is None

    def test_create_middleware_jwt_enabled_no_secret_raises_error(self) -> None:
        """Test that error is raised when JWT enabled but no secret key."""
        # With the new model_validator, this should raise during Settings creation
        with pytest.raises(ValueError, match="SECURITY ERROR.*JWT authentication is enabled"):
            Settings(jwt_enabled=True, jwt_secret_key=None)

    def test_create_middleware_jwt_enabled_empty_secret_raises_error(self) -> None:
        """Test that error is raised when JWT enabled but secret key is empty."""
        # With the new model_validator, this should raise during Settings creation
        with pytest.raises(ValueError, match="SECURITY ERROR.*secret key is empty"):
            Settings(jwt_enabled=True, jwt_secret_key=SecretStr(""))

    def test_create_middleware_jwt_enabled_whitespace_secret_raises_error(self) -> None:
        """Test that error is raised when JWT enabled but secret key is whitespace."""
        # With the new model_validator, this should raise during Settings creation
        with pytest.raises(ValueError, match="SECURITY ERROR.*secret key is empty"):
            Settings(jwt_enabled=True, jwt_secret_key=SecretStr("   "))

    def test_create_middleware_jwt_enabled_success(self) -> None:
        """Test successful middleware creation from settings."""
        settings = Settings(
            jwt_enabled=True,
            jwt_secret_key=SecretStr("test-secret"),
            jwt_algorithm="HS256",
            jwt_issuer="reasoning-mcp",
            jwt_audience="mcp-client",
        )
        middleware = create_middleware_from_settings(settings)
        assert middleware is not None
        assert middleware.secret_key == "test-secret"
        assert middleware.algorithm == "HS256"
        assert middleware.issuer == "reasoning-mcp"
        assert middleware.audience == "mcp-client"

    def test_create_middleware_different_algorithms(self) -> None:
        """Test middleware creation with different algorithms."""
        algorithms = ["HS256", "HS384", "HS512"]
        for algorithm in algorithms:
            settings = Settings(
                jwt_enabled=True,
                jwt_secret_key=SecretStr("test-secret"),
                jwt_algorithm=algorithm,
            )
            middleware = create_middleware_from_settings(settings)
            assert middleware is not None
            assert middleware.algorithm == algorithm

    def test_jwt_secret_is_secret_str_type(self) -> None:
        """Test that jwt_secret_key uses SecretStr for secure handling."""
        settings = Settings(
            jwt_enabled=True,
            jwt_secret_key=SecretStr("my-super-secret"),
        )
        # Verify it's a SecretStr type
        assert isinstance(settings.jwt_secret_key, SecretStr)
        # Verify the value is accessible via get_secret_value()
        assert settings.jwt_secret_key.get_secret_value() == "my-super-secret"
        # Verify str() doesn't expose the secret
        assert "my-super-secret" not in str(settings.jwt_secret_key)

    def test_jwt_disabled_allows_none_secret(self) -> None:
        """Test that jwt_secret_key can be None when JWT is disabled."""
        settings = Settings(jwt_enabled=False, jwt_secret_key=None)
        assert settings.jwt_secret_key is None
        # Should not raise any errors
        middleware = create_middleware_from_settings(settings)
        assert middleware is None


class TestJWTIntegration:
    """Integration tests for JWT authentication flow."""

    def test_full_jwt_flow(self) -> None:
        """Test complete JWT flow: create, extract, validate."""
        # Create token
        token = create_token(
            user_id="user123",
            secret_key="test-secret",
            expire_minutes=30,
            role="admin",
        )

        # Extract from Authorization header
        auth_header = f"Bearer {token}"
        extracted_token = extract_bearer_token(auth_header)
        assert extracted_token == token

        # Validate token
        payload = validate_token(extracted_token, secret_key="test-secret")
        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"

    def test_jwt_flow_with_middleware_settings(self) -> None:
        """Test JWT flow using middleware from settings."""
        settings = Settings(
            jwt_enabled=True,
            jwt_secret_key=SecretStr("test-secret"),
            jwt_algorithm="HS256",
            jwt_expire_minutes=60,
            jwt_issuer="reasoning-mcp",
            jwt_audience="mcp-client",
        )

        # Create middleware
        middleware = create_middleware_from_settings(settings)
        assert middleware is not None

        # Extract secret value for token operations
        secret_value = settings.jwt_secret_key.get_secret_value()

        # Create token with same settings
        token = create_token(
            user_id="user123",
            secret_key=secret_value,
            algorithm=settings.jwt_algorithm,
            expire_minutes=settings.jwt_expire_minutes,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )

        # Validate token
        payload = validate_token(
            token,
            secret_key=secret_value,
            algorithm=settings.jwt_algorithm,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
        assert payload["sub"] == "user123"

    def test_jwt_error_hierarchy(self) -> None:
        """Test JWT error exception hierarchy."""
        assert issubclass(TokenMissingError, JWTAuthError)
        assert issubclass(TokenInvalidError, JWTAuthError)

        # Test that JWTAuthError can catch both
        with pytest.raises(JWTAuthError):
            raise TokenMissingError("Test error")

        with pytest.raises(JWTAuthError):
            raise TokenInvalidError("Test error")


class TestGenerateApiKey:
    """Test API key generation."""

    def test_generate_api_key_default_length(self) -> None:
        """Test API key generation with default length."""
        key = generate_api_key()
        assert isinstance(key, str)
        assert key.startswith("rmcp_")
        # URL-safe base64 produces ~4/3 bytes in characters, so 32 bytes -> ~43 chars
        assert len(key) > 40

    def test_generate_api_key_custom_length(self) -> None:
        """Test API key generation with custom length."""
        key = generate_api_key(length=16)
        assert key.startswith("rmcp_")
        # Shorter key for 16 bytes
        assert len(key) > 20

    def test_generate_api_key_unique(self) -> None:
        """Test that generated keys are unique."""
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100  # All unique

    def test_generate_api_key_format(self) -> None:
        """Test API key format is URL-safe."""
        key = generate_api_key()
        # Remove prefix and check URL-safe characters
        key_part = key[5:]  # Remove "rmcp_"
        # URL-safe base64 uses A-Z, a-z, 0-9, -, _
        import re

        assert re.match(r"^[A-Za-z0-9_-]+$", key_part)


class TestHashApiKey:
    """Test API key hashing."""

    def test_hash_api_key_string(self) -> None:
        """Test hashing a string API key."""
        key = "rmcp_test123"
        hashed = hash_api_key(key)
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA-256 produces 64 hex chars

    def test_hash_api_key_secret_str(self) -> None:
        """Test hashing a SecretStr API key."""
        key = SecretStr("rmcp_test123")
        hashed = hash_api_key(key)
        assert isinstance(hashed, str)
        assert len(hashed) == 64

    def test_hash_api_key_consistent(self) -> None:
        """Test that same key produces same hash."""
        key = "rmcp_test123"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)
        assert hash1 == hash2

    def test_hash_api_key_secret_str_same_as_string(self) -> None:
        """Test that SecretStr and string produce same hash."""
        key_str = "rmcp_test123"
        key_secret = SecretStr("rmcp_test123")
        assert hash_api_key(key_str) == hash_api_key(key_secret)

    def test_hash_api_key_different_keys(self) -> None:
        """Test that different keys produce different hashes."""
        hash1 = hash_api_key("rmcp_key1")
        hash2 = hash_api_key("rmcp_key2")
        assert hash1 != hash2


class TestRegisterApiKey:
    """Test API key registration."""

    def setup_method(self) -> None:
        """Clear API keys before each test."""
        clear_api_keys()

    def teardown_method(self) -> None:
        """Clear API keys after each test."""
        clear_api_keys()

    def test_register_api_key_basic(self) -> None:
        """Test basic API key registration."""
        key = generate_api_key()
        info = register_api_key(key, user_id="user123")
        assert info.user_id == "user123"
        assert info.permissions == ["read"]  # default
        assert info.description == ""  # default
        assert info.revoked is False

    def test_register_api_key_with_description(self) -> None:
        """Test API key registration with description."""
        key = generate_api_key()
        info = register_api_key(key, user_id="user123", description="Test key")
        assert info.description == "Test key"

    def test_register_api_key_with_permissions(self) -> None:
        """Test API key registration with custom permissions."""
        key = generate_api_key()
        info = register_api_key(
            key, user_id="user123", permissions=["read", "write", "admin"]
        )
        assert info.permissions == ["read", "write", "admin"]

    def test_register_api_key_with_expiration(self) -> None:
        """Test API key registration with expiration."""
        key = generate_api_key()
        expires_at = datetime.now(UTC) + timedelta(days=30)
        info = register_api_key(key, user_id="user123", expires_at=expires_at)
        assert info.expires_at == expires_at

    def test_register_api_key_secret_str(self) -> None:
        """Test API key registration with SecretStr."""
        key = SecretStr(generate_api_key())
        info = register_api_key(key, user_id="user123")
        assert info.user_id == "user123"


class TestValidateApiKey:
    """Test API key validation."""

    def setup_method(self) -> None:
        """Clear API keys before each test."""
        clear_api_keys()

    def teardown_method(self) -> None:
        """Clear API keys after each test."""
        clear_api_keys()

    def test_validate_api_key_success(self) -> None:
        """Test successful API key validation."""
        key = generate_api_key()
        register_api_key(key, user_id="user123")
        info = validate_api_key(key)
        assert info is not None
        assert info.user_id == "user123"

    def test_validate_api_key_not_found(self) -> None:
        """Test validation of unregistered key returns None."""
        key = generate_api_key()
        info = validate_api_key(key)
        assert info is None

    def test_validate_api_key_revoked(self) -> None:
        """Test validation of revoked key returns None."""
        key = generate_api_key()
        register_api_key(key, user_id="user123")
        revoke_api_key(key)
        info = validate_api_key(key)
        assert info is None

    def test_validate_api_key_expired(self) -> None:
        """Test validation of expired key returns None."""
        key = generate_api_key()
        # Register with past expiration (naive datetime to match is_valid)
        expires_at = datetime.now() - timedelta(days=1)
        register_api_key(key, user_id="user123", expires_at=expires_at)
        info = validate_api_key(key)
        assert info is None

    def test_validate_api_key_updates_last_used(self) -> None:
        """Test that validation updates last_used_at."""
        key = generate_api_key()
        info = register_api_key(key, user_id="user123")
        assert info.last_used_at is None

        validated = validate_api_key(key)
        assert validated is not None
        assert validated.last_used_at is not None

    def test_validate_api_key_secret_str(self) -> None:
        """Test validation with SecretStr key."""
        key_str = generate_api_key()
        register_api_key(key_str, user_id="user123")
        info = validate_api_key(SecretStr(key_str))
        assert info is not None
        assert info.user_id == "user123"


class TestRevokeApiKey:
    """Test API key revocation."""

    def setup_method(self) -> None:
        """Clear API keys before each test."""
        clear_api_keys()

    def teardown_method(self) -> None:
        """Clear API keys after each test."""
        clear_api_keys()

    def test_revoke_api_key_success(self) -> None:
        """Test successful API key revocation."""
        key = generate_api_key()
        register_api_key(key, user_id="user123")
        result = revoke_api_key(key)
        assert result is True

    def test_revoke_api_key_not_found(self) -> None:
        """Test revocation of unregistered key returns False."""
        key = generate_api_key()
        result = revoke_api_key(key)
        assert result is False

    def test_revoke_api_key_sets_revoked_flag(self) -> None:
        """Test that revocation sets revoked flag and timestamp."""
        key = generate_api_key()
        info = register_api_key(key, user_id="user123")
        assert info.revoked is False
        assert info.revoked_at is None

        revoke_api_key(key)
        assert info.revoked is True
        assert info.revoked_at is not None

    def test_revoke_api_key_secret_str(self) -> None:
        """Test revocation with SecretStr key."""
        key_str = generate_api_key()
        register_api_key(key_str, user_id="user123")
        result = revoke_api_key(SecretStr(key_str))
        assert result is True


class TestListApiKeys:
    """Test listing API keys."""

    def setup_method(self) -> None:
        """Clear API keys before each test."""
        clear_api_keys()

    def teardown_method(self) -> None:
        """Clear API keys after each test."""
        clear_api_keys()

    def test_list_api_keys_empty(self) -> None:
        """Test listing when no keys registered."""
        keys = list_api_keys()
        assert keys == []

    def test_list_api_keys_single(self) -> None:
        """Test listing single registered key."""
        key = generate_api_key()
        register_api_key(key, user_id="user123")
        keys = list_api_keys()
        assert len(keys) == 1
        assert keys[0].user_id == "user123"

    def test_list_api_keys_multiple(self) -> None:
        """Test listing multiple registered keys."""
        for i in range(5):
            key = generate_api_key()
            register_api_key(key, user_id=f"user{i}")
        keys = list_api_keys()
        assert len(keys) == 5


class TestClearApiKeys:
    """Test clearing API keys."""

    def setup_method(self) -> None:
        """Clear API keys before each test."""
        clear_api_keys()

    def teardown_method(self) -> None:
        """Clear API keys after each test."""
        clear_api_keys()

    def test_clear_api_keys_empty(self) -> None:
        """Test clearing when no keys registered."""
        count = clear_api_keys()
        assert count == 0

    def test_clear_api_keys_with_keys(self) -> None:
        """Test clearing registered keys."""
        for i in range(5):
            key = generate_api_key()
            register_api_key(key, user_id=f"user{i}")
        count = clear_api_keys()
        assert count == 5
        assert list_api_keys() == []

    def test_clear_api_keys_invalidates_keys(self) -> None:
        """Test that cleared keys are no longer valid."""
        key = generate_api_key()
        register_api_key(key, user_id="user123")
        clear_api_keys()
        info = validate_api_key(key)
        assert info is None


class TestJWTAuthMiddlewareRegister:
    """Tests for JWTAuthMiddleware.register method."""

    def test_register_adds_on_tool_call_decorator(self) -> None:
        """Test register adds the on_tool_call decorator to MCP."""
        middleware = JWTAuthMiddleware(secret_key="test-secret")

        mock_mcp = MagicMock()
        mock_decorator = MagicMock(return_value=lambda f: f)
        mock_mcp.on_tool_call = mock_decorator

        middleware.register(mock_mcp)

        mock_mcp.on_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_middleware_rejects_missing_auth(self) -> None:
        """Test registered middleware rejects requests without auth header."""
        middleware = JWTAuthMiddleware(secret_key="test-secret")

        mock_mcp = MagicMock()
        captured_middleware: Callable[..., Any] | None = None

        def capture_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            nonlocal captured_middleware
            captured_middleware = func
            return func

        mock_mcp.on_tool_call = capture_decorator

        middleware.register(mock_mcp)

        assert captured_middleware is not None

        # Create a request without auth header
        mock_request = MagicMock()
        mock_request.params = MagicMock()
        mock_request.params._meta = MagicMock()
        mock_request.params._meta.authorization = None

        mock_call_next = AsyncMock(return_value={"result": "success"})

        with pytest.raises(TokenMissingError, match="Authorization required"):
            await captured_middleware(mock_request, mock_call_next)

    @pytest.mark.asyncio
    async def test_register_middleware_validates_token(self) -> None:
        """Test registered middleware validates and passes valid tokens."""
        middleware = JWTAuthMiddleware(secret_key="test-secret")

        mock_mcp = MagicMock()
        captured_middleware: Callable[..., Any] | None = None

        def capture_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            nonlocal captured_middleware
            captured_middleware = func
            return func

        mock_mcp.on_tool_call = capture_decorator

        middleware.register(mock_mcp)

        assert captured_middleware is not None

        # Create a valid token
        token = create_token(user_id="user123", secret_key="test-secret", expire_minutes=30)

        # Create a request with valid auth header
        mock_request = MagicMock()
        mock_request.params = MagicMock()
        mock_request.params._meta = MagicMock()
        mock_request.params._meta.authorization = f"Bearer {token}"

        mock_call_next = AsyncMock(return_value={"result": "success"})

        result = await captured_middleware(mock_request, mock_call_next)

        assert result == {"result": "success"}
        mock_call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_middleware_rejects_invalid_token(self) -> None:
        """Test registered middleware rejects invalid tokens."""
        middleware = JWTAuthMiddleware(secret_key="test-secret")

        mock_mcp = MagicMock()
        captured_middleware: Callable[..., Any] | None = None

        def capture_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            nonlocal captured_middleware
            captured_middleware = func
            return func

        mock_mcp.on_tool_call = capture_decorator

        middleware.register(mock_mcp)

        assert captured_middleware is not None

        # Create a request with invalid token
        mock_request = MagicMock()
        mock_request.params = MagicMock()
        mock_request.params._meta = MagicMock()
        mock_request.params._meta.authorization = "Bearer invalid-token"

        mock_call_next = AsyncMock(return_value={"result": "success"})

        with pytest.raises(TokenInvalidError):
            await captured_middleware(mock_request, mock_call_next)


class TestAPIKeyAuthMiddleware:
    """Tests for APIKeyAuthMiddleware class."""

    def test_init_default_header(self) -> None:
        """Test default header name is X-API-Key."""
        middleware = APIKeyAuthMiddleware()
        assert middleware.header_name == "X-API-Key"

    def test_init_custom_header(self) -> None:
        """Test custom header name is accepted."""
        middleware = APIKeyAuthMiddleware(header_name="X-Custom-Key")
        assert middleware.header_name == "X-Custom-Key"


class TestAPIKeyAuthMiddlewareRegister:
    """Tests for APIKeyAuthMiddleware.register method."""

    def setup_method(self) -> None:
        """Clear API keys before each test."""
        clear_api_keys()

    def teardown_method(self) -> None:
        """Clear API keys after each test."""
        clear_api_keys()

    def test_register_adds_on_tool_call_decorator(self) -> None:
        """Test register adds the on_tool_call decorator to MCP."""
        middleware = APIKeyAuthMiddleware()

        mock_mcp = MagicMock()
        mock_decorator = MagicMock(return_value=lambda f: f)
        mock_mcp.on_tool_call = mock_decorator

        middleware.register(mock_mcp)

        mock_mcp.on_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_middleware_rejects_missing_api_key(self) -> None:
        """Test registered middleware rejects requests without API key."""
        middleware = APIKeyAuthMiddleware()

        mock_mcp = MagicMock()
        captured_middleware: Callable[..., Any] | None = None

        def capture_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            nonlocal captured_middleware
            captured_middleware = func
            return func

        mock_mcp.on_tool_call = capture_decorator

        middleware.register(mock_mcp)

        assert captured_middleware is not None

        # Create a request without API key
        mock_request = MagicMock()
        mock_request.params = MagicMock()
        mock_request.params._meta = MagicMock()
        mock_request.params._meta.x_api_key = None

        mock_call_next = AsyncMock(return_value={"result": "success"})

        with pytest.raises(APIKeyMissingError, match="API key required"):
            await captured_middleware(mock_request, mock_call_next)

    @pytest.mark.asyncio
    async def test_register_middleware_validates_api_key(self) -> None:
        """Test registered middleware validates and passes valid API keys."""
        middleware = APIKeyAuthMiddleware()

        mock_mcp = MagicMock()
        captured_middleware: Callable[..., Any] | None = None

        def capture_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            nonlocal captured_middleware
            captured_middleware = func
            return func

        mock_mcp.on_tool_call = capture_decorator

        middleware.register(mock_mcp)

        assert captured_middleware is not None

        # Create and register a valid API key
        api_key = generate_api_key()
        register_api_key(api_key, user_id="user123", permissions=["read", "write"])

        # Create a request with valid API key
        mock_request = MagicMock()
        mock_request.params = MagicMock()
        mock_request.params._meta = MagicMock()
        mock_request.params._meta.x_api_key = api_key

        mock_call_next = AsyncMock(return_value={"result": "success"})

        result = await captured_middleware(mock_request, mock_call_next)

        assert result == {"result": "success"}
        mock_call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_middleware_rejects_invalid_api_key(self) -> None:
        """Test registered middleware rejects invalid API keys."""
        middleware = APIKeyAuthMiddleware()

        mock_mcp = MagicMock()
        captured_middleware: Callable[..., Any] | None = None

        def capture_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            nonlocal captured_middleware
            captured_middleware = func
            return func

        mock_mcp.on_tool_call = capture_decorator

        middleware.register(mock_mcp)

        assert captured_middleware is not None

        # Create a request with invalid API key
        mock_request = MagicMock()
        mock_request.params = MagicMock()
        mock_request.params._meta = MagicMock()
        mock_request.params._meta.x_api_key = "invalid-key"

        mock_call_next = AsyncMock(return_value={"result": "success"})

        with pytest.raises(APIKeyInvalidError):
            await captured_middleware(mock_request, mock_call_next)


class TestCreateApiKeyMiddlewareFromSettings:
    """Tests for create_api_key_middleware_from_settings function."""

    def test_create_middleware_api_key_disabled(self) -> None:
        """Test that None is returned when API key auth is disabled."""
        settings = Settings(api_key_enabled=False)
        middleware = create_api_key_middleware_from_settings(settings)
        assert middleware is None

    def test_create_middleware_api_key_enabled(self) -> None:
        """Test successful middleware creation when enabled."""
        settings = Settings(api_key_enabled=True)
        middleware = create_api_key_middleware_from_settings(settings)
        assert middleware is not None
        assert isinstance(middleware, APIKeyAuthMiddleware)

    def test_create_middleware_custom_header(self) -> None:
        """Test middleware creation with custom header."""
        settings = Settings(
            api_key_enabled=True,
            api_key_header="X-Custom-Auth",
        )
        middleware = create_api_key_middleware_from_settings(settings)
        assert middleware is not None
        assert middleware.header_name == "X-Custom-Auth"

    def test_create_middleware_default_header(self) -> None:
        """Test middleware creation uses default header."""
        settings = Settings(api_key_enabled=True)
        middleware = create_api_key_middleware_from_settings(settings)
        assert middleware is not None
        assert middleware.header_name == "X-API-Key"
