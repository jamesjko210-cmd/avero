from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field


DEFAULT_STATUS_HOST = "127.0.0.1"
DEFAULT_STATUS_PORT = 8766
STATUS_AUTH_ENV = "JARVIS_STATUS_AUTH_TOKEN"
STATUS_AUTH_USERNAME = "jarvis"
MIN_STATUS_AUTH_TOKEN_LENGTH = 32
MAX_STATUS_AUTH_TOKEN_LENGTH = 256


@dataclass(frozen=True)
class StatusHostConfig:
    host: str
    configured: bool
    valid: bool
    source: str


@dataclass(frozen=True)
class StatusPortConfig:
    port: int
    configured: bool
    valid: bool
    source: str


@dataclass(frozen=True)
class StatusAuthConfig:
    token: str = field(repr=False, compare=False)
    configured: bool
    valid: bool
    source: str


def status_host_from_env() -> str:
    return status_host_config_from_env().host


def status_host_config_from_env() -> StatusHostConfig:
    from jarvis_v2.env import load_env

    load_env()
    raw = (os.getenv("JARVIS_STATUS_HOST") or "").strip()
    if not raw:
        return StatusHostConfig(DEFAULT_STATUS_HOST, configured=False, valid=True, source="default")
    if (
        any(char.isspace() for char in raw)
        or any(ord(char) < 32 for char in raw)
        or not status_host_is_loopback(raw)
    ):
        return StatusHostConfig(DEFAULT_STATUS_HOST, configured=True, valid=False, source="env-invalid")
    return StatusHostConfig(raw, configured=True, valid=True, source="env")


def status_host_is_loopback(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def status_port_from_env() -> int:
    return status_port_config_from_env().port


def status_port_config_from_env() -> StatusPortConfig:
    from jarvis_v2.env import load_env

    load_env()
    raw = (os.getenv("JARVIS_STATUS_PORT") or "").strip()
    if not raw:
        return StatusPortConfig(DEFAULT_STATUS_PORT, configured=False, valid=True, source="default")
    try:
        port = int(raw)
    except ValueError:
        return StatusPortConfig(DEFAULT_STATUS_PORT, configured=True, valid=False, source="env-invalid")
    if not 1 <= port <= 65535:
        return StatusPortConfig(DEFAULT_STATUS_PORT, configured=True, valid=False, source="env-invalid")
    return StatusPortConfig(port, configured=True, valid=True, source="env")


def status_auth_config_from_env() -> StatusAuthConfig:
    from jarvis_v2.env import load_env

    load_env()
    raw = os.getenv(STATUS_AUTH_ENV)
    if raw is None or raw == "":
        return StatusAuthConfig("", configured=False, valid=False, source="env-missing")
    if not status_auth_token_is_valid(raw):
        return StatusAuthConfig("", configured=True, valid=False, source="env-invalid")
    return StatusAuthConfig(raw, configured=True, valid=True, source="env")


def status_auth_token_is_valid(token: str) -> bool:
    return (
        MIN_STATUS_AUTH_TOKEN_LENGTH <= len(token) <= MAX_STATUS_AUTH_TOKEN_LENGTH
        and all(33 <= ord(char) <= 126 for char in token)
    )


def status_base_url(host: str | None = None, port: int | None = None) -> str:
    resolved_host = host if host is not None else status_host_from_env()
    resolved_port = port if port is not None else status_port_from_env()
    return f"http://{_url_host(resolved_host)}:{resolved_port}"


def _url_host(host: str) -> str:
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]"
    return host
