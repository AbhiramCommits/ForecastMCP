"""Role-based tool permissions. Denials raise and are logged as spans."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ALL_TOOLS = {
    "list_series",
    "describe_series",
    "forecast",
    "ensemble_forecast",
    "backtest",
    "model_card",
    "sql",
    "finetune",
}

ROLE_ALLOWLIST: dict[str, set[str]] = {
    "analyst": {
        "list_series",
        "describe_series",
        "forecast",
        "ensemble_forecast",
        "backtest",
        "model_card",
        "sql",
    },
    "admin": set(ALL_TOOLS),
}


class PermissionDenied(RuntimeError):
    """Role is not allowed to call the requested tool."""

    def __init__(self, role: str, tool: str):
        self.role = role
        self.tool = tool
        super().__init__(f"role {role!r} is not permitted to call tool {tool!r}")


def check_permission(role: str, tool: str) -> None:
    """Raise ``PermissionDenied`` unless ``role`` may call ``tool``."""
    allowed = ROLE_ALLOWLIST.get(role)
    if allowed is None:
        raise PermissionDenied(role, tool)
    if tool not in allowed:
        raise PermissionDenied(role, tool)


def allowed_tools(role: str) -> list[str]:
    """Sorted tool names allowed for a role (empty for unknown roles)."""
    return sorted(ROLE_ALLOWLIST.get(role, set()))


def log_denial(event: Callable[[str, dict[str, Any]], None], role: str, tool: str) -> None:
    """Emit a non-silent denial span/event."""
    event("permission.denied", {"role": role, "tool": tool, "allowed": False})
