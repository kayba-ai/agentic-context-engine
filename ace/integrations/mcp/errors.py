class ACEMCPError(Exception):
    """Base exception for ACE MCP extensions."""

    def __init__(self, message: str, code: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ValidationError(ACEMCPError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, "ACE_MCP_VALIDATION_ERROR", details)


class SessionNotFoundError(ACEMCPError):
    def __init__(self, session_id: str):
        super().__init__(
            f"Session not found: {session_id}",
            "ACE_MCP_SESSION_NOT_FOUND",
            {"session_id": session_id},
        )


class ForbiddenInSafeModeError(ACEMCPError):
    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool {tool_name} is forbidden in safe mode",
            "ACE_MCP_FORBIDDEN_IN_SAFE_MODE",
            {"tool_name": tool_name},
        )


class SaveLoadDisabledError(ACEMCPError):
    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool {tool_name} is disabled (allow_save_load=false)",
            "ACE_MCP_SAVE_LOAD_DISABLED",
            {"tool_name": tool_name},
        )


class ProviderError(ACEMCPError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, "ACE_MCP_PROVIDER_ERROR", details)


class TimeoutError(ACEMCPError):
    def __init__(self, message: str = "Operation timed out"):
        super().__init__(message, "ACE_MCP_TIMEOUT")


class InternalError(ACEMCPError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, "ACE_MCP_INTERNAL_ERROR", details)


def map_error_to_mcp(err: Exception) -> dict:
    if isinstance(err, ACEMCPError):
        return {"code": err.code, "message": err.message, "details": err.details}
    return {
        "code": "ACE_MCP_INTERNAL_ERROR",
        "message": f"An unexpected error occurred: {str(err)}",
        "details": {"type": type(err).__name__},
    }
