class BuildError:
    def __init__(self, error_type, raw_message, details=None):
        self.error_type = error_type
        self.raw_message = raw_message
        self.details = details or {}

    def __repr__(self):
        return f"BuildError(type={self.error_type})"

def parse_build_log(log_text: str) -> BuildError | None:
    if "insufficient workspace" in log_text:
        return BuildError(
            error_type="workspace_too_small",
            raw_message=log_text,
        )
    if "no optimization profile has been defined" in log_text:
        return BuildError(
            error_type="missing_optimization_profile",
            raw_message=log_text,
        )
    return None
