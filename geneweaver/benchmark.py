"""Small helpers for presenting measured alignment performance."""


def format_seconds(seconds: float) -> str:
    """Format a duration with enough precision for short educational runs."""
    return f"{seconds:.6f} s"
