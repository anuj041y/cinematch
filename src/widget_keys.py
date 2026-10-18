"""Build unique Streamlit widget keys."""

from __future__ import annotations

from urllib.parse import quote


def widget_key(action: str, section: str, profile_id: int | str | None = None,
               movie_id: str | None = None) -> str:
    """Combine the widget action, section, profile, and movie identifiers."""
    parts = ["cinematch", action, section]
    if profile_id is not None:
        parts.extend(("profile", str(profile_id)))
    if movie_id is not None:
        parts.extend(("movie", str(movie_id)))
    return ":".join(quote(part, safe="") for part in parts)
