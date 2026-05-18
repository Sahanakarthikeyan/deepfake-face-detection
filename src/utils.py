"""
utils.py
========
Shared utility helpers for the CSWin deepfake detection pipeline.
Lightweight — no architecture code, no heavy imports at module level.
"""

import os


# ══════════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_path(*parts, env_var=None, default=None):
    """
    Resolve a file path with optional environment-variable override.

    Priority: env_var (if set and non-empty) > os.path.join(*parts) > default

    Example:
        model_path = resolve_path("outputs", "cswin_best.keras",
                                  env_var="MODEL_PATH")
    """
    if env_var:
        override = os.environ.get(env_var, "").strip()
        if override:
            return override
    if parts:
        return os.path.join(*parts)
    return default


# ══════════════════════════════════════════════════════════════════════════════
# METRIC FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_metrics(results: dict, width: int = 20) -> str:
    """Pretty-print a metrics dict (e.g. from model.evaluate)."""
    lines = ["=" * (width + 10)]
    for k, v in results.items():
        lines.append(f"  {k:<{width}}: {v:.4f}")
    lines.append("=" * (width + 10))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def list_images(directory: str):
    """Return sorted list of image file paths under a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return paths


def safe_makedirs(*paths):
    """Create one or more directories, ignoring errors if they already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)