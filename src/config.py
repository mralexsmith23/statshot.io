"""Project paths and settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHOT_CACHE_DIR = DATA_DIR / "shot_cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------
# Current-season parquet files older than this are considered stale and
# re-fetched from the API.  Completed seasons never expire.
CACHE_MAX_AGE_HOURS: int = 24

# Only these columns are persisted to parquet — everything else from the API
# response is dropped to save ~40-60 % disk space.
SHOT_KEEP_COLUMNS: list[str] = [
    "PLAYER_ID", "PLAYER_NAME",
    "TEAM_ID", "TEAM_NAME",
    "GAME_DATE",
    "LOC_X", "LOC_Y",
    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE", "SHOT_TYPE", "ACTION_TYPE",
    "SHOT_MADE_FLAG", "SHOT_ATTEMPTED_FLAG",
]

# A parquet file must contain at least these columns to be considered valid.
SHOT_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "LOC_X", "LOC_Y", "SHOT_MADE_FLAG",
})

# ---------------------------------------------------------------------------
# Contact form (Formspree — hides your email from the public)
# ---------------------------------------------------------------------------
# Sign up free at https://formspree.io, create a form pointed at your email,
# and paste the form ID here (the "f/xxxxx" part of your endpoint URL).
FORMSPREE_FORM_ID: str = "xdkojkwg"

# ---------------------------------------------------------------------------
# Google Analytics
# ---------------------------------------------------------------------------
# Set to your GA4 Measurement ID (e.g. "G-XXXXXXXXXX") to enable tracking.
# Leave empty to disable.
GA_TRACKING_ID: str = "G-M56R0CSJPQ"
