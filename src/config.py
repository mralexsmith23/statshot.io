"""Project paths and settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHOT_CACHE_DIR = DATA_DIR / "shot_cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FG_COMPARISON_DIR = OUTPUTS_DIR / "FGComparison"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FG_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
