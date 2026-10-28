"""Shared settings for the PL transfer fee project."""

from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Season start years to include: 2020 = 2020/21 ... 2025/26
SEASONS = list(range(2020, 2026))

PREMIER_LEAGUE_ID = "GB1"

# Transfermarkt club ids for the traditional "big six"
BIG_SIX_CLUB_IDS = {
    11,   # Arsenal
    31,   # Liverpool
    148,  # Tottenham Hotspur
    281,  # Manchester City
    631,  # Chelsea
    985,  # Manchester United
}

TOP5_LEAGUE_IDS = {"ES1", "L1", "IT1", "FR1"}  # excluding the PL itself

# Approximate Premier League deadline days, keyed by season start year.
# Used only for a rough "days before deadline" feature. Verify against
# official dates before relying on this in a write-up.
SUMMER_DEADLINES = {
    2020: date(2020, 10, 5),
    2021: date(2021, 8, 31),
    2022: date(2022, 9, 1),
    2023: date(2023, 9, 1),
    2024: date(2024, 8, 30),
    2025: date(2025, 9, 1),
}
WINTER_DEADLINES = {
    2020: date(2021, 2, 1),
    2021: date(2022, 1, 31),
    2022: date(2023, 1, 31),
    2023: date(2024, 2, 1),
    2024: date(2025, 2, 3),
    2025: date(2026, 2, 2),
}


def seller_tier(league_id) -> str:
    """Bucket the selling club's league into a readable tier."""
    if league_id == PREMIER_LEAGUE_ID:
        return "Premier League"
    if league_id in TOP5_LEAGUE_IDS:
        return "Other top-5 league"
    if isinstance(league_id, str) and league_id:
        return "Other covered league"
    # Clubs outside the leagues the dataset scrapes (EFL Championship,
    # smaller European leagues, most of South America/Asia/Africa...)
    return "Outside covered leagues"
