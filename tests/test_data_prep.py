import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_prep import academy_parent_ids, days_since_last_valuation  # noqa: E402


def transfers(rows):
    return pd.DataFrame(rows, columns=["from_club_id", "from_club_name", "to_club_id", "to_club_name"])


def test_academy_side_maps_to_parent_club():
    t = transfers([
        (9265, "Man City U21", 180, "Southampton"),
        (281, "Man City", 31, "Liverpool"),
        (12124, "Aston Villa U21", 631, "Chelsea"),
        (631, "Chelsea", 405, "Aston Villa"),
    ])
    assert academy_parent_ids(t) == {9265: 281, 12124: 405}


def test_reserve_b_teams_and_unknown_parents_are_left_alone():
    t = transfers([
        (5000, "Barcelona B", 281, "Man City"),      # B team plays in its own league
        (7000, "Nowhere FC U19", 281, "Man City"),   # parent never appears
        (131, "Barcelona", 631, "Chelsea"),
    ])
    assert academy_parent_ids(t) == {}


def test_ambiguous_parent_name_picks_the_most_common_id():
    t = transfers([
        (1, "Pisa U19", 10, "AC Pisa"),
        (11, "AC Pisa", 20, "Other"),
        (11, "AC Pisa", 21, "Other2"),
        (2, "AC Pisa U19", 3, "X"),
    ])
    assert academy_parent_ids(t)[2] == 11


def test_days_since_last_valuation_uses_last_valuation_strictly_before_deal():
    deals = pd.DataFrame({
        "transfer_id": [0, 1, 2],
        "player_id": [7, 7, 8],
        "transfer_date": pd.to_datetime(["2024-07-01", "2025-01-15", "2024-08-01"]),
    })
    vals = pd.DataFrame({
        "player_id": [7, 7, 7, 8],
        "date": pd.to_datetime(["2024-03-01", "2024-07-01", "2024-12-01", "2024-09-01"]),
    })
    out = days_since_last_valuation(deals, vals).set_index("transfer_id")["days_since_valuation"]
    assert out[0] == 122   # 1 Mar -> 1 Jul; the 1 Jul valuation is not "before"
    assert out[1] == 45    # 1 Dec -> 15 Jan
    assert pd.isna(out[2])  # only valuation is after the deal
