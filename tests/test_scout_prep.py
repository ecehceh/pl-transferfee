import numpy as np
import pandas as pd

from scout_prep import add_trajectory, normalize_name, player_side, position_group


# --- position groups ---------------------------------------------------------
def test_position_group_splits_centre_backs_and_full_backs():
    assert position_group("DF", "Centre-Back") == "CB"
    assert position_group("DF", "Right-Back") == "FB"
    assert position_group("DF", "Left-Back") == "FB"
    assert position_group("DF", None) == "CB"  # unknown defender -> CB


def test_position_group_keeps_other_groups():
    assert position_group("GK", "Goalkeeper") == "GK"
    assert position_group("MF", "Defensive Midfield") == "DM"
    assert position_group("MF", "Attacking Midfield") == "AM"
    assert position_group("MF", "Central Midfield") == "MF"
    assert position_group("FW", "Right Winger") == "W"
    assert position_group("FW", "Centre-Forward") == "FW"


def test_player_side_reads_left_right():
    assert player_side("Left-Back") == "Left"
    assert player_side("Right Winger") == "Right"
    assert player_side("Centre-Back") == "Any"
    assert player_side(None) == "Any"


# --- name normalisation (used for the value join) ----------------------------
def test_normalize_name_strips_accents_and_punctuation():
    assert normalize_name("Mëssi, Leo-nel") == "messi leo nel"
    assert normalize_name(None) == ""


# --- trajectory --------------------------------------------------------------
def test_add_trajectory_buckets_rising_and_fading():
    # a clearly improving player and a clearly declining one across two seasons
    rows = []
    # filler players so the within-season z-scores have spread
    rng = np.random.default_rng(0)
    for s in (2425, 2526):
        for i in range(20):
            rows.append({"name_key": f"filler{i}", "season": s,
                         "np_xg_p90": rng.random(), "xa_p90": rng.random(),
                         "ss_rating": 6 + rng.random(), "dribbles_p90": rng.random(),
                         "tackles_won_p90": rng.random()})
    # riser: low last season, top this season
    rows += [
        {"name_key": "riser", "season": 2425, "np_xg_p90": 0.0, "xa_p90": 0.0,
         "ss_rating": 6.0, "dribbles_p90": 0.0, "tackles_won_p90": 0.0},
        {"name_key": "riser", "season": 2526, "np_xg_p90": 2.0, "xa_p90": 2.0,
         "ss_rating": 9.0, "dribbles_p90": 2.0, "tackles_won_p90": 2.0},
    ]
    # faller: top last season, low this season
    rows += [
        {"name_key": "faller", "season": 2425, "np_xg_p90": 2.0, "xa_p90": 2.0,
         "ss_rating": 9.0, "dribbles_p90": 2.0, "tackles_won_p90": 2.0},
        {"name_key": "faller", "season": 2526, "np_xg_p90": 0.0, "xa_p90": 0.0,
         "ss_rating": 6.0, "dribbles_p90": 0.0, "tackles_won_p90": 0.0},
    ]
    df = pd.DataFrame(rows)
    out = add_trajectory(df)
    traj = out.drop_duplicates("name_key").set_index("name_key")["trajectory"]
    assert traj["riser"] == "Rising"
    assert traj["faller"] == "Fading"


def test_add_trajectory_marks_single_season_players():
    df = pd.DataFrame({
        "name_key": ["solo", "solo_b"],
        "season": [2526, 2526],
        "np_xg_p90": [0.5, 0.3], "xa_p90": [0.2, 0.1], "ss_rating": [7.0, 6.5],
        "dribbles_p90": [1.0, 0.5], "tackles_won_p90": [1.0, 2.0],
    })
    out = add_trajectory(df)
    assert (out["trajectory"] == "New / one season").all()
