"""
Scouting metrics and role templates for the "club fit finder".

Two things live here, both deliberately data-driven rather than hard-coded in
the app:

METRICS   the scouting vocabulary. Each metric maps a readable label to a
          column in data/scout_players.parquet (produced by scout_prep.py),
          with a direction: +1 means "more is better", -1 means "less is
          better" (e.g. errors, times dribbled past). Percentiles and fit are
          always computed on the "higher = better" orientation.

ROLES     a library of playing-style templates. A role is a position group
          plus a set of metric weights: which stats define that job, and how
          much each one matters. Weights are relative; they're normalised
          before use, so (3, 1, 1) just means the first metric counts three
          times as much as the others.

A role's `pos_group` must be one of the groups scout_prep.py assigns:
GK, DF, DM, MF, AM, W, FW. Fit is only ever computed against candidates in the
same position group, so a winger template never ranks a centre-back.
"""

from dataclasses import dataclass, field

# --- scouting vocabulary -----------------------------------------------------
# label -> (column in scout_players.parquet, direction)
# direction +1: higher is better; -1: lower is better.
METRICS = {
    # attacking output
    "Non-penalty xG p90": ("np_xg_p90", +1),
    "xA p90": ("xa_p90", +1),
    "Shots p90": ("shots_p90", +1),
    "Key passes p90": ("key_passes_p90", +1),
    "Big chances created p90": ("big_chances_created_p90", +1),
    "Goal conversion %": ("ss_goalConversionPercentage", +1),
    "Non-penalty goals minus xG": ("np_g_minus_xg", +1),
    # possession / passing
    "Pass accuracy %": ("ss_accuratePassesPercentage", +1),
    "Final-third passes p90": ("final_third_passes_p90", +1),
    "Long-ball accuracy %": ("ss_accurateLongBallsPercentage", +1),
    "Progressive carries (dribbles) p90": ("dribbles_p90", +1),
    "Dribble success %": ("ss_successfulDribblesPercentage", +1),
    "Touches p90": ("touches_p90", +1),
    "Crosses accuracy %": ("ss_accurateCrossesPercentage", +1),
    "Dispossessed p90": ("dispossessed_p90", -1),
    "Possession lost p90": ("possession_lost_p90", -1),
    # defending
    "Tackles won p90": ("tackles_won_p90", +1),
    "Interceptions p90": ("interceptions_p90", +1),
    "Clearances p90": ("clearances_p90", +1),
    "Blocks p90": ("blocks_p90", +1),
    "Ball recoveries p90": ("ball_recovery_p90", +1),
    "Possession won att. third p90": ("poss_won_att_third_p90", +1),
    "Ground duels won %": ("ss_groundDuelsWonPercentage", +1),
    "Aerial duels won %": ("ss_aerialDuelsWonPercentage", +1),
    "Errors leading to shot p90": ("errors_p90", -1),
    "Dribbled past p90": ("dribbled_past_p90", -1),
    # physical / misc
    "Top speed (km/h)": ("ss_topSpeed", +1),
    "SofaScore rating": ("ss_rating", +1),
}

# Column -> direction, for quick lookup downstream.
METRIC_DIRECTION = {col: direction for (col, direction) in METRICS.values()}
# Readable label for a column.
METRIC_LABEL = {col: label for label, (col, _d) in METRICS.items()}


@dataclass(frozen=True)
class Role:
    """A playing-style template: a position group and weighted metrics."""

    name: str
    pos_group: str
    description: str
    # column -> relative weight (normalised at use time)
    weights: dict = field(default_factory=dict)

    def metric_columns(self):
        return list(self.weights)


def _role(name, pos_group, description, weights):
    # validate columns exist in the vocabulary so typos fail loudly
    unknown = [c for c in weights if c not in METRIC_DIRECTION]
    if unknown:
        raise ValueError(f"Role {name!r} references unknown metrics: {unknown}")
    return Role(name=name, pos_group=pos_group, description=description, weights=dict(weights))


# --- role library ------------------------------------------------------------
# Position groups: GK, DF (centre-back), DM, MF, AM, W (wide), FW (centre-fwd).
ROLES = {
    "ball_playing_cb": _role(
        "Ball-playing centre-back", "DF",
        "Defends and starts play: high pass accuracy, progressive passing, aerials, few errors.",
        {
            "ss_accuratePassesPercentage": 3,
            "final_third_passes_p90": 2,
            "ss_accurateLongBallsPercentage": 2,
            "ss_aerialDuelsWonPercentage": 2,
            "interceptions_p90": 2,
            "clearances_p90": 1,
            "errors_p90": 2,
            "dribbled_past_p90": 1,
        },
    ),
    "stopper_cb": _role(
        "No-nonsense centre-back", "DF",
        "Wins the ball and clears danger: aerials, clearances, blocks, duels.",
        {
            "ss_aerialDuelsWonPercentage": 3,
            "clearances_p90": 2,
            "blocks_p90": 2,
            "ss_groundDuelsWonPercentage": 2,
            "interceptions_p90": 2,
            "tackles_won_p90": 1,
            "errors_p90": 2,
        },
    ),
    "attacking_fullback": _role(
        "Attacking full-back", "DF",
        "Overlaps and crosses while defending the flank.",
        {
            "ss_accurateCrossesPercentage": 2,
            "dribbles_p90": 2,
            "final_third_passes_p90": 2,
            "xa_p90": 2,
            "tackles_won_p90": 2,
            "interceptions_p90": 2,
            "ss_groundDuelsWonPercentage": 1,
            "dribbled_past_p90": 1,
        },
    ),
    "holding_mid": _role(
        "Holding midfielder", "DM",
        "Shields the defence and recycles possession.",
        {
            "ss_accuratePassesPercentage": 3,
            "interceptions_p90": 3,
            "tackles_won_p90": 2,
            "ball_recovery_p90": 2,
            "ss_groundDuelsWonPercentage": 2,
            "possession_lost_p90": 2,
            "final_third_passes_p90": 1,
        },
    ),
    "box_to_box_mid": _role(
        "Box-to-box midfielder", "MF",
        "Covers ground both ways: carries, duels, chips in with goals and assists.",
        {
            "dribbles_p90": 2,
            "ball_recovery_p90": 2,
            "tackles_won_p90": 2,
            "np_xg_p90": 2,
            "xa_p90": 2,
            "final_third_passes_p90": 2,
            "ss_groundDuelsWonPercentage": 1,
            "touches_p90": 1,
        },
    ),
    "deep_playmaker": _role(
        "Deep-lying playmaker", "DM",
        "Dictates tempo from deep: passing volume, accuracy, long-ball range.",
        {
            "ss_accuratePassesPercentage": 3,
            "final_third_passes_p90": 3,
            "ss_accurateLongBallsPercentage": 2,
            "touches_p90": 2,
            "xa_p90": 2,
            "possession_lost_p90": 2,
            "interceptions_p90": 1,
        },
    ),
    "advanced_playmaker": _role(
        "Advanced playmaker (No.10)", "AM",
        "Creates in the final third: key passes, xA, dribbles, final-third passing.",
        {
            "xa_p90": 3,
            "key_passes_p90": 3,
            "big_chances_created_p90": 2,
            "final_third_passes_p90": 2,
            "dribbles_p90": 2,
            "np_xg_p90": 1,
            "dispossessed_p90": 1,
        },
    ),
    "wide_creator": _role(
        "Creative winger", "W",
        "Beats players and creates: dribbles, crosses, key passes, xA.",
        {
            "dribbles_p90": 3,
            "ss_successfulDribblesPercentage": 1,
            "xa_p90": 2,
            "key_passes_p90": 2,
            "ss_accurateCrossesPercentage": 2,
            "np_xg_p90": 1,
            "dispossessed_p90": 1,
        },
    ),
    "inside_forward": _role(
        "Inside forward", "W",
        "Wide starter who scores: cuts in to shoot, high xG, direct running.",
        {
            "np_xg_p90": 3,
            "shots_p90": 2,
            "dribbles_p90": 2,
            "ss_goalConversionPercentage": 2,
            "xa_p90": 1,
            "np_g_minus_xg": 1,
        },
    ),
    "poacher": _role(
        "Poacher", "FW",
        "Lives in the box: high xG, shot volume, clinical conversion.",
        {
            "np_xg_p90": 3,
            "shots_p90": 2,
            "ss_goalConversionPercentage": 3,
            "np_g_minus_xg": 2,
            "big_chances_created_p90": 1,
        },
    ),
    "target_man": _role(
        "Target man", "FW",
        "Hold-up striker: aerials, link play, goals.",
        {
            "ss_aerialDuelsWonPercentage": 3,
            "np_xg_p90": 2,
            "ss_accuratePassesPercentage": 2,
            "xa_p90": 1,
            "touches_p90": 1,
            "dispossessed_p90": 1,
        },
    ),
    "pressing_forward": _role(
        "Pressing forward", "FW",
        "Leads the press and still scores: ball recoveries high up, duels, xG.",
        {
            "poss_won_att_third_p90": 3,
            "ball_recovery_p90": 2,
            "np_xg_p90": 2,
            "ss_groundDuelsWonPercentage": 2,
            "shots_p90": 1,
        },
    ),
}


def role_choices():
    """(key, display name) pairs for a dropdown, grouped sensibly by position."""
    order = ["GK", "DF", "DM", "MF", "AM", "W", "FW"]
    keys = sorted(ROLES, key=lambda k: (order.index(ROLES[k].pos_group), ROLES[k].name))
    return [(k, ROLES[k].name) for k in keys]
