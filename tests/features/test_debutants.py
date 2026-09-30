"""find_candidates: classes, exclusions and the leakage rules (ADR 0016)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "models"))   # the shared debutant / model test kits
from debutant_testkit import GHOST_ID, STASH_ID, TARGET, UDFA_ID, build

from src.features.debutants import STASH, UNDRAFTED, camp_ids, find_candidates
from src.features.offseason import event_player_table
from src.value.league import load_league

T = build()
EV = event_player_table(T["offseason_logs"], T["offseason_team_games"], load_league()["scoring"])
PRIOR = T["game_logs"][T["game_logs"]["season"] < TARGET]["player_id"].unique()


def cands(profiles=None, **kw):
    return find_candidates(TARGET, prior_player_ids=PRIOR, profiles=profiles if profiles is not None else T["player_profiles"],
                           source_ids=camp_ids(EV, TARGET), events=EV, **kw).set_index("player_id")


def test_classes_and_slots():
    c = cands()
    assert c.loc[STASH_ID, "klass"] == STASH and c.loc[STASH_ID, "pick"] == 20 and c.loc[STASH_ID, "years_since_draft"] == 3
    assert c.loc[STASH_ID, "intl"] == 1.0 and 24 < c.loc[STASH_ID, "age"] < 25.5
    assert c.loc[UDFA_ID, "klass"] == UNDRAFTED and c.loc[UDFA_ID, "pick"] == 61
    assert c.loc[GHOST_ID, "klass"] == UNDRAFTED and np.isnan(c.loc[GHOST_ID, "age"])       # no profile row: unknown, not dropped


def test_veterans_and_the_current_draft_class_are_never_candidates():
    c = cands()
    assert not set(c.index) & set(PRIOR)
    start = int(TARGET[:4])
    assert not (pd.to_numeric(c["draft_year"], errors="coerce") == start).any()


def test_from_year_at_or_after_the_target_is_not_used_to_include_or_exclude():
    p = T["player_profiles"].copy()
    p.loc[p["player_id"] == STASH_ID, "from_year"] = 2030          # a "future debut" marker must change nothing
    assert set(cands().index) == set(cands(p).index)


def test_a_from_year_before_the_log_window_excludes_a_prior_pro():
    p = T["player_profiles"].copy()
    p.loc[p["player_id"] == UDFA_ID, "from_year"] = 2005
    assert UDFA_ID in cands().index
    assert UDFA_ID not in cands(p, window_start=2012).index


def test_evidence_columns_come_from_the_preseason():
    c = cands()
    assert c.loc[STASH_ID, "pre_gp"] == 3 and c.loc[STASH_ID, "pre_mpg"] > 0


def test_the_ghosts_name_comes_from_the_box_score_when_the_profile_is_missing():
    names = pd.Series({GHOST_ID: "Ghost Guy"})
    assert cands(names=names).loc[GHOST_ID, "player_name"] == "Ghost Guy"
