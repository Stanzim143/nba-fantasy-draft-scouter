"""Synthetic camp participants and profiles for the debutant tests (imported by name)."""
from __future__ import annotations

import pandas as pd
from model_testkit import make_league
from offseason_testkit import make_offseason

TARGET = "2018-19"
STASH_ID, UDFA_ID, GHOST_ID = 90001, 90002, 90003


def build(seed: int = 1):
    """Tables (+ offseason logs, profiles) with a stash (drafted 2015, pick 20), an undrafted camp player and a ghost
    (a camp participant with no profile row at all), all with preseason games in the target season only."""
    tables = make_league()
    logs, tg = make_offseason(tables, "signal", seed=seed)
    donor = logs[(logs["event_season"] == TARGET) & (logs["context"] == "preseason")]
    pid0 = int(donor["player_id"].iloc[0])
    extra = []
    for new_id, name in ((STASH_ID, "Stash Guy"), (UDFA_ID, "Udfa Guy"), (GHOST_ID, "Ghost Guy")):
        rows = donor[donor["player_id"] == pid0].copy()
        rows["player_id"], rows["player_name"] = new_id, name
        extra.append(rows)
    logs = pd.concat([logs, *extra], ignore_index=True)
    players = tables["players"]
    prof = pd.DataFrame({
        "player_id": players["player_id"].astype("int64"), "player_name": players["player_name"], "birthdate": players["birthdate"],
        "country": "USA", "prev_org": "Duke", "origin": "usa", "prev_org_type": "college",
        "height_in": players["height_in"], "weight_lb": players["weight_lb"], "position": players["position"],
        "draft_year": players["draft_year"], "draft_round": players["draft_round"], "draft_number": players["draft_number"],
        "from_year": players["from_year"]})
    new = pd.DataFrame([
        dict(player_id=STASH_ID, player_name="Stash Guy", birthdate=pd.Timestamp("1994-05-01"), country="Serbia", prev_org="Club X",
             origin="intl", prev_org_type="other", height_in=82.0, weight_lb=215.0, position="F", draft_year=2015, draft_round=1,
             draft_number=20, from_year=2019),
        dict(player_id=UDFA_ID, player_name="Udfa Guy", birthdate=pd.Timestamp("1996-02-01"), country="USA", prev_org="Duke",
             origin="usa", prev_org_type="college", height_in=76.0, weight_lb=190.0, position="G", draft_year=pd.NA,
             draft_round=pd.NA, draft_number=pd.NA, from_year=2019)])
    for c in ("draft_year", "draft_round", "draft_number", "from_year"):
        prof[c] = prof[c].astype("Int64")
        new[c] = new[c].astype("Int64")
    prof = pd.concat([prof, new], ignore_index=True)
    return {**tables, "offseason_logs": logs, "offseason_team_games": tg, "player_profiles": prof}
