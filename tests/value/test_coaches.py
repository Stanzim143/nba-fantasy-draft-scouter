"""src.value.coaches (the descriptive coach table), src.backtest.coach_report (the analysis) and the baseline_coach projector."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from coach_testkit import T1, T2, coach_table, logs, season_games, two_team_history  # noqa: E402

from src.backtest import coach_report as R  # noqa: E402
from src.features import coach as C  # noqa: E402
from src.value import coaches as V  # noqa: E402


def with_abbr(h):
    h.game_logs["team_abbr"] = np.where(h.game_logs["team_id"] == T1, "AAA", "BBB")
    return h


def live_table(coaches):
    live = coach_table([("2021-22", T1, [("newbie", "")]), ("2021-22", T2, [("star", "")])])
    live["start_date"] = pd.Timestamp("2026-06-01")
    return pd.concat([coaches[coaches["season"] != "2021-22"], live], ignore_index=True)


def test_coach_table_flags_new_coaches_and_first_timers():
    h, coaches = two_team_history()
    t = V.coach_table(with_abbr(h), live_table(coaches)).set_index("team")
    assert bool(t.loc["AAA", "new_to_team"]) and bool(t.loc["AAA", "first_time_head_coach"]) and t.loc["AAA", "previous_coach"] == "star"
    assert bool(t.loc["BBB", "new_to_team"]) and not t.loc["BBB", "first_time_head_coach"] and t.loc["BBB", "prior_seasons"] == 3
    assert t.loc["BBB", "shift_top5"] > 0 and t.loc["AAA", "shift_top5"] == 0.0
    assert list(t.index)[0] in ("AAA", "BBB") and t["new_to_team"].all()


def test_describe_words_the_shift_and_the_first_timer():
    h, coaches = two_team_history()
    t = V.coach_table(with_abbr(h), live_table(coaches)).set_index("team")
    assert "first-time head coach" in V.describe(t.loc["AAA"])
    assert "tighter rotation" in V.describe(t.loc["BBB"]) and "shorter rotation" in V.describe(t.loc["BBB"])
    flat = t.loc["BBB"].copy()
    flat[["shift_top5", "shift_depth10", "shift_pace"]] = 0.0
    assert "no large expected change" in V.describe(flat)


def test_affected_players_are_ranked_players_on_teams_with_a_new_coach():
    h, coaches = two_team_history()
    t = V.coach_table(with_abbr(h), live_table(coaches))
    board = pd.DataFrame({"player_id": [201, 101, 300], "rank": [5, 9, 400], "name": ["b", "a", "z"], "position": ["G", "F", "C"]})
    roster = pd.DataFrame({"player_id": [201, 101, 300], "team_id": [T2, T1, T2]})
    out = V.affected_players(t, board, roster, top=150)
    assert list(out["player_id"]) == [201, 101]


# --------------------------------------------------------------------------- the analysis

def styled_history(n_seasons=14):
    """Two teams that swap two coaches every other season: 'star' coach = 38 minute stars, 'deep' coach = deep rotation, so the style
    provably follows the coach and the persistence / transfer tests must say so."""
    spec, rows = [], []
    for k in range(n_seasons):
        s = f"{2010 + k}-{str(11 + k)[-2:]}"
        a, b = ("star", "deep") if (k // 2) % 2 == 0 else ("deep", "star")
        for team, coach in ((T1, a), (T2, b)):
            spec += season_games(s, team, 25, [38, 36, 34, 32, 30, 12, 8, 6, 4, 2] if coach == "star" else [28, 27, 26, 25, 24, 22, 20, 18, 12, 8])
            rows.append((s, team, [(coach, "")]))
    gl = logs(spec)
    bio = gl[["season", "player_id"]].drop_duplicates().assign(age_at_season_start=27.0, team_id=0)
    return gl, bio, coach_table(rows)


def test_style_follows_the_coach_in_a_world_where_it_does():
    gl, bio, coaches = styled_history()
    styled = R._styled(gl, bio, coaches)
    p = R.style_persistence(styled).set_index("style")
    assert p.loc["star", "n_new"] > 0 and p.loc["star", "n_same"] > 0
    assert p.loc["star", "r_same_coach"] > 0.9 or np.isnan(p.loc["star", "r_same_coach"])
    tr = R.coach_transfer(styled, n_boot=200).set_index("style")
    assert np.isfinite(tr.loc["depth10", "coef"]) and 0.0 <= tr.loc["depth10", "p_perm"] <= 1.0   # perfectly swapped teams are collinear: only sanity here
    prof = R.coach_profiles(styled).set_index("coach_key")
    assert prof.loc["star", "star_x"] > prof.loc["deep", "star_x"] and prof.loc["star", "seasons"] == 14


def test_render_produces_the_sections_and_survives_a_missing_archetype_table():
    gl, bio, coaches = styled_history()
    styled = R._styled(gl, bio, coaches)
    text = R.render(R.style_persistence(styled), R.coach_transfer(styled, n_boot=50), R.coach_profiles(styled), None)
    for head in ("## 1.", "## 2.", "## 4."):
        assert head in text
    assert "## 3." not in text


# --------------------------------------------------------------------------- the projector

def test_projector_is_registered_and_degrades_to_baseline_without_a_coach_table(tmp_path, monkeypatch):
    from src.models.registry import available_projectors, get_projector

    assert "baseline_coach" in available_projectors()
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    proj = get_projector("baseline_coach")
    h, _ = two_team_history()
    assert proj._build_coach_features(h, None, np.zeros(1), np.zeros(1), np.ones(1)) is None      # no team_coaches.parquet: no layer


def test_coach_layer_ignores_the_future_of_the_coach_table():
    """Perturbing every coach row after the target season must not change the target season's features or shifts."""
    h, coaches = two_team_history()
    f1 = C.CoachFeatures.build_context(h, coaches, None)
    future = coach_table([("2022-23", T1, [("zzz", "")]), ("2023-24", T2, [("zzz", "")])])
    f2 = C.CoachFeatures.build_context(h, pd.concat([coaches, future], ignore_index=True), None)
    pids = np.array([100, 105, 200, 209])
    s = np.full(4, 2021)
    assert np.array_equal(f1.features(pids, s), f2.features(pids, s))


# --------------------------------------------------------------------------- the app's data prep

def test_coach_view_summary_and_missing_data(tmp_path):
    from src.app import coach_view as CV

    h, coaches = two_team_history()
    t = V.coach_table(with_abbr(h), live_table(coaches))
    lines = CV.summary_lines(t)
    assert len(lines) == 2 and any("first-time head coach" in x for x in lines) and any("replacing" in x for x in lines)
    try:
        CV.load_table("2026-27", tmp_path)
    except CV.CoachesUnavailable as exc:
        assert "wiki_coaches" in str(exc) or "not found" in str(exc)
    else:
        raise AssertionError("expected CoachesUnavailable")
    board = pd.DataFrame({"player_id": [1], "rank": [1], "name": ["x"], "position": ["G"]})
    assert CV.affected(t, board, tmp_path).empty


def test_transfer_test_does_not_invent_an_effect_from_mean_reversion():
    """Team style that just mean-reverts (no coach effect) must not look like a coach effect: the coefficient on a coach prior that
    is independent noise should be near 0 and its permutation p-value unremarkable (the review-found flaw in the first version)."""
    rng = np.random.default_rng(1)
    n = 1000
    prev = rng.normal(0, 1, n)
    cp = rng.normal(0, 1, n)                                  # a coach's prior style, unrelated to anything
    cur = 0.4 * prev + rng.normal(0, 0.8, n)                  # the team just regresses toward the mean
    coef, lo, hi, p = R._coef_ci(cp, prev, cur, n_boot=300)
    assert abs(coef) < 0.2 and lo < 0 < hi and p > 0.05
    cur2 = 0.4 * prev + 0.5 * cp + rng.normal(0, 0.8, n)      # and a real effect is found
    coef2, lo2, _, p2 = R._coef_ci(cp, prev, cur2, n_boot=300)
    assert abs(coef2 - 0.5) < 0.15 and lo2 > 0 and p2 < 0.01
