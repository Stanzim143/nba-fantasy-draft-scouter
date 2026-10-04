"""BaselineProjector: the explainable preseason projection model (ADR 0003, docs/modeling.md).

Components are modelled separately and combined, exactly as PLANNING.md section 4 lays out:

1. minutes per game      recency-weighted, age-adjusted, shrunk toward the league mean
2. per-minute production eight counting rates + three shooting percentages, recency-weighted,
                         age-adjusted, shrunk toward a role-appropriate prior (position, minutes)
3. availability          games played as a Beta-shaped distribution bounded by the schedule
4. volatility            shrunk per-player game-level FP spread -> floor / median / ceiling
5. rookies               draft-slot prior learned from the history's own rookies

Everything that can be estimated from the history is (age curves, shrinkage strength, recency
decay, dispersion, FP quantile shape); see ``BaselineConfig`` for the search grids and the few
documented fall-backs. The league scoring is a constructor argument (default: ``config/league.yaml``)
and is applied only at the end through ``fantasy_points_frame``.

The model only ever sees a ``History`` and calls ``History.assert_no_future``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import pandas as pd

from src.contracts import PROJECTION_STATS, STAT_COLUMN_MAP, History, season_start, season_str, validate_table
from src.models.appearance import AppearanceModel, gp_mixture
from src.models.availability import AvailabilityModel
from src.models.config import BaselineConfig
from src.models.panel import PanelData, build_panel, target_season_games
from src.models.rates import (
    MINUTES, SPECS, VOLATILITY, Frame, SpecFit, fit_spec, frame_for_target, frame_from_panel_rows, predict_rate,
)
from src.models.rookies import RookiePrior, fit_rookie_prior, pick_effective, to_float
from src.models.season_interval import SeasonFppgUncertainty, simulate_total_quantiles
from src.models.volatility import VolatilityPrior, add_ratio_columns, fit_volatility_prior, floor_median_ceiling
from src.value.league import load_league
from src.value.frame import fantasy_points_frame
from src.value.positions import position_group


@dataclass
class FittedBaseline:
    """Everything estimated from one History. Inspectable in tests and notebooks."""

    config: BaselineConfig
    scoring: Mapping[str, float]
    target_season: str
    target_start: int
    season_games: int
    panel: pd.DataFrame
    panel_data: PanelData
    minutes: SpecFit
    rates: dict[str, SpecFit]
    volatility_fit: SpecFit
    vol_prior: VolatilityPrior
    availability: AvailabilityModel
    rookie_prior: RookiePrior
    players: pd.DataFrame
    quantile_shape: tuple[float, float, float]
    names: pd.Series = field(repr=False)
    injury_features: object = None  # src.features.injury.InjuryFeatures, only set by an injury-aware subclass
    roster_features: object = None  # src.features.roster.RosterFeatures, only set by a roster-aware subclass
    transactions_features: object = None  # src.features.transactions.TransactionFeatures, only set by a transactions-aware subclass
    coach_features: object = None  # src.features.coach.CoachFeatures, only set by a coach-aware subclass (ADR 0020)
    rookie_adjust: object = None  # callable(pids) -> (mpg_scale, games_scale, rate_scale) arrays, only set by src.models.debutants (origin option)
    appearance: AppearanceModel | None = None  # hurdle stage of availability (ADR 0031), only set when config.appearance_hurdle
    season_uncertainty: SeasonFppgUncertainty | None = None  # season-mean FPPG spread (ADR 0033), only set when config.season_intervals

    # ------------------------------------------------------------------ projection
    def active_player_ids(self) -> np.ndarray:
        """Players who played in one of the last ``active_seasons`` completed seasons of the history."""
        p = self.panel
        recent = p[p["s"] > p["s"].max() - self.config.active_seasons]
        return np.sort(recent["player_id"].unique())

    def rookie_player_ids(self) -> np.ndarray:
        """Players drafted in the target year who have no games in the history."""
        dy = to_float(self.players["draft_year"])
        cand = self.players.loc[dy == self.target_start, "player_id"].to_numpy("int64")
        return np.sort(np.setdiff1d(cand, self.panel["player_id"].unique()))

    def _positions(self, pids: np.ndarray) -> np.ndarray:
        if not self.config.use_position_priors:
            return np.full(len(pids), "U", dtype=object)
        pos = self.players.drop_duplicates("player_id").set_index("player_id")["position"]
        return np.array([position_group(p) for p in pos.reindex(pids).to_numpy()])

    def _core(self, frame: Frame) -> pd.DataFrame:
        """Component estimates for every row of a lagged frame (target season or a training row)."""
        mpg, _ = predict_rate(self.minutes, frame, None)
        if self.roster_features is not None:
            mpg = mpg + self.roster_features.build(frame.pids, frame.target_s)
        if self.transactions_features is not None:
            mpg = mpg + self.transactions_features.build(frame.pids, frame.target_s)
        if self.coach_features is not None:
            mpg = mpg + self.coach_features.build(frame.pids, frame.target_s)
        mpg = np.clip(mpg, 0.0, self.config.mpg_max)
        out = {"player_id": frame.pids, "proj_mpg": mpg, "age": frame.age}
        for name, fit in self.rates.items():
            out[name], n_eff = predict_rate(fit, frame, mpg)
            if name == "fga":
                out["data_weight"] = n_eff / (n_eff + fit.kappa)
        f_lags = frame.lags["f"]
        extra = (self.injury_features.build(frame.pids, frame.target_s, frame.age)
                 if self.injury_features is not None else None)
        out["mu_f"] = self.availability.predict_mean(f_lags, frame.age, mpg, extra=extra)
        out["p_appear"] = (self.appearance.predict(f_lags, frame.age, mpg, extra=extra) if self.appearance is not None
                           else np.ones(len(frame.pids)))
        z, _ = predict_rate(self.volatility_fit, frame, None)
        out["z_hat"] = z
        out["n_hist_seasons"] = np.isfinite(f_lags).sum(axis=1)
        out["is_rookie"] = np.zeros(len(frame.pids), dtype=bool)
        return pd.DataFrame(out)

    def _veterans(self) -> pd.DataFrame:
        pids = self.active_player_ids()
        frame = frame_for_target(self.panel, pids, self.target_start,
                                 self.panel_data.ages.at(pids, self.target_start), self._positions(pids), self.config)
        return self._core(frame)

    def _rookies(self) -> pd.DataFrame:
        pids = self.rookie_player_ids()
        if not len(pids):
            return pd.DataFrame()
        pl = self.players.drop_duplicates("player_id").set_index("player_id").reindex(pids)
        group = self._positions(pids)
        age = self.panel_data.ages.at(pids, self.target_start)
        pick = pick_effective(pl["draft_number"].to_numpy(), pl["draft_round"].to_numpy())
        if self.rookie_adjust is None:
            return self.prior_rows(pids, pick, group, age)
        m_r, f_r, k_r = self.rookie_adjust(pids)
        prior_f = np.clip(self.rookie_prior.predict("f", pick, group, age), 0.02, 0.985)
        return self.prior_rows(pids, pick, group, age, mpg_scale=m_r, mu_f=np.clip(prior_f * f_r, 0.02, 0.985), rate_scale=k_r)

    def prior_rows(self, pids: np.ndarray, pick: np.ndarray, group: np.ndarray, age: np.ndarray,
                   mpg_scale: float | np.ndarray = 1.0, mu_f: np.ndarray | None = None,
                   rate_scale: float | np.ndarray = 1.0) -> pd.DataFrame:
        """Component estimates for players with no NBA history from the draft-slot prior (rookies and, via
        ``src.models.debutants``, stash debutants and undrafted signees). ``mpg_scale`` multiplies the prior
        minutes and ``mu_f`` replaces the games-played fraction; with the defaults this is exactly the rookie path."""
        rp = self.rookie_prior
        mpg = np.clip(rp.predict("mpg", pick, group, age) * mpg_scale, 0.5, self.config.mpg_max)
        out = {"player_id": pids, "proj_mpg": mpg, "age": age}
        for name, spec in SPECS.items():
            lo, hi = (0.0, 1.0) if spec.kind == "pct" else (0.0, 5.0)
            v = rp.predict(name, pick, group, age)
            out[name] = np.clip(v if spec.kind == "pct" else v * rate_scale, lo, hi)
        out["data_weight"] = np.zeros(len(pids))
        out["mu_f"] = np.clip(rp.predict("f", pick, group, age), 0.02, 0.985) if mu_f is None else np.asarray(mu_f, float)
        out["p_appear"] = np.ones(len(pids))     # the appearance stage covers veterans only (rookies keep their draft-slot prior)
        out["z_hat"] = np.ones(len(pids))
        out["n_hist_seasons"] = np.zeros(len(pids), dtype=int)
        out["is_rookie"] = np.ones(len(pids), dtype=bool)
        return pd.DataFrame(out)

    def calibrate_quantiles(self, train: Frame, min_games: int = 500,
                            oof: pd.DataFrame | None = None) -> tuple[float, float, float]:
        """Quantile shape of ``(game FP - projection) / projected game sd`` over the history's own games.

        Every historical player-season with an earlier season is projected from its lags alone (the
        same way the target season is), and the pooled standardised residual quantiles at the
        configured probabilities become the multipliers behind ``fppg_p10/p50/p90``. Because the
        residual is taken around the *projection*, the floor and ceiling include projection error
        as well as game-to-game noise, so about 10% of games fall below the floor.

        ``oof`` (see ``BaselineProjector.oof_projections``) supplies the projections of the most recent seasons from models refit
        on the history before each; when given, those replace the in-sample projections (the model fit on the very rows it
        projects), which are the fall-back when no fold was usable.
        """
        if oof is not None and len(oof):
            keys = oof[["player_id", "s", "pred", "sd"]]
        else:
            rows = train.has_history()
            if not rows.any():
                return self.quantile_shape
            sub = train.subset(rows)
            d = self._core(sub)
            pred = self._fppg(d)
            sd = d["z_hat"].to_numpy(float) * self.vol_prior.sd_prior(pred)
            keys = pd.DataFrame({"player_id": sub.pids, "s": sub.target_s, "pred": pred, "sd": sd})
        g = self.panel_data.game_fp.merge(keys, on=["player_id", "s"])
        if len(g) < min_games:
            return self.quantile_shape
        u = ((g["fp"] - g["pred"]) / g["sd"]).to_numpy(float)
        q = np.quantile(u, self.config.quantiles)
        return (float(q[0]), float(q[1]), float(q[2]))

    def fit_season_uncertainty(self, train: Frame, oof: pd.DataFrame | None = None) -> SeasonFppgUncertainty:
        """Relative spread of the season-mean FPPG miss, by seasons of history (ADR 0033).

        Every historical player-season after the first is projected from its own lags (veterans via ``_core``, players
        with no history via the draft-slot prior) and compared with the realised season mean. With ``oof`` (out-of-fold
        projections of the most recent seasons, see ``BaselineProjector.oof_projections``) those replace the in-sample ones.
        """
        if oof is not None and len(oof):
            act = self.panel[["player_id", "s", "fppg", "gp"]]
            m = oof.merge(act, on=["player_id", "s"])
            return SeasonFppgUncertainty.fit(m["pred"], m["fppg"], m["gp"], m["sd"], m["n_hist"])
        later = train.target_s > self.panel["s"].min()
        vet = train.has_history() & later
        sub = train.subset(vet)
        d = self._core(sub)
        pred = self._fppg(d)
        sd = d["z_hat"].to_numpy(float) * self.vol_prior.sd_prior(pred)
        actual, gp = (self.panel[c].to_numpy(float)[vet] for c in ("fppg", "gp"))
        n_hist = d["n_hist_seasons"].to_numpy()
        new = ~train.has_history() & later
        if new.any():
            pids = train.pids[new]
            pl = self.players.drop_duplicates("player_id").set_index("player_id").reindex(pids)
            pick = pick_effective(pl["draft_number"].to_numpy(), pl["draft_round"].to_numpy())
            d0 = self.prior_rows(pids, pick, self._positions(pids), train.age[new])
            pred0 = self._fppg(d0)
            pred = np.r_[pred, pred0]
            sd = np.r_[sd, self.vol_prior.sd_prior(pred0)]
            actual = np.r_[actual, self.panel["fppg"].to_numpy(float)[new]]
            gp = np.r_[gp, self.panel["gp"].to_numpy(float)[new]]
            n_hist = np.r_[n_hist, np.zeros(int(new.sum()), dtype=int)]
        return SeasonFppgUncertainty.fit(pred, actual, gp, sd, n_hist)

    def predict(self) -> pd.DataFrame:
        parts = [self._veterans()]
        rookies = self._rookies()
        if len(rookies):
            parts.append(rookies)
        d = pd.concat(parts, ignore_index=True)
        return self._assemble(d)

    def _stat_block(self, d: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-game projected stats from minutes and rates, respecting box-score identities."""
        mpg = d["proj_mpg"].to_numpy(float)
        fga_pm = d["fga"].to_numpy(float)
        fg3a_pm = np.minimum(d["fg3a"].to_numpy(float), fga_pm)
        fta_pm = d["fta"].to_numpy(float)
        fga, fg3a, fta = fga_pm * mpg, fg3a_pm * mpg, fta_pm * mpg
        fg3m = fg3a * d["fg3_pct"].to_numpy(float)
        fgm = np.maximum(fga * d["fg_pct"].to_numpy(float), fg3m)
        ftm = fta * d["ft_pct"].to_numpy(float)
        return {
            "pts": 2 * fgm + fg3m + ftm, "fgm": fgm, "fga": fga, "fg3m": fg3m, "ftm": ftm, "fta": fta,
            "reb": d["reb"].to_numpy(float) * mpg, "ast": d["ast"].to_numpy(float) * mpg,
            "stl": d["stl"].to_numpy(float) * mpg, "blk": d["blk"].to_numpy(float) * mpg,
            "tov": d["tov"].to_numpy(float) * mpg,
        }

    def _fppg(self, d: pd.DataFrame) -> np.ndarray:
        stats = self._stat_block(d)
        frame = pd.DataFrame({f"proj_{c}": v for c, v in stats.items()})
        return fantasy_points_frame(frame, self.scoring, prefix="proj_").to_numpy()

    def _assemble(self, d: pd.DataFrame) -> pd.DataFrame:
        L = float(self.season_games)
        mpg = d["proj_mpg"].to_numpy(float)
        stats = self._stat_block(d)
        p_appear = d["p_appear"].to_numpy(float) if "p_appear" in d.columns else np.ones(len(d))
        out = pd.DataFrame({
            "season": self.target_season,
            "player_id": d["player_id"].to_numpy("int64"),
            "player_name": self.names.reindex(d["player_id"]).to_numpy(),
            "model": "baseline",
            "proj_gp": d["mu_f"].to_numpy(float) * p_appear * L,
            "proj_mpg": mpg,
        })
        for col in STAT_COLUMN_MAP.values():
            out[f"proj_{col}"] = stats[col]
        assert set(f"proj_{c}" for c in STAT_COLUMN_MAP.values()) == set(PROJECTION_STATS)
        out["proj_fppg"] = fantasy_points_frame(out, self.scoring, prefix="proj_").to_numpy()
        out["proj_total_fp"] = out["proj_fppg"] * out["proj_gp"]
        sd = d["z_hat"].to_numpy(float) * self.vol_prior.sd_prior(out["proj_fppg"].to_numpy())
        lo, mid, hi = floor_median_ceiling(out["proj_fppg"].to_numpy(), sd, self.quantile_shape)
        out["fppg_p10"], out["fppg_p50"], out["fppg_p90"] = lo, mid, hi
        if self.appearance is None:
            gp_sd, gp_lo, gp_hi = self.availability.distribution(d["mu_f"].to_numpy(float), L)
        else:
            _, gp_sd, gp_lo, gp_hi = gp_mixture(self.availability, d["mu_f"].to_numpy(float), p_appear, L)
            out["proj_p_appear"] = p_appear
        if self.season_uncertainty is not None:
            q = simulate_total_quantiles(
                d["player_id"].to_numpy("int64"), out["proj_fppg"].to_numpy(float), d["mu_f"].to_numpy(float), p_appear,
                L, self.availability, self.season_uncertainty.rel_tau(d["n_hist_seasons"].to_numpy()), sd,
                quantiles=self.config.quantiles)
            out["proj_total_fp_p10"], out["proj_total_fp_p50"], out["proj_total_fp_p90"] = q[:, 0], q[:, 1], q[:, 2]
        out["proj_fppg_sd"] = sd
        out["proj_gp_sd"] = gp_sd
        out["proj_gp_p10"], out["proj_gp_p90"] = gp_lo, gp_hi
        out["age"] = d["age"].to_numpy(float)
        out["is_rookie"] = d["is_rookie"].to_numpy(bool)
        out["n_hist_seasons"] = d["n_hist_seasons"].to_numpy("int64")
        w = d["data_weight"].to_numpy(float)
        out["confidence"] = np.where(w < 0.5, "low", np.where(w < 0.8, "medium", "high"))
        return out.sort_values("player_id", kind="mergesort").reset_index(drop=True)


class BaselineProjector:
    """``Projector`` implementation. ``scoring=None`` reads ``config/league.yaml``."""

    name = "baseline"

    def __init__(self, scoring: Mapping[str, float] | None = None, config: BaselineConfig | None = None,
                 name: str | None = None):
        self.scoring: Mapping[str, float] = dict(scoring) if scoring is not None else dict(load_league()["scoring"])
        self.config = config or BaselineConfig()
        if name is not None:
            self.name = name

    def fit(self, history: History) -> FittedBaseline:
        history.assert_no_future()
        cfg = self.config
        pd_ = build_panel(history, self.scoring)
        if not cfg.use_position_priors:
            pd_.df["pos_group"] = "U"
        vol_prior = fit_volatility_prior(
            pd_.df, pd_.game_fp, min_games=cfg.vol_min_games,
            quantile_min_games=cfg.vol_quantile_min_games, quantiles=cfg.quantiles)
        panel = add_ratio_columns(pd_.df, vol_prior, cfg.vol_min_games)
        train = frame_from_panel_rows(panel, cfg)

        minutes = fit_spec(MINUTES, panel, train, cfg)
        # Tune the rate shrinkage against the minutes the model would actually forecast (not the realised MPG).
        mpg_pred = np.full(len(panel), np.nan)
        has_hist = train.has_history()
        mpg_pred[has_hist] = np.clip(predict_rate(minutes, train.subset(has_hist), None)[0], 0.0, cfg.mpg_max)
        rates = {name: fit_spec(spec, panel, train, cfg, mpg_pred) for name, spec in SPECS.items()}
        vol_fit = fit_spec(VOLATILITY, panel, train, cfg)

        rows = train.has_history()
        sub = train.subset(rows)
        mpg_train, _ = predict_rate(minutes, sub, None)
        injury_features = self._build_injury_features(history, sub)
        roster_features = self._build_roster_features(
            history, sub, mpg_train, panel["mpg"].to_numpy(float)[rows], panel["gp"].to_numpy(float)[rows])
        transactions_features = self._build_transactions_features(
            history, sub, mpg_train, panel["mpg"].to_numpy(float)[rows], panel["gp"].to_numpy(float)[rows])
        extra_train = injury_features.build(sub.pids, sub.target_s, sub.age) if injury_features is not None else None
        availability = AvailabilityModel.fit(
            sub.lags["f"], panel["f"].to_numpy(float)[rows], sub.age, np.clip(mpg_train, 0, cfg.mpg_max),
            decay=cfg.availability_decay, C=cfg.availability_C, min_rows=cfg.availability_min_rows,
            extra=extra_train)

        appearance = self._fit_appearance(pd_, panel, minutes, injury_features) if cfg.appearance_hurdle else None

        rookie_prior = fit_rookie_prior(
            panel, history.players, {n: (s.num, s.den) for n, s in SPECS.items()},
            self._replacement_prior(panel, minutes, rates), cfg.min_rookie_rows)

        season_games = cfg.season_games or target_season_games(pd_.season_len)
        names = history.players.drop_duplicates("player_id").set_index("player_id")["player_name"]
        logs_names = history.game_logs.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
        names = pd.concat([names, logs_names[~logs_names.index.isin(names.index)]])
        fitted = FittedBaseline(
            config=cfg, scoring=self.scoring, target_season=history.target_season,
            target_start=season_start(history.target_season), season_games=season_games,
            panel=panel, panel_data=pd_, minutes=minutes, rates=rates, volatility_fit=vol_fit,
            vol_prior=vol_prior, availability=availability, rookie_prior=rookie_prior,
            players=history.players, quantile_shape=vol_prior.q, names=names, appearance=appearance,
            injury_features=injury_features, roster_features=roster_features,
            transactions_features=transactions_features,
            coach_features=self._build_coach_features(
                history, sub, mpg_train, panel["mpg"].to_numpy(float)[rows], panel["gp"].to_numpy(float)[rows]))
        oof = self.oof_projections(history, panel) if cfg.oof_calibration else None
        fitted.quantile_shape = fitted.calibrate_quantiles(train, oof=oof)
        if cfg.season_intervals:
            fitted.season_uncertainty = fitted.fit_season_uncertainty(train, oof=oof)
        return fitted

    def oof_projections(self, history: History, panel: pd.DataFrame) -> pd.DataFrame | None:
        """Expanding-window out-of-fold projections of the last ``oof_folds`` training seasons.

        For each such season ``s`` (with at least ``oof_min_seasons`` earlier ones) a copy of this projector is refit on the
        history strictly before ``s`` and projects ``s``; the result has one row per projected player with ``pred`` (FPPG),
        ``sd`` (game-level sd), ``n_hist`` and the season index ``s``. The inner refits do not calibrate (no recursion).
        ``None`` when no season qualifies.
        """
        cfg = self.config
        seasons = np.sort(panel["s"].unique())[1:]
        folds = [int(s) for s in seasons if (panel["s"] < s).sum() and len(np.unique(panel.loc[panel["s"] < s, "s"])) >= cfg.oof_min_seasons]
        folds = folds[-cfg.oof_folds:] if cfg.oof_folds > 0 else []
        if not folds:
            return None
        inner = copy.copy(self)
        inner.config = replace(cfg, oof_calibration=False, season_intervals=False)
        tables = {"game_logs": history.game_logs, "team_games": history.team_games, "players": history.players,
                  "player_season_bio": history.player_season_bio, **history.extras}
        parts = []
        for s in folds:
            out = inner.fit(History.until(tables, season_str(s))).predict()
            parts.append(pd.DataFrame({"player_id": out["player_id"].to_numpy("int64"), "s": s,
                                       "pred": out["proj_fppg"].to_numpy(float), "sd": out["proj_fppg_sd"].to_numpy(float),
                                       "n_hist": out["n_hist_seasons"].to_numpy("int64")}))
        return pd.concat(parts, ignore_index=True)

    def _fit_appearance(self, pd_: PanelData, panel: pd.DataFrame, minutes: SpecFit,
                        injury_features: object | None = None) -> AppearanceModel:
        """Hurdle stage (ADR 0031): P(>= 1 game) over every historical (eligible player, season).

        Eligible = played in one of the ``active_seasons`` seasons before the row's season, the same rule
        ``active_player_ids`` applies to the target season; the label is whether the player has a panel row
        (>= 1 game) in that season. Rows are built season by season from the panel alone (no future data).
        """
        cfg = self.config
        seasons = np.sort(panel["s"].unique())
        played = panel.groupby("s")["player_id"].agg(lambda x: set(x.tolist()))
        lags, ages, mpgs, labels, extras = [], [], [], [], []
        for s in seasons[1:]:
            prior = panel[(panel["s"] < s) & (panel["s"] >= s - cfg.active_seasons)]
            pids = np.sort(prior["player_id"].unique())
            if not len(pids):
                continue
            age = pd_.ages.at(pids, int(s))
            frame = frame_for_target(panel, pids, int(s), age, np.full(len(pids), "U"), cfg)
            mpg = np.clip(predict_rate(minutes, frame, None)[0], 0.0, cfg.mpg_max)
            lags.append(frame.lags["f"])
            if injury_features is not None:
                extras.append(injury_features.build(pids, frame.target_s, frame.age))
            ages.append(frame.age)
            mpgs.append(mpg)
            labels.append(np.isin(pids, list(played.get(s, set()))).astype(float))
        if not lags:
            return AppearanceModel.fit(np.empty((0, cfg.n_lags)), np.empty(0), np.empty(0), np.empty(0),
                                       decay=cfg.availability_decay, C=cfg.appearance_C,
                                       min_rows=cfg.appearance_min_rows)
        return AppearanceModel.fit(np.vstack(lags), np.concatenate(labels), np.concatenate(ages), np.concatenate(mpgs),
                                   decay=cfg.availability_decay, C=cfg.appearance_C, min_rows=cfg.appearance_min_rows,
                                   extra=np.vstack(extras) if extras else None)

    def _build_injury_features(self, history: History, sub) -> object | None:
        """Hook for injury-aware subclasses (see ``BaselineInjuryProjector``).

        ``sub`` is the training ``Frame`` restricted to rows with an earlier season (same rows
        the availability model is fit on). Returning ``None`` (the default) means "no extra
        availability features" -- ``AvailabilityModel`` then behaves exactly as before.
        """
        return None

    def _build_roster_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                               weight: np.ndarray) -> object | None:
        """Hook for roster-aware subclasses (see BaselineRosterProjector). Returning None (the
        default) means "no context adjustment to minutes" -- mpg passes through unchanged, identical
        to before this layer existed."""
        return None

    def _build_transactions_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                                     weight: np.ndarray) -> object | None:
        """Hook for transactions-aware subclasses (see BaselineTransactionsProjector). Returning None
        (the default) means "no context adjustment to minutes" -- mpg passes through unchanged,
        identical to before this layer existed. Independent of ``_build_roster_features``: both
        adjustments can coexist additively in ``_core`` in principle, but no registered projector
        sets both at once."""
        return None

    def _build_coach_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                              weight: np.ndarray) -> object | None:
        """Hook for coach-aware subclasses (see BaselineCoachProjector, ADR 0020). ``None`` (the default) means no coach
        adjustment: minutes pass through unchanged, identical to before this layer existed."""
        return None

    @staticmethod
    def _replacement_prior(panel: pd.DataFrame, minutes: SpecFit, rates: dict[str, SpecFit]) -> dict[str, dict[str, float]]:
        """Replacement-level rookie: the league's 20th-percentile role, average-position rates."""
        rot = panel[panel["gp"] >= 20]
        rot = rot if len(rot) else panel
        order = np.argsort(rot["mpg"].to_numpy())
        cw = np.cumsum(rot["gp"].to_numpy(float)[order]) / rot["gp"].sum()
        mpg_repl = float(rot["mpg"].to_numpy()[order][np.searchsorted(cw, 0.20)])
        groups = ("G", "F", "C", "U")
        table: dict[str, dict[str, float]] = {"mpg": {g: mpg_repl for g in groups},
                                              "f": {g: float(panel["f"].median()) for g in groups}}
        for name, fit in rates.items():
            table[name] = {g: float(fit.prior_for(np.array([g]), np.array([mpg_repl]))[0]) for g in groups}
        return table

    def project(self, history: History) -> pd.DataFrame:
        out = self.fit(history).predict()
        out["model"] = self.name
        return validate_table(out, "projections")
