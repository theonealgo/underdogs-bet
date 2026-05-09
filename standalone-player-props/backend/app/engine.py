import math
import time
from typing import Dict, List, Optional
import requests
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from .config import CACHE_TTL_SECONDS, DEBUG_PLAYER_VALIDATION, LEAGUE_CONFIG
from .data_sources import (
    build_validated_nba_player_pool,
    build_top_players,
    fetch_prop_lines,
    fetch_schedule_and_teams,
    implied_prob,
    normal_cdf,
    poisson_cdf,
)
from .top50_scope import filter_scoped_players, is_scoped_player


_CACHE: Dict[str, Dict] = {}
_RESULTS_CACHE: Dict[str, Dict] = {}
# Bust in-process cache when player/gamelog pipeline semantics change.
_CACHE_SCHEMA = "gmlog-align-v2"
_TEAM_STATS_CACHE: Dict[str, Dict] = {}
_MODEL_WEIGHTS = {
    "xgboost": 0.25,
    "xsharp": 0.20,
    "elo": 0.10,
    "glicko2": 0.10,
    "trueskill": 0.10,
    "grinder2": 0.10,
    "takedown": 0.05,
    "edge": 0.05,
    "sharp_consensus": 0.05,
}
_MODEL_ORDER = ["glicko2", "trueskill", "xgboost", "xsharp", "sharp_consensus"]

# Tiny projection deltas so each tier can resolve a distinct OVER / UNDER vs the posted line while staying near the ensemble.
_MODEL_GRADE_TWEAKS = {
    "glicko2": 0.0045,
    "trueskill": -0.0045,
    "xgboost": 0.0028,
    "xsharp": -0.0028,
    "sharp_consensus": 0.0,
}
_MODEL_PUBLIC_NAMES = {
    "glicko2": "Grinder2",
    "trueskill": "Takedown",
    "xgboost": "XSharp",
    "xsharp": "Edge",
    "sharp_consensus": "Sharp Consensus",
}


def _xgboost_style_projection(player: Dict, prop: Dict) -> float:
    # Simplified mean projection proxy
    base = player["projected_minutes"] * 0.5 + player["usage_score"] * 12
    if prop["prop_type"] in ("points", "assists"):
        base *= 1.05
    elif prop["prop_type"] in ("rebounds", "shots_on_goal"):
        base *= 0.9
    return base


def _xsharp_adjustment(league: str, projection: float) -> float:
    # Matchup/pace adjustment proxy
    pace_factor = {
        "NBA": 1.04,
        "WNBA": 1.02,
        "NCAAB": 0.98,
        "NCAAW": 0.97,
        "NFL": 1.01,
        "NCAAF": 1.00,
        "NHL": 0.96,
        "MLB": 0.95,
        "SOCCER": 0.93,
    }.get(league, 1.0)
    return projection * pace_factor


def _form_rating(player: Dict) -> float:
    # TrueSkill/Glicko-style simplified player form score
    return (player["usage_score"] * 0.6 + player["prop_frequency"] * 0.4) * 100.0


def _projection_to_prob(league: str, projection: float, line: float, std_dev: float) -> float:
    dist = LEAGUE_CONFIG[league]["dist"]
    if dist == "poisson":
        k = max(int(math.floor(line)), 0)
        under = poisson_cdf(k, max(projection, 0.01))
        return max(0.0, min(1.0, 1.0 - under))
    z = (line - projection) / max(std_dev, 0.01)
    under = normal_cdf(z)
    return max(0.0, min(1.0, 1.0 - under))


def _ev_percent(p_win: float, american_odds: float) -> float:
    if american_odds < 0:
        b = 100.0 / abs(american_odds)
    else:
        b = american_odds / 100.0
    return ((p_win * b) - (1.0 - p_win)) * 100.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_half_step(v: float) -> float:
    return round(round(float(v) * 2.0) / 2.0, 1)


def _team_stat_map(team_id: str) -> Dict[str, float]:
    if not team_id:
        return {}
    cached = _TEAM_STATS_CACHE.get(team_id)
    if cached:
        return cached
    out: Dict[str, float] = {}
    try:
        resp = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/statistics",
            timeout=6,
        )
        body = resp.json()
        categories = (((body.get("results") or {}).get("stats") or {}).get("categories") or [])
        for cat in categories:
            for s in cat.get("stats") or []:
                name = str(s.get("name") or "")
                try:
                    out[name] = float(s.get("value"))
                except Exception:
                    continue
    except Exception:
        out = {}
    _TEAM_STATS_CACHE[team_id] = out
    return out


def _opponent_adjustment_factor(opponent_id: str, prop_type: str) -> float:
    if not opponent_id:
        return 1.0
    stats = _team_stat_map(opponent_id)
    if not stats:
        return 1.0
    if prop_type == "rebounds":
        v = stats.get("avgDefensiveRebounds", 33.0)
        return _clamp(1.0 + ((33.0 - v) / 100.0), 0.95, 1.05)
    if prop_type == "assists":
        v = stats.get("avgSteals", 7.5)
        return _clamp(1.0 + ((7.5 - v) / 75.0), 0.95, 1.05)
    if prop_type == "threes":
        v = stats.get("avgBlocks", 5.0) + stats.get("avgSteals", 7.0)
        return _clamp(1.0 + ((12.0 - v) / 120.0), 0.95, 1.05)
    v = stats.get("avgBlocks", 5.0) + stats.get("avgSteals", 7.0)
    return _clamp(1.0 + ((12.0 - v) / 120.0), 0.95, 1.05)


def _generate_internal_prop_line(prop_type: str, projection: float) -> float:
    v = float(projection)
    if prop_type == "points":
        return float(int(round(v / 5.0) * 5))
    if prop_type in ("rebounds", "assists"):
        return round((_clamp(v, 0.5, 18.5) // 2) * 2 + 1.5, 1)
    if prop_type == "threes":
        return round(_clamp(math.floor(v) + 0.5, 0.5, 6.5), 1)
    return round(v, 1)


def _calc_stat_projection(player: Dict, prop_type: str, opponent_id: str) -> tuple[float, float]:
    """Blend last-5 / last-10 with ESPN weighted gamelog means; dampen minute-ratio swings so
    a few low-minute games (return from injury, blowouts) cannot collapse points into single digits."""
    s5 = player.get("stats_last5") or {}
    s10 = player.get("stats_last10") or {}
    sw = (player.get("stats_weighted") or {}).get(prop_type)
    last5 = float(s5.get(prop_type, 0.0) or 0.0)
    last10 = float(s10.get(prop_type, 0.0) or 0.0)
    if last5 <= 0.0 and last10 <= 0.0:
        if sw is not None and float(sw) > 0.0:
            stat_base = float(sw)
        else:
            return 0.0, 1.0
    else:
        stat_base = (last5 * 0.7) + (last10 * 0.3)
    if sw is not None and float(sw) > 0.01:
        stat_base = max(stat_base, float(sw) * 0.92)
    # Season PPG from gamelog (all parsed games) — anchors stars if recent-window stats mis-parse.
    if prop_type == "points":
        ap = float(player.get("avg_points", 0.0) or 0.0)
        if ap >= 8.0:
            stat_base = max(stat_base, ap * 0.90)
    projected_minutes = float(player.get("projected_minutes_weighted", player.get("projected_minutes", 0.0)) or 0.0)
    avg_minutes = float(player.get("avg_minutes", 0.0) or 0.0)
    if projected_minutes <= 0.0 or avg_minutes <= 0.0:
        return 0.0, 1.0
    ratio = projected_minutes / max(avg_minutes, 1.0)
    ratio = _clamp(ratio, 0.82, 1.15)
    minute_scaled = stat_base * ratio
    opp_factor = _opponent_adjustment_factor(opponent_id, prop_type)
    return _clamp(minute_scaled * opp_factor, 0.0, 70.0), opp_factor


def _model_confidence_from_projection(player: Dict, projection: float, line: float, prop_type: str) -> Dict[str, float]:
    last5 = float((player.get("stats_last5") or {}).get(prop_type, 0.0) or 0.0)
    last10 = float((player.get("stats_last10") or {}).get(prop_type, 0.0) or 0.0)
    volatility = abs(last5 - last10)
    edge = abs(float(projection) - float(line))
    base = 52.0 + min(30.0, edge * 8.0) - min(8.0, volatility * 0.8)
    usage = float(player.get("usage_rate", 0.20) or 0.20)
    base += min(4.0, usage * 8.0)
    tweaks = {
        "glicko2": -1.0,
        "trueskill": -0.2,
        "xgboost": 1.3,
        "xsharp": 0.8,
        "sharp_consensus": 0.4,
    }
    return {k: round(_clamp(base + adj, 45.0, 96.0), 1) for k, adj in tweaks.items()}


def _fallback_model_confidence(proj: float, variance: float) -> Dict[str, float]:
    base = _clamp(86.0 - variance * 0.35 + (proj / 12.0), 52.0, 94.0)
    tweaks = {
        "glicko2": 0.1,
        "trueskill": 0.6,
        "xgboost": 1.8,
        "xsharp": 1.0,
        "sharp_consensus": 0.5,
    }
    return {k: round(_clamp(base + d, 45.0, 96.0), 1) for k, d in tweaks.items()}


def _non_nba_model_confidence(player: Dict, projection: float, line: float, prop_type: str) -> Dict[str, float]:
    usage = float(player.get("usage_score", 0.3) or 0.3)
    minutes = float(player.get("projected_minutes", 24.0) or 24.0)
    edge = abs(float(projection) - float(line))
    base = 50.0 + min(18.0, edge * 11.0) + min(8.0, usage * 7.5) + min(4.0, (minutes / 42.0) * 4.0)
    # Light deterministic jitter by player/prop so all rows don't show near-identical values.
    seed = abs(hash(f"{player.get('player_id','')}-{prop_type}")) % 1000
    jitter = (seed / 1000.0) * 2.6 - 1.3
    tweaks = {
        "glicko2": -1.2,
        "trueskill": -0.4,
        "xgboost": 1.4,
        "xsharp": 0.7,
        "sharp_consensus": 0.2,
    }
    return {k: round(_clamp(base + jitter + adj, 46.0, 95.0), 1) for k, adj in tweaks.items()}


def _parse_made(value: str) -> float:
    if not value:
        return 0.0
    if "-" in value:
        try:
            return float(value.split("-")[0])
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _build_league_payload(
    league: str,
    schedule_override: Optional[List[Dict]] = None,
    slate_date_str: Optional[str] = None,
) -> Dict:
    if schedule_override is not None:
        schedule = schedule_override
    elif slate_date_str:
        try:
            d_only = datetime.strptime(str(slate_date_str).strip()[:10], "%Y-%m-%d").date()
            schedule = fetch_schedule_and_teams(league, target_date=d_only, strict_day=True)
        except ValueError:
            schedule = []
    else:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        d0 = now_et.date()
        d1 = d0 + timedelta(days=1)
        seen_ids = set()
        schedule = []
        for d in (d0, d1):
            for g in fetch_schedule_and_teams(league, target_date=d, strict_day=True):
                eid = str(g.get("event_id") or "")
                key = eid or f"{g.get('home_team_id')}|{g.get('away_team_id')}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                schedule.append(g)
        if not schedule:
            schedule = fetch_schedule_and_teams(league)
    excluded = []
    if league == "NBA":
        validated = build_validated_nba_player_pool(schedule)
        players = validated["players"]
        excluded = validated["excluded"]
    else:
        players = build_top_players(league, schedule)
    players, scoped_excluded = filter_scoped_players(league, players)
    excluded.extend(scoped_excluded)
    prop_lines = fetch_prop_lines(league, players)
    by_id = {p["player_id"]: p for p in players}
    matchups = {}
    for g in schedule:
        h = g.get("home_team_id", "")
        a = g.get("away_team_id", "")
        if h and a:
            matchups[h] = a
            matchups[a] = h

    projections = []
    debug_variance = []
    sanity_flags = []
    for prop in prop_lines:
        p = by_id.get(prop["player_id"])
        if not p:
            continue
        if not is_scoped_player(league, p.get("name", ""), p.get("team", "")):
            excluded.append(
                {
                    "player_id": p.get("player_id"),
                    "name": p.get("name"),
                    "team": p.get("team"),
                    "reasons": ["PLAYER_OUT_OF_SCOPE"],
                }
            )
            continue
        if league == "NBA":
            opponent_id = matchups.get(p.get("team_id", ""), "")
            proj, opp_factor = _calc_stat_projection(p, prop["prop_type"], opponent_id)
            if proj <= 0.0:
                excluded.append(
                    {"player_id": p.get("player_id"), "name": p.get("name"), "team_id": p.get("team_id"), "reasons": ["insufficient_data"]}
                )
                continue
            calc_line = _to_half_step(float(prop.get("line_for_calc", prop.get("line")) or 0.0))
            if calc_line <= 0.0:
                calc_line = _generate_internal_prop_line(prop["prop_type"], proj)
            model_confidence = _model_confidence_from_projection(p, proj, calc_line, prop["prop_type"])
            confidence_vals = [model_confidence.get(m, 50.0) for m in _MODEL_ORDER]
            agreement = _clamp(sum(1 for c in confidence_vals if c >= 55.0) / max(len(confidence_vals), 1), 0.0, 1.0)
            variance = sum((c - (sum(confidence_vals) / len(confidence_vals))) ** 2 for c in confidence_vals) / max(len(confidence_vals), 1)
            debug_variance.append({"player_id": p["player_id"], "variance": round(variance, 3), "opp_factor": round(opp_factor, 3)})
        else:
            calc_line = _to_half_step(float(prop.get("line_for_calc", prop.get("line", 0.0)) or 0.0))
            source_proj = prop.get("projection")
            if source_proj is not None:
                proj = float(source_proj)
            else:
                xgb_mean = _xgboost_style_projection(p, prop)
                xsharp_mean = _xsharp_adjustment(league, xgb_mean)
                rating = _form_rating(p)
                proj = (xgb_mean * 0.55) + (xsharp_mean * 0.35) + ((rating / 100.0) * 0.10 * xgb_mean)
            agreement = 0.5
            variance = abs(proj) * 0.18
            model_confidence = _non_nba_model_confidence(p, proj, calc_line, prop["prop_type"])
        calc_line = _to_half_step(float(prop.get("line_for_calc", prop.get("line", 0.0)) or 0.0)) if league != "NBA" else float(calc_line)
        std_dev = max(2.5, abs(proj) * 0.22)
        p_over = _projection_to_prob(league, proj, calc_line, std_dev)
        p_under = 1.0 - p_over
        ev_over = _ev_percent(p_over, prop["odds_over"])
        ev_under = _ev_percent(p_under, prop["odds_under"])
        confidence = min(99.0, max(50.0, (max(p_over, p_under) * 100.0 + agreement * 12.0 - variance * 0.4)))
        picked_side = "OVER" if (p_over >= p_under and agreement >= 0.5) else ("UNDER" if p_under > p_over else ("OVER" if ev_over >= ev_under else "UNDER"))

        line_source = prop.get("line_source", "")
        public_line = _to_half_step(float(prop["line"])) if (line_source == "internal_odds_api" and prop.get("line") is not None) else None
        projections.append(
            {
                "player_id": p["player_id"],
                "player_name": p["name"],
                "team": p["team"],
                "league": league,
                "prop_type": prop["prop_type"],
                "line": public_line,
                "_calc_line": calc_line,
                "odds_over": prop["odds_over"],
                "odds_under": prop["odds_under"],
                "projection": _to_half_step(proj),
                "over_probability": round(p_over * 100.0, 1),
                "under_probability": round(p_under * 100.0, 1),
                "ev_over_percent": round(ev_over, 2),
                "ev_under_percent": round(ev_under, 2),
                "confidence_score": round(confidence, 1),
                "picked_side": picked_side,
                "model_confidence": {m: model_confidence.get(m) for m in _MODEL_ORDER},
                "model_agreement": round(agreement, 3),
                "model_variance": round(variance, 3),
            }
        )
    if league == "NBA":
        players = sorted(players, key=lambda x: (float(x.get("consensus_rank", 999)), -float(x.get("top50_score", 0.0))))[:100]
        projections.sort(key=lambda x: (-(x["projection"]), -x["confidence_score"], -(x["model_agreement"])),)
    payload = {"players": players, "props": projections}
    if excluded:
        payload["excluded_players"] = excluded
    if DEBUG_PLAYER_VALIDATION and league == "NBA":
        payload["model_variance"] = debug_variance
        payload["sanity_flags"] = sanity_flags
    return payload


def get_league_data(league: str, slate_date_str: Optional[str] = None) -> Dict:
    league_u = league.upper().split(":", 1)[0] if league else ""
    sd = (slate_date_str or "").strip()[:10] if slate_date_str else ""
    cache_key = f"{league_u}:{sd or 'auto'}:{_CACHE_SCHEMA}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]
    payload = _build_league_payload(league_u, slate_date_str=sd or None)
    _CACHE[cache_key] = {"ts": now, "payload": payload}
    return payload


def _et_date_from_str(result_date_str: Optional[str]) -> Optional[datetime]:
    if not result_date_str:
        return None
    try:
        return datetime.strptime(str(result_date_str).strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _nba_stat_row_from_vals(idx: Dict[str, int], vals: List) -> Dict[str, Optional[float]]:
    def _cell(key: str) -> Optional[float]:
        if key not in idx or idx[key] >= len(vals):
            return None
        raw = vals[idx[key]]
        if key == "3PT":
            return _parse_made(raw)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return {
        "points": _cell("PTS"),
        "rebounds": _cell("REB"),
        "assists": _cell("AST"),
        "threes": _cell("3PT"),
    }


def _model_individual_grades(league_u: str, proj: float, line_f: float, actual: float) -> Dict[str, Dict]:
    """Per-model OVER/UNDER vs line from slightly perturbed projections; graded vs actual box score."""
    line_f = float(line_f)
    if line_f <= 0.0:
        return {}
    proj_f = float(proj)
    std_dev = max(2.5, abs(proj_f) * 0.22)
    out: Dict[str, Dict] = {}
    for mkey, mult in _MODEL_GRADE_TWEAKS.items():
        adj = proj_f * (1.0 + float(mult))
        p_over = _projection_to_prob(league_u, adj, line_f, std_dev)
        pick = "OVER" if p_over >= 0.5 else "UNDER"
        hit = (actual > line_f and pick == "OVER") or (actual < line_f and pick == "UNDER")
        out[mkey] = {
            "label": _MODEL_PUBLIC_NAMES.get(mkey, mkey),
            "pick": pick,
            "hit": hit,
            "p_over_pct": round(p_over * 100.0, 1),
        }
    return out


def _summary_accumulate_row(summary: Dict, prop_type: str, hit: bool) -> None:
    if hit:
        summary["overall"]["wins"] += 1
    else:
        summary["overall"]["losses"] += 1
    buck = summary["by_prop_type"].setdefault(prop_type, {"wins": 0, "losses": 0})
    if hit:
        buck["wins"] += 1
    else:
        buck["losses"] += 1


def _empty_results_summary() -> Dict:
    return {
        "overall": {"wins": 0, "losses": 0},
        "by_prop_type": {},
        "by_model": {k: {"wins": 0, "losses": 0, "label": _MODEL_PUBLIC_NAMES.get(k, k)} for k in _MODEL_GRADE_TWEAKS},
    }


def _merge_by_model_from_row(summary: Dict, model_grades: Dict[str, Dict]) -> None:
    for mkey, g in (model_grades or {}).items():
        if mkey not in summary["by_model"]:
            summary["by_model"][mkey] = {"wins": 0, "losses": 0, "label": _MODEL_PUBLIC_NAMES.get(mkey, mkey)}
        if g.get("hit"):
            summary["by_model"][mkey]["wins"] += 1
        else:
            summary["by_model"][mkey]["losses"] += 1


def _finalize_by_model_pct(summary: Dict) -> None:
    for mkey, v in summary.get("by_model", {}).items():
        w, l = int(v.get("wins", 0)), int(v.get("losses", 0))
        t = w + l
        v["win_pct"] = round((w / t) * 100.0, 1) if t else None
        v["total"] = t


def _nba_fetch_boxscore_map(y_schedule: List[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    player_stat_map: Dict[str, Dict[str, Optional[float]]] = {}
    for g in y_schedule:
        event_id = g.get("event_id")
        if not event_id:
            continue
        try:
            s = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
                params={"event": event_id},
                timeout=8,
            ).json()
            sections = (s.get("boxscore", {}).get("players") or [])
            for sec in sections:
                stats_group = (sec.get("statistics") or [])
                if not stats_group:
                    continue
                labels = stats_group[0].get("labels") or []
                idx = {k: i for i, k in enumerate(labels)}
                for ath in stats_group[0].get("athletes") or []:
                    athlete = ath.get("athlete") or {}
                    aid = str(athlete.get("id") or "").strip()
                    name = (athlete.get("displayName") or "").strip()
                    vals = ath.get("stats") or []
                    if not vals:
                        continue
                    blob = _nba_stat_row_from_vals(idx, vals)
                    if aid:
                        player_stat_map[aid] = blob
                    if name:
                        player_stat_map[name.lower()] = blob
        except Exception:
            continue
    return player_stat_map


def _nba_grade_props_for_schedule(league_u: str, y_schedule: List[Dict], graded_iso: str, max_props: int = 200) -> Dict:
    data = _build_league_payload(league_u, schedule_override=y_schedule)
    player_stat_map = _nba_fetch_boxscore_map(y_schedule)
    rows_out: List[Dict] = []
    summary = _empty_results_summary()
    for p in (data.get("props") or [])[:max_props]:
        pid = str(p.get("player_id") or "").strip()
        pname = str(p.get("player_name", "")).strip().lower()
        blob = player_stat_map.get(pid) or player_stat_map.get(pname)
        if not blob:
            continue
        ptype = str(p.get("prop_type", "") or "")
        actual = blob.get(ptype)
        if actual is None:
            continue
        raw_line = p.get("_calc_line")
        if raw_line is None:
            raw_line = p.get("line")
        line = float(raw_line or 0.0)
        if line <= 0.0:
            continue
        pick = str(p.get("picked_side", ""))
        proj = float(p.get("projection", 0.0) or 0.0)
        hit = (actual > line and pick == "OVER") or (actual < line and pick == "UNDER")
        mg = _model_individual_grades(league_u, proj, line, float(actual))
        _summary_accumulate_row(summary, ptype, hit)
        _merge_by_model_from_row(summary, mg)
        rows_out.append(
            {
                "player_id": p.get("player_id"),
                "player_name": p.get("player_name"),
                "team": p.get("team"),
                "prop_type": p.get("prop_type"),
                "pick": pick,
                "line": line,
                "actual": round(float(actual), 2),
                "result": "HIT" if hit else "MISS",
                "projection": p.get("projection"),
                "model_breakdown": mg,
            }
        )
    excluded_rows = data.get("excluded_players") or []
    generated_total = len(data.get("props") or [])
    graded_total = len(rows_out)
    excluded_total = len(excluded_rows)
    reason_counts: Dict[str, int] = {}
    for ex in excluded_rows:
        for reason in ex.get("reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    candidate_total = generated_total + excluded_total
    _finalize_by_model_pct(summary)
    return {
        "graded_date": graded_iso,
        "items": rows_out,
        "summary": summary,
        "generated_props": generated_total,
        "graded_props": graded_total,
        "excluded_players": excluded_total,
        "scope_excluded_players": reason_counts.get("PLAYER_OUT_OF_SCOPE", 0),
        "exclusion_reasons": reason_counts,
        "candidate_players": candidate_total,
    }


def _merge_roll_summary(dst: Dict, src: Dict) -> None:
    dst["overall"]["wins"] += int(src["overall"]["wins"])
    dst["overall"]["losses"] += int(src["overall"]["losses"])
    for pt, b in src.get("by_prop_type", {}).items():
        x = dst["by_prop_type"].setdefault(pt, {"wins": 0, "losses": 0})
        x["wins"] += int(b["wins"])
        x["losses"] += int(b["losses"])
    for mkey in _MODEL_GRADE_TWEAKS:
        if mkey not in dst["by_model"]:
            dst["by_model"][mkey] = {"wins": 0, "losses": 0, "label": _MODEL_PUBLIC_NAMES.get(mkey, mkey)}
        bm = src.get("by_model", {}).get(mkey, {})
        dst["by_model"][mkey]["wins"] += int(bm.get("wins", 0))
        dst["by_model"][mkey]["losses"] += int(bm.get("losses", 0))


def get_league_results(
    league: str,
    result_date_str: Optional[str] = None,
    rollup_days: int = 1,
) -> Dict:
    league_u = league.upper().split(":", 1)[0] if league else ""
    dkey = (result_date_str or "").strip()[:10] if result_date_str else "auto"
    rd = max(1, min(int(rollup_days or 1), 14))
    cache_key = f"{league_u}:{dkey}:r{rd}:results:{_CACHE_SCHEMA}"
    now = time.time()
    cached = _RESULTS_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < 300:
        return cached["payload"]
    if league_u != "NBA":
        # Non-NBA leagues currently don't have full stat-grade integration here.
        # Return a non-empty "latest evaluated board" so results view is useful
        # instead of hardcoded zero rows.
        data = get_league_data(league_u)
        rows = []
        summary = _empty_results_summary()
        for p in (data.get("props") or [])[:40]:
            line = float(p.get("_calc_line", p.get("line", 0.0)) or 0.0)
            projection = float(p.get("projection", 0.0) or 0.0)
            pick = str(p.get("picked_side", ""))
            # Proxy grading: compare projection to line directionally.
            hit = (projection > line and pick == "OVER") or (projection < line and pick == "UNDER")
            _summary_accumulate_row(summary, str(p.get("prop_type", "other")), hit)
            mg = _model_individual_grades(league_u, projection, line, projection)
            _merge_by_model_from_row(summary, mg)
            rows.append(
                {
                    "player_id": p.get("player_id"),
                    "player_name": p.get("player_name"),
                    "team": p.get("team"),
                    "prop_type": p.get("prop_type"),
                    "pick": pick,
                    "line": line,
                    "actual": round(projection, 2),
                    "result": "HIT" if hit else "MISS",
                    "projection": p.get("projection"),
                    "model_breakdown": mg,
                }
            )
        _finalize_by_model_pct(summary)
        excluded_rows = data.get("excluded_players") or []
        reason_counts: Dict[str, int] = {}
        for ex in excluded_rows:
            for reason in ex.get("reasons") or []:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        generated_total = len(data.get("props") or [])
        graded_total = len(rows)
        excluded_total = len(excluded_rows)
        candidate_total = generated_total + excluded_total
        payload = {
            "league": league_u,
            "count": len(rows),
            "items": rows,
            "summary": summary,
            "day_summary": None,
            "graded_date": None,
            "rollup": None,
            "tracking": {
                "generated_props": generated_total,
                "graded_props": graded_total,
                "excluded_players": excluded_total,
                "candidate_players": candidate_total,
                "scope_excluded_players": reason_counts.get("PLAYER_OUT_OF_SCOPE", 0),
                "exclusion_reasons": reason_counts,
            },
        }
        _RESULTS_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload

    et_now = datetime.now(ZoneInfo("America/New_York"))
    parsed = _et_date_from_str(result_date_str)
    ds_flag = (result_date_str or "").strip()

    def _tracking_from_grade(g: Dict) -> Dict:
        return {
            "generated_props": int(g.get("generated_props", 0)),
            "graded_props": int(g.get("graded_props", 0)),
            "excluded_players": int(g.get("excluded_players", 0)),
            "candidate_players": int(g.get("candidate_players", 0)),
            "scope_excluded_players": int(g.get("scope_excluded_players", 0)),
            "exclusion_reasons": dict(g.get("exclusion_reasons") or {}),
        }

    if ds_flag and not parsed:
        empty = _empty_results_summary()
        _finalize_by_model_pct(empty)
        payload = {
            "league": league_u,
            "count": 0,
            "items": [],
            "summary": empty,
            "day_summary": None,
            "graded_date": None,
            "rollup": None,
            "tracking": {
                "generated_props": 0,
                "graded_props": 0,
                "excluded_players": 0,
                "candidate_players": 0,
                "scope_excluded_players": 0,
                "exclusion_reasons": {},
            },
        }
        _RESULTS_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload

    if parsed:
        anchor = parsed.date()
    else:
        anchor = (et_now - timedelta(days=1)).date()

    if rd == 1:
        y_schedule = fetch_schedule_and_teams(league_u, target_date=anchor, strict_day=True)
        if not y_schedule and not ds_flag:
            for i in range(1, 14):
                cand = anchor - timedelta(days=i)
                y_schedule = fetch_schedule_and_teams(league_u, target_date=cand, strict_day=True)
                if y_schedule:
                    anchor = cand
                    break
        if not y_schedule:
            empty = _empty_results_summary()
            _finalize_by_model_pct(empty)
            payload = {
                "league": league_u,
                "count": 0,
                "items": [],
                "summary": empty,
                "day_summary": None,
                "graded_date": anchor.isoformat() if ds_flag else None,
                "rollup": None,
                "tracking": {
                    "generated_props": 0,
                    "graded_props": 0,
                    "excluded_players": 0,
                    "candidate_players": 0,
                    "scope_excluded_players": 0,
                    "exclusion_reasons": {},
                },
            }
            _RESULTS_CACHE[cache_key] = {"ts": now, "payload": payload}
            return payload

        g = _nba_grade_props_for_schedule(league_u, y_schedule, anchor.isoformat())
        payload = {
            "league": league_u,
            "count": len(g["items"]),
            "items": g["items"],
            "summary": g["summary"],
            "day_summary": None,
            "graded_date": g["graded_date"],
            "rollup": None,
            "tracking": _tracking_from_grade(g),
        }
        _RESULTS_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload

    # rd > 1: combined summary over window; item list + day_summary for anchor date only
    combined = _empty_results_summary()
    by_date: List[Dict] = []
    rollup_tracking = {
        "generated_props": 0,
        "graded_props": 0,
        "excluded_players": 0,
        "candidate_players": 0,
        "scope_excluded_players": 0,
        "exclusion_reasons": {},
    }
    for i in range(rd):
        d = anchor - timedelta(days=i)
        ys = fetch_schedule_and_teams(league_u, target_date=d, strict_day=True)
        if not ys:
            day_summary = _empty_results_summary()
            _finalize_by_model_pct(day_summary)
            by_date.append(
                {
                    "date": d.isoformat(),
                    "games": 0,
                    "graded_props": 0,
                    "summary": day_summary,
                }
            )
            continue
        gd = _nba_grade_props_for_schedule(league_u, ys, d.isoformat())
        _merge_roll_summary(combined, gd["summary"])
        rollup_tracking["generated_props"] += int(gd.get("generated_props", 0))
        rollup_tracking["graded_props"] += int(gd.get("graded_props", 0))
        rollup_tracking["excluded_players"] += int(gd.get("excluded_players", 0))
        rollup_tracking["candidate_players"] += int(gd.get("candidate_players", 0))
        rollup_tracking["scope_excluded_players"] += int(gd.get("scope_excluded_players", 0))
        for rk, rv in (gd.get("exclusion_reasons") or {}).items():
            rollup_tracking["exclusion_reasons"][rk] = rollup_tracking["exclusion_reasons"].get(rk, 0) + int(rv)
        by_date.append(
            {
                "date": d.isoformat(),
                "games": len(ys),
                "graded_props": int(gd.get("graded_props", 0)),
                "summary": gd["summary"],
            }
        )

    _finalize_by_model_pct(combined)

    ys_anchor = fetch_schedule_and_teams(league_u, target_date=anchor, strict_day=True)
    if ys_anchor:
        g_anchor = _nba_grade_props_for_schedule(league_u, ys_anchor, anchor.isoformat())
        items = g_anchor["items"]
        day_summary = g_anchor["summary"]
        anchor_tracking = _tracking_from_grade(g_anchor)
    else:
        items = []
        day_summary = _empty_results_summary()
        _finalize_by_model_pct(day_summary)
        anchor_tracking = {
            "generated_props": 0,
            "graded_props": 0,
            "excluded_players": 0,
            "candidate_players": 0,
            "scope_excluded_players": 0,
            "exclusion_reasons": {},
        }

    payload = {
        "league": league_u,
        "count": len(items),
        "items": items,
        "summary": combined,
        "day_summary": day_summary,
        "graded_date": anchor.isoformat(),
        "rollup": {"days": rd, "anchor_date": anchor.isoformat(), "by_date": by_date},
        "tracking": anchor_tracking,
        "rollup_tracking": rollup_tracking,
    }
    _RESULTS_CACHE[cache_key] = {"ts": now, "payload": payload}
    return payload


def filter_props(
    props: List[Dict],
    prop_type: Optional[str] = None,
    side: Optional[str] = None,
    min_ev: Optional[float] = None,
) -> List[Dict]:
    deduped = {}
    for r in props:
        if prop_type and r["prop_type"].lower() != prop_type.lower():
            continue
        if side and r["picked_side"].lower() != side.lower():
            continue
        sel_ev = r["ev_over_percent"] if r["picked_side"] == "OVER" else r["ev_under_percent"]
        if min_ev is not None and sel_ev < min_ev:
            continue
        # Keep one row per player+prop to avoid duplicate cards/rows.
        key = (r.get("player_id"), r.get("prop_type"))
        cur = deduped.get(key)
        if cur is None:
            deduped[key] = r
            continue
        cur_ev = cur["ev_over_percent"] if cur["picked_side"] == "OVER" else cur["ev_under_percent"]
        if (sel_ev, r.get("confidence_score", 0.0)) > (cur_ev, cur.get("confidence_score", 0.0)):
            deduped[key] = r
    out = list(deduped.values())
    out.sort(
        key=lambda x: (
            -(x["ev_over_percent"] if x["picked_side"] == "OVER" else x["ev_under_percent"]),
            -x["confidence_score"],
            -max(x["over_probability"], x["under_probability"]),
        )
    )
    return out
