"""
Human-readable spread / ATS / total copy for results UI.

Spread quoting (standard board):
  • Favorite → negative number (e.g. −10). They cover only if they win by *more than* 10.
  • Underdog → positive number (e.g. +10). They cover if they win outright *or* lose by fewer than 10.

Internal grading (unchanged, matches NHL77FINAL):
  • margin_home = home_score − away_score.
  • line_home = closing spread attached to the home team in the listing (− if home is favored).

UI copy uses favorite/underdog and team names — not “home line” jargon.

Spread and total *lines* (O/U) use half-point increments only (.0 / .5).
Final scores and combined points stay whatever the game produced (usually whole points).
"""

from __future__ import annotations

import math

_EPS = 1e-9


def round_spread_half(value):
    """Whole or half points only — standard sportsbook spread increments."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    out = round(x * 2.0) / 2.0
    return 0.0 if abs(out) < _EPS else out


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def ats_cover_side(margin_home: float, line_home: float) -> str | None:
    """Return 'HOME', 'AWAY', or 'PUSH'; None if inputs invalid."""
    mh = _finite(margin_home)
    hl = _finite(line_home)
    if mh is None or hl is None:
        return None
    # Same test as NHL77FINAL._actual_ats_side
    v = mh + hl
    if abs(v) < _EPS:
        return 'PUSH'
    return 'HOME' if v > 0 else 'AWAY'


def describe_ats_closing_line(
    home_team: str,
    away_team: str,
    margin_home,
    line_home,
) -> str | None:
    """
    Explain ATS vs the closing spread in plain language.

    Example:
      Thunder favored by 15.5 at home; won by 18 → covers by 2.5 points.
    """
    mh = _finite(margin_home)
    hl = round_spread_half(line_home)
    if mh is None or hl is None:
        return None

    side = ats_cover_side(mh, hl)
    if side is None:
        return None

    # Two-sided quote: minus = favorite (must cover), plus = dog (hang within or win).
    fav_line = abs(hl)
    if hl < -_EPS:
        closing = f"Closing: {home_team} −{fav_line:.1f} · {away_team} +{fav_line:.1f}"
    elif hl > _EPS:
        closing = f"Closing: {away_team} −{fav_line:.1f} · {home_team} +{fav_line:.1f}"
    else:
        closing = "Closing: pick'em (0)"

    # Same number as the “Final margin” row: home points − away points (not a spread price).
    margin_txt = f"Scoring margin {mh:+.1f} (home − away on the scoreboard)."

    if side == 'PUSH':
        return f"{closing}. {margin_txt} Push against the spread."

    cover_team = home_team if side == 'HOME' else away_team
    residual = mh + hl
    beat_disp = round_spread_half(abs(residual))
    if beat_disp is None:
        beat_disp = abs(residual)
    return (
        f"{closing}. {margin_txt} {cover_team} covers — "
        f"{beat_disp:.1f} points clear of the number."
    )


def describe_xsharp_vs_final_margin(
    home_team: str,
    away_team: str,
    margin_home,
    xsharp_margin,
) -> str | None:
    """
    Compare actual final margin to XSharp projected margin (both home − away).

    Uses team names instead of HOME/AWAY tokens.
    """
    xm_raw = _finite(xsharp_margin)
    am = _finite(margin_home)
    if xm_raw is None or am is None:
        return None
    xm = round_spread_half(xm_raw)
    if xm is None:
        return None

    def _side_from_margin(m: float) -> str | None:
        if abs(m) <= _EPS:
            return None
        return home_team if m > 0 else away_team

    pred_winner = _side_from_margin(xm)
    act_winner = _side_from_margin(am)

    ax = abs(xm)
    aa = abs(am)

    if pred_winner is None:
        return (
            f"XSharp near pick'em (projected margin {xm:+.1f}). "
            f"Final margin {am:+.1f} — {act_winner or 'even'} by {aa:.1f}."
        )

    if act_winner is None:
        return (
            f"XSharp projected {pred_winner} by {ax:.1f}. "
            f"Final margin {am:+.1f} (effectively tied)."
        )

    match = pred_winner == act_winner
    direction = "Same side as final" if match else "Opposite side of final"

    return (
        f"XSharp projected {pred_winner} by {ax:.1f}; "
        f"final {act_winner} by {aa:.1f}. {direction}."
    )


def describe_ou_vs_line(actual_total, line) -> str | None:
    at = _finite(actual_total)
    lr = _finite(line)
    ln = round_spread_half(lr) if lr is not None else None
    if at is None or ln is None:
        return None
    if abs(at - ln) < _EPS:
        return f"Push — combined score {at:.1f} landed on the closing total {ln:.1f}."
    side = "OVER" if at > ln else "UNDER"
    diff = abs(at - ln)
    return f"{side} — final combined {at:.1f} vs closing total {ln:.1f} (gap {diff:.1f})."


def describe_ou_vs_xsharp(actual_total, xsharp_adj, xsharp_raw) -> str | None:
    ref_raw = xsharp_adj if xsharp_adj is not None else xsharp_raw
    ref_a = _finite(ref_raw)
    at = _finite(actual_total)
    if ref_a is None or at is None:
        return None
    ref = round_spread_half(ref_a)
    if abs(at - ref) < _EPS:
        return f"Final combined {at:.1f} matched XSharp projection {ref:.1f}."
    side = "OVER" if at > ref else "UNDER"
    tail = ""
    if xsharp_adj is not None and xsharp_raw is not None:
        try:
            if abs(float(xsharp_adj) - float(xsharp_raw)) > 0.05:
                tail = " (adjusted XSharp total)"
        except (TypeError, ValueError):
            pass
    return f"{side} — final {at:.1f} vs XSharp {ref:.1f}{tail}."
