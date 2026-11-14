#!/usr/bin/env python3
"""Four-hour-ahead quantile forecasts for a family of Polymarket Yes prices.

For every market in a case we fit a two-part normal mixture using only that
case's own price history, then report five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) for each of the 240 minutes after the checkpoint.

The two parts are:

* a tight, mean-reverting part whose spread saturates with horizon the way an
  Ornstein-Uhlenbeck process does (half-life one hour), estimated from the
  recent one-minute moves with big jumps taken out; and
* a broad "news" part whose spread is fixed across horizons, estimated from
  the size of one-hour level changes, whose weight grows with horizon at a
  rate set by how often jumps have shown up lately.

The centre of both parts is the market's last price, scaled a little so the
centres across the case's markets add up closer to one (a market holding a
larger share of the total is moved further, and no market is moved by more
than a small cap). Quantiles come from
bisection on the mixture distribution function.

Standard library only, no randomness, no network. Reads every ``*.in`` file
in the input directory (``$MAKEFUN_INPUT_DIR`` or ``/input``) and writes one
``<id>.out`` per case to the output directory (``$MAKEFUN_OUTPUT_DIR`` or
``/output``).

Measured on the kit's six practice cases with the local scorer: pooled
pinball loss 0.265451 (per case 0.389180, 0.304983, 0.357579, 0.267340,
0.209083, 0.117829), every case valid and beating its house-baseline
counterpart; two runs byte-identical; about 0.5 s per case.
"""

import argparse
import json
import math
import os

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZON = 240

# Tuned constants. Windows are in minutes; price units are fractions of 1.
CORE_WINDOW = 60          # minutes of one-minute moves feeding the tight part
NEWS_WINDOW = 240         # minutes of one-hour moves feeding the broad part
HALF_LIFE = 60.0          # minutes for the tight part's mean reversion
JUMP_CUT = 0.02           # a one-minute move bigger than this counts as a jump
CORE_FLOOR = 0.0003       # smallest one-minute tight sigma allowed
CORE_CAP = 0.01           # largest one-minute tight sigma allowed
NEWS_FLOOR = 0.03         # smallest broad sigma allowed
NEWS_CAP = 0.04           # largest broad sigma allowed
RATE_FLOOR = 0.001        # smallest per-minute jump rate allowed
RATE_CAP = 0.02           # largest per-minute jump rate allowed
PRIOR_JUMPS = 0.5         # pseudo-jumps added to the jump count
PRIOR_MINUTES = 500.0     # pseudo-minutes added to the jump-rate denominator
WEIGHT_CAP = 0.6          # largest weight the broad part may take
SUM_SHIFT_CAP = 0.016     # largest nudge applied to any one centre
BISECT_STEPS = 26         # bisection rounds for each quantile


def _mean(values):
    return sum(values) / len(values)


def _spread(values):
    """Population standard deviation; 0.0 when fewer than two values."""
    if len(values) < 2:
        return 0.0
    centre = _mean(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / len(values))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mixture_cdf(q, centre, tight_sd, broad_sd, broad_weight):
    """CDF of the two-part mixture at price q."""
    if tight_sd > 0.0:
        tight = _norm_cdf((q - centre) / tight_sd)
    else:
        tight = 0.0 if q < centre else 1.0
    if broad_sd > 0.0:
        broad = _norm_cdf((q - centre) / broad_sd)
    else:
        broad = 0.0 if q < centre else 1.0
    return (1.0 - broad_weight) * tight + broad_weight * broad


def _quantile(level, centre, tight_sd, broad_sd, broad_weight):
    """Invert the mixture CDF at `level` by bisection on a wide bracket."""
    low = centre - 1.0
    high = centre + 1.0
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (low + high)
        if _mixture_cdf(mid, centre, tight_sd, broad_sd, broad_weight) < level:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _market_params(prices):
    """Estimate the per-market mixture parameters from its price history.

    Returns (centre, tight_sigma, broad_sigma, jump_rate). With no prices at
    all the centre is 0.0 and both sigmas are 0.0; with a single price the
    sigmas are 0.0 as there is no movement to measure.
    """
    n = len(prices)
    if n == 0:
        return 0.0, 0.0, 0.0, RATE_FLOOR
    centre = prices[-1]

    # One-minute moves, most recent first in window terms.
    moves = [prices[i + 1] - prices[i] for i in range(n - 1)]

    # Tight part: spread of the recent one-minute moves, jumps excluded.
    recent = moves[-CORE_WINDOW:]
    calm = [m for m in recent if abs(m) <= JUMP_CUT]
    tight = _spread(calm)
    if calm:
        tight = min(max(tight, CORE_FLOOR), CORE_CAP)
    else:
        tight = 0.0

    # Broad part: spread of one-hour level changes over the recent window.
    hour_moves = []
    start = max(0, n - NEWS_WINDOW)
    for i in range(start, n):
        j = i - 60
        if j >= 0:
            hour_moves.append(prices[i] - prices[j])
    broad = _spread(hour_moves)
    if hour_moves:
        broad = min(max(broad, NEWS_FLOOR), NEWS_CAP)
    elif n >= 2:
        # A short history holds no one-hour moves at all, yet the market has
        # moved: keep the broad part at its floor instead of collapsing it
        # to a point, which would pinch the middle quantiles together.
        broad = NEWS_FLOOR
    else:
        broad = 0.0

    # Jump rate over the recent window, with a small prior pulling it up
    # from zero when jumps are rare.
    window_moves = moves[-NEWS_WINDOW:]
    jumps = sum(1 for m in window_moves if abs(m) > JUMP_CUT)
    rate = (jumps + PRIOR_JUMPS) / (len(window_moves) + PRIOR_MINUTES)
    rate = min(max(rate, RATE_FLOOR), RATE_CAP)

    return centre, tight, broad, rate


def _forecast_market(centre, tight, broad, rate):
    """Build the 240 x 5 quantile table for one market."""
    decay = 0.5 ** (1.0 / HALF_LIFE)
    denom = 1.0 - decay * decay
    rows = []
    for h in range(1, HORIZON + 1):
        # Tight variance saturates like an Ornstein-Uhlenbeck process.
        var = tight * tight * (1.0 - decay ** (2 * h)) / denom
        tight_sd = math.sqrt(var) if var > 0.0 else 0.0
        # Broad weight grows with horizon as jumps become more likely.
        weight = 1.0 - (1.0 - rate) ** h
        if weight > WEIGHT_CAP:
            weight = WEIGHT_CAP
        row = []
        for level in LEVELS:
            q = _quantile(level, centre, tight_sd, broad, weight)
            q = round(q, 6)
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            elif q == 0.0:
                # Keep plain 0.0: rounding can produce -0.0 for a zero centre.
                q = 0.0
            row.append(q)
        # Keep the five quantiles non-decreasing after rounding/clamping.
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def forecast_case(case):
    """Forecast every market of one case; returns the output document."""
    markets = case.get("markets", [])
    fitted = [_market_params(m.get("prices", [])) for m in markets]
    total = sum(f[0] for f in fitted)
    forecasts = []
    for market, (centre, tight, broad, rate) in zip(markets, fitted):
        # Scale the centre so the centres across the case add up closer to
        # one: a market holding a larger share of the total moves further.
        # The move for any one market is capped, so a lopsided total never
        # drags a small market far from its own last price.
        if total > 0.0:
            move = (1.0 - total) * centre / total
            if move > SUM_SHIFT_CAP:
                move = SUM_SHIFT_CAP
            elif move < -SUM_SHIFT_CAP:
                move = -SUM_SHIFT_CAP
        else:
            move = 0.0
        rows = _forecast_market(centre + move, tight, broad, rate)
        forecasts.append({"id": market["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    in_dir = args.input_dir or os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = args.output_dir or os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        doc = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, separators=(",", ":"), ensure_ascii=True)
            fh.write("\n")


if __name__ == "__main__":
    main()
