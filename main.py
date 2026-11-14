"""Polymarket FAA outage attribution: 4-hour quantile forecaster.

Forecasts the Yes price of every market of the FAA outage attribution
family, minute by minute, 4 hours ahead, as five quantiles per minute.

Model (per market, fitted only on the case's own price history):
  - Two-component normal mixture for the h-step-ahead price:
      * core: an Ornstein-Uhlenbeck (mean-reverting) component. Its
        1-minute sigma is the std of non-jump returns over the last
        60 min; its variance saturates with horizon (half-life 60 min)
        instead of growing like a random walk, because these prices are
        strongly mean-reverting.
      * news: a wide component centred on the last price, whose sigma is
        the std of 60-minute level moves over the last 4 h (floored at
        0.03, capped at 0.04) and whose weight grows with horizon as
        1-(1-p)^h, p = jump rate over the last 4 h (moves > 0.02 count
        as jumps).
  - Median stays at the last price (no drift: momentum was tested on the
    sibling event's practice cases and hurts).
  - Medians are shrunk slightly toward a sum of 1.0 across markets (the
    family's probabilities must sum to ~1; shift capped at 0.016/market).

Tuned on the 6 practice cases (same event shape, NVIDIA event) against
live Polymarket answers:
  pooled pinball loss 0.265396 vs house baseline 0.363589.
For this FAA competition a focused 11-variant sweep around the winning
config (sq_win 45/90, news_floor 0.025/0.035, sum1_max_shift 0.014/0.018
and their combinations) found nothing that cleared the bar: the best
combination reached only 0.264026, a 0.0014 gain driven by two cases
while two others got slightly worse, so the round-4 config stands.
Stdlib only, deterministic, no network, no randomness.
"""
import argparse
import json
import math
import os

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
SQRT2 = math.sqrt(2.0)

DEFAULTS = dict(
    window=240,          # minutes of history for parameter estimation
    jump_thresh=0.02,    # |return| above this counts as a jump (for p_news)
    p_floor=0.001,       # floor on per-minute news probability
    p_cap=0.02,          # cap on per-minute news probability
    p_prior=0.001,       # Beta prior mean for p_news
    p_prior_n=500,       # Beta prior strength (pseudo-minutes)
    sq_floor=0.0003,     # floor on core 1-min sigma
    sq_cap=0.01,         # cap on core 1-min sigma
    sq_win=60,           # minutes for core-sigma estimation (0 = same as window)
    hl=60,               # mean-reversion half-life (min) for the core; 0 = none
    news_win=240,        # minutes for the news-size estimate
    news_cap=0.04,       # cap on news sigma
    news_floor=0.03,     # floor on news sigma
    w_cap=0.6,           # cap on the news weight w(h)
    drift_win=0,         # minutes for drift estimation (0 = no drift)
    drift_shrink=0.0,
    drift_cap=0.02,
    level_win=360,       # minutes for the recent-mean level
    level_hl=0,          # half-life (min) for centre gliding; 0 = off
    sum1=True,
    sum1_target=1.0,
    sum1_max_shift=0.016,
    asym=False,          # split-normal (separate up/down scales)
    hot_boost=0.0,       # extra news weight when a jump hit in last 60 min
)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def market_params(prices, P):
    if not prices:
        prices = [0.0]
    n = len(prices)
    w = min(P["window"], n - 1)
    rets = [prices[i] - prices[i - 1] for i in range(n - w, n)]
    sqw = min(P["sq_win"], n - 1) if P["sq_win"] else w
    rets_sq = [prices[i] - prices[i - 1] for i in range(n - sqw, n)]
    J = P["jump_thresh"]
    njump = sum(1 for r in rets if abs(r) > J)
    p = (njump + P["p_prior"] * P["p_prior_n"]) / (len(rets) + P["p_prior_n"])
    p = min(max(p, P["p_floor"]), P["p_cap"])
    # Core sigma: std of non-jump returns over the short window.
    quiet = [r for r in rets_sq if abs(r) <= J]
    if len(quiet) > 1:
        mq = sum(quiet) / len(quiet)
        sq = math.sqrt(sum((r - mq) ** 2 for r in quiet) / (len(quiet) - 1))
    elif quiet:
        sq = abs(quiet[0])
    else:
        sq = 0.0
    sq = min(max(sq, P["sq_floor"]), P["sq_cap"])
    # Asymmetric scales (only used when asym is on): RMS of up/down returns.
    up = [r for r in rets_sq if r > 0]
    dn = [r for r in rets_sq if r < 0]
    sq_up = math.sqrt(sum(r * r for r in up) / len(up)) if up else sq
    sq_dn = math.sqrt(sum(r * r for r in dn) / len(dn)) if dn else sq
    sq_up = min(max(sq_up, P["sq_floor"]), P["sq_cap"])
    sq_dn = min(max(sq_dn, P["sq_floor"]), P["sq_cap"])
    # Hot flag: a jump in the last 60 minutes?
    hot = 1.0 if any(abs(r) > J for r in rets[-60:]) else 0.0
    # News sigma: std of 60-minute level moves in the news window.
    nw = min(P["news_win"], n - 60)
    moves60 = [prices[i] - prices[i - 60] for i in range(n - nw, n)] if nw > 0 else []
    if len(moves60) > 1:
        mm = sum(moves60) / len(moves60)
        sn = math.sqrt(sum((v - mm) ** 2 for v in moves60) / (len(moves60) - 1))
    else:
        sn = P["news_floor"]
    sn = min(max(sn, P["news_floor"]), P["news_cap"])
    # Asymmetric news scales: RMS of up/down jumps.
    jup = [r for r in rets if r > J]
    jdn = [r for r in rets if r < -J]
    sn_up = math.sqrt(sum(r * r for r in jup) / len(jup)) if jup else sn
    sn_dn = math.sqrt(sum(r * r for r in jdn) / len(jdn)) if jdn else sn
    sn_up = min(max(sn_up, P["news_floor"]), P["news_cap"])
    sn_dn = min(max(sn_dn, P["news_floor"]), P["news_cap"])
    # Drift (off by default).
    drift = 0.0
    if P["drift_win"] and P["drift_shrink"]:
        dw = min(P["drift_win"], n - 1)
        dret = [prices[i] - prices[i - 1] for i in range(n - dw, n)]
        drift = (sum(dret) / len(dret)) * 60.0 * P["drift_shrink"]
        drift = max(-P["drift_cap"], min(P["drift_cap"], drift))
    lw = min(P["level_win"], n)
    mean_recent = sum(prices[n - lw:]) / lw if lw > 0 else prices[-1]
    return {"x": prices[-1], "p": p, "sq": sq, "sn": sn, "drift": drift,
            "mean_recent": mean_recent, "sq_up": sq_up, "sq_dn": sq_dn,
            "sn_up": sn_up, "sn_dn": sn_dn, "hot": hot}


def split_cdf(x, mu, s_dn, s_up):
    """CDF of the two-piece normal: scale s_dn below mu, s_up above."""
    if x < mu:
        return (2.0 * s_dn / (s_up + s_dn)) * norm_cdf((x - mu) / s_dn)
    return (s_dn / (s_up + s_dn)
            + (2.0 * s_up / (s_up + s_dn)) * (norm_cdf((x - mu) / s_up) - 0.5))


def mixture_quantiles(mu, s_core, s_news, w, iters=26, asym=None):
    """Quantiles of the two-component mixture by bisection on its CDF.

    asym: None (symmetric) or (core_up, core_dn, news_up, news_dn).
    """
    s_core = max(s_core, 1e-9)
    s_news = max(s_news, 1e-9)
    s = max(s_core, s_news)
    lo = mu - 6.0 * s - 0.02
    hi = mu + 6.0 * s + 0.02
    los = [lo] * 5
    his = [hi] * 5
    for _ in range(iters):
        for j, tau in enumerate(TAUS):
            mid = 0.5 * (los[j] + his[j])
            if asym is None:
                f = (1.0 - w) * norm_cdf((mid - mu) / s_core) + w * norm_cdf((mid - mu) / s_news)
            else:
                f = (1.0 - w) * split_cdf(mid, mu, asym[1], asym[0]) + w * split_cdf(mid, mu, asym[3], asym[2])
            if f < tau:
                los[j] = mid
            else:
                his[j] = mid
    return [0.5 * (los[j] + his[j]) for j in range(5)]


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_case(case, P):
    horizon = case["horizon_minutes"]
    hl = P["hl"]
    if hl and hl > 0:
        phi = 0.5 ** (1.0 / hl)
        denom = 1.0 - phi * phi
    else:
        phi = 1.0
        denom = None
    params = [(m["id"], market_params(m["prices"], P)) for m in case["markets"]]
    shifts = {}
    if P["sum1"]:
        s = sum(p["x"] for _, p in params)
        if s > 0:
            for mid, p in params:
                target = p["x"] * (P["sum1_target"] / s)
                shifts[mid] = max(-P["sum1_max_shift"],
                                  min(P["sum1_max_shift"], target - p["x"]))
    out = {}
    for mid, p in params:
        x = p["x"]
        shift = shifts.get(mid, 0.0)
        rows = []
        lhl = P["level_hl"]
        for h in range(1, horizon + 1):
            mu = x + p["drift"] * h / 60.0 + shift
            if lhl:
                mu += (p["mean_recent"] - x) * (1.0 - 0.5 ** (h / lhl))
            if denom is not None:
                var_core = p["sq"] ** 2 * (1.0 - phi ** (2 * h)) / denom
            else:
                var_core = p["sq"] ** 2 * h
            s_core = math.sqrt(var_core)
            w = min(1.0 - (1.0 - p["p"]) ** h, P["w_cap"])
            if P["hot_boost"]:
                w = min(w * (1.0 + P["hot_boost"] * p["hot"]), P["w_cap"])
            asym = None
            if P["asym"]:
                asym = (p["sq_up"] * (s_core / p["sq"]) if p["sq"] > 0 else p["sq_up"],
                        p["sq_dn"] * (s_core / p["sq"]) if p["sq"] > 0 else p["sq_dn"],
                        p["sn_up"], p["sn_dn"])
            qs = mixture_quantiles(mu, s_core, p["sn"], w, asym=asym)
            row = [price6(v) for v in qs]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        out[mid] = rows
    return out


def run(input_dir, output_dir, P):
    os.makedirs(output_dir, exist_ok=True)
    for fn in sorted(os.listdir(input_dir)):
        if not fn.endswith(".in"):
            continue
        cid = fn[:-3]
        with open(os.path.join(input_dir, fn)) as f:
            case = json.load(f)
        fc = forecast_case(case, P)
        forecasts = [{"id": m["id"], "q": fc[m["id"]]} for m in case["markets"]]
        with open(os.path.join(output_dir, f"{cid}.out"), "w") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")
        print(f"{cid}: {len(forecasts)} markets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    P = dict(DEFAULTS)
    P.update(json.loads(a.params))
    inp = a.input or os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    outp = a.output or os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    run(inp, outp, P)
