"""
Genetic search over the 3-EMA crossover and the contract expiry, with a live equity page.

A standalone research script: it loads one span of V75 quotes once, computes every EMA length
once with vectorbt, then backtests settings in parallel on 80% of the CPU cores. Money uses the
shared exact Deriv rules (stake, 95.2% payout, tie loses, 5 open contracts). Settings are
searched on the first 70% of the span; the winner is tested once on the last 30%.

Open the printed address in a browser: every finished backtest adds a gray equity curve and the
best one so far is drawn in red.

Example:
    python scripts/ema_genetic.py --start 2026-07-20T00:00:00Z --end 2026-08-17T00:00:00Z \
        --output /tmp/ema-genetic
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import plotly.offline
import vectorbt as vbt
from deriv_mt5.rules import gross_credit_cents, is_win, stake_cents
from vectorbt_research.dataset import read_quote_window
from vectorbt_research.input import _utc_nanoseconds as utc_ns
from vectorbt_research.input import parse_quote_rows

DATASET = Path("/mnt/c/LeonAI-DO/dev/TRADING/Datasets/Volatility75Index_yearly_deduped")
MINUTE_NS = 60_000_000_000
FRESH_NS = 10_000_000_000  # an exit price up to 10 s before expiry is fresh
MAX_OPEN = 5
START_CASH = 1_000_000  # cents
MAX_DRAWDOWN = 0.10
FAST, MIDDLE, SLOW = (5, 20), (15, 60), (100, 300)
EXPIRIES = (1, 5, 15, 60)  # minutes
CURVE_POINTS = 200  # points kept per gray curve on the live page

# Filled once in the parent before the worker processes fork, then only read.
DATA: dict = {}


def load(start_ns, end_ns, scenario):
    """
    Load quotes (plus one hour after the end for open contracts) and every EMA length.
    """
    year = time.gmtime(start_ns // 10**9).tm_year
    rows = read_quote_window(
        DATASET / f"Volatility75Index_{year}.parquet", start_ns, end_ns + 61 * MINUTE_NS
    )
    quotes, _ = parse_quote_rows(rows)
    field = f"{scenario}_half_cents"
    ts = np.array([q.timestamp_ns for q in quotes], dtype=np.int64)
    price = np.array([getattr(q, field) for q in quotes], dtype=np.int64)
    in_span = ts < end_ns
    minute = ts[in_span] // MINUTE_NS
    last = np.flatnonzero(np.diff(minute, append=minute[-1] + 1))
    close = price[in_span][last].astype(np.float64)
    windows = sorted(set(range(FAST[0], SLOW[1] + 1)))
    emas = vbt.MA.run(close, window=windows, ewm=True).ma.to_numpy()
    DATA.update(
        ts=ts,
        price=price,
        bar_close_ns=(minute[last] + 1) * MINUTE_NS,
        close=close,
        ema_column={w: i for i, w in enumerate(windows)},
        emas=emas,
    )


def signals(fast, middle, slow, trend=False):
    """
    Return (start_ns, is_call) for every bar where the close crosses all three EMAs.

    With `trend`, a Rise also needs the slow EMA rising on that bar, and a Fall falling.
    """
    cols = [DATA["ema_column"][w] for w in (fast, middle, slow)]
    ema = DATA["emas"][:, cols]
    ready = ~np.isnan(ema).any(axis=1)
    close = DATA["close"][:, None]
    above = ready & (close > ema).all(axis=1)
    below = ready & (close < ema).all(axis=1)
    was_ready = np.concatenate(([False], ready[:-1]))
    rise = was_ready & above & ~np.concatenate(([False], above[:-1]))
    fall = was_ready & below & ~np.concatenate(([False], below[:-1]))
    if trend:
        slope = np.diff(ema[:, 2], prepend=np.nan)
        rise &= slope > 0
        fall &= slope < 0
    fire = rise | fall
    return DATA["bar_close_ns"][fire], rise[fire]


def simulate(fast, middle, slow, expiry_min, from_ns, to_ns, trend=False):
    """
    Backtest one setting on signals starting in [from_ns, to_ns) with the exact cash rules.

    Returns the equity curve (times and account values in cents) and the counts.
    """
    ts, price = DATA["ts"], DATA["price"]
    start, is_call = signals(fast, middle, slow, trend)
    keep = (start >= from_ns) & (start < to_ns)
    start, is_call = start[keep], is_call[keep]
    expiry = start + expiry_min * MINUTE_NS
    entry = np.searchsorted(ts, start, side="left")
    exit_ = np.searchsorted(ts, expiry, side="right") - 1
    available, open_, skipped, unresolved = START_CASH, [], 0, 0
    times, values = [from_ns], [START_CASH]
    trades = wins = profit = loss = 0
    in_market = []  # (entry time, expiry) of every contract taken

    def settle(until_ns):
        """Pay every due contract in admission order; value = free cash + locked stakes."""
        nonlocal available, trades, wins, profit, loss
        for contract in list(open_):
            due, stake, credit, fresh = contract
            if fresh and due <= until_ns:
                open_.remove(contract)
                available += credit
                trades += 1
                wins += credit > 0
                profit += max(0, credit - stake)
                loss += stake if credit == 0 else 0
                times.append(due)
                values.append(available + sum(c[1] for c in open_))

    for i in range(len(start)):
        if entry[i] >= len(ts) or ts[entry[i]] > expiry[i]:
            skipped += 1
            continue
        settle(ts[entry[i]])
        if len(open_) >= MAX_OPEN:
            skipped += 1
            continue
        stake = stake_cents(available)
        if stake is None:
            skipped += 1
            continue
        available -= stake
        in_market.append((int(ts[entry[i]]), int(expiry[i])))
        fresh = exit_[i] >= entry[i] and expiry[i] - ts[exit_[i]] <= FRESH_NS
        direction = "CALL" if is_call[i] else "PUT"
        won = fresh and is_win(direction, int(price[entry[i]]), int(price[exit_[i]]))
        open_.append(
            (int(expiry[i]), stake, gross_credit_cents(stake, bool(won)), fresh)
        )
    settle(np.iinfo(np.int64).max)
    unresolved = len(open_)
    values_arr = np.array(values, dtype=np.int64)
    peaks = np.maximum.accumulate(values_arr)
    drawdown = float(((peaks - values_arr) / peaks).max())
    return {
        "times": times,
        "values": values,
        "net_cents": int(values_arr[-1] - START_CASH),
        "available_cents": available,
        "drawdown": drawdown,
        "trades": trades,
        "wins": wins,
        "skipped": skipped,
        "unresolved": unresolved,
        "profit_cents": profit,
        "loss_cents": loss,
        "in_market": in_market,
    }


def fitness(result):
    """Net profit; settings that break the 10% drawdown limit rank below every valid one."""
    penalty = 0 if result["drawdown"] <= MAX_DRAWDOWN else 10**12
    return result["net_cents"] - penalty


def evaluate(genome, from_ns, to_ns):
    result = simulate(*genome, from_ns, to_ns)
    return genome, result


def random_genome(rng):
    while True:
        fast = rng.randint(*FAST)
        middle = rng.randint(*MIDDLE)
        slow = rng.randint(*SLOW)
        if fast < middle < slow:
            return (fast, middle, slow, rng.choice(EXPIRIES))


def child(a, b, rng):
    """Uniform crossover, then mutate each gene with probability 0.2; keep fast < middle < slow."""
    for _ in range(100):
        genes = [rng.choice(pair) for pair in zip(a, b, strict=True)]
        for i, (low, high), step in zip(
            range(3), (FAST, MIDDLE, SLOW), (2, 5, 20), strict=True
        ):
            if rng.random() < 0.2:
                genes[i] = min(high, max(low, genes[i] + rng.randint(-step, step)))
        if rng.random() < 0.2:
            genes[3] = rng.choice(EXPIRIES)
        if genes[0] < genes[1] < genes[2]:
            return tuple(genes)
    return random_genome(rng)


def thin(times, values):
    """Keep at most CURVE_POINTS points of a curve for the live page."""
    if len(times) <= CURVE_POINTS:
        return times, values
    pick = np.linspace(0, len(times) - 1, CURVE_POINTS).astype(int)
    return [times[i] for i in pick], [values[i] for i in pick]


class LivePage:
    """Serves the output folder and rewrites live.json as results arrive."""

    def __init__(self, folder, port):
        self.folder, self.curves, self.best, self.status = folder, [], None, {}
        (folder / "plotly.min.js").write_text(
            plotly.offline.get_plotlyjs(), encoding="utf-8"
        )
        (folder / "index.html").write_text(_PAGE, encoding="utf-8")
        handler = partial(_QuietHandler, directory=str(folder))
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.last_write = 0.0

    def add(self, genome, result, is_best):
        t, v = thin(result["times"], result["values"])
        self.curves.append({"x": [x // 10**6 for x in t], "y": [y / 100 for y in v]})
        if is_best:
            self.best = {
                "x": [x // 10**6 for x in result["times"]],
                "y": [y / 100 for y in result["values"]],
            }
        if is_best or time.time() - self.last_write > 0.5:
            self.write()

    def write(self):
        data = {"status": self.status, "gray": self.curves, "best": self.best}
        tmp = self.folder / "live.json.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.folder / "live.json")
        self.last_write = time.time()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>EMA genetic search</title>
<script src="plotly.min.js"></script><style>body{font-family:sans-serif;margin:12px}
#s{font-size:14px;margin-bottom:6px}</style></head><body><div id="s">Waiting for results…</div>
<div id="c" style="height:80vh"></div><script>
async function tick(){try{const d=await (await fetch('live.json?'+Date.now())).json();
const s=d.status;document.getElementById('s').textContent=
`Round ${s.round}/${s.rounds} · backtests ${s.done} · best: EMA ${s.best_setting} · `+
`net ${s.best_net} USD · drawdown ${s.best_dd} · trades ${s.best_trades}`+(s.test?` · held-back test: ${s.test}`:'');
const tr=d.gray.map(c=>({x:c.x.map(t=>new Date(t)),y:c.y,mode:'lines',type:'scattergl',
line:{color:'rgba(128,128,128,0.25)',width:1},hoverinfo:'skip',showlegend:false}));
if(d.best)tr.push({x:d.best.x.map(t=>new Date(t)),y:d.best.y,mode:'lines',type:'scattergl',
line:{color:'red',width:2.5},name:'Best so far'});
Plotly.react('c',tr,{margin:{t:10},yaxis:{title:'Account value (USD)'},xaxis:{title:'Time (UTC)'}});
}catch(e){} setTimeout(tick,1000)} tick();</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default="mid", choices=["bid", "mid", "ask"])
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true", help="only run 9/21/200, 1 min")
    args = parser.parse_args()
    start_ns, end_ns = utc_ns(args.start), utc_ns(args.end)
    t0 = time.time()
    load(start_ns, end_ns, args.scenario)
    print(
        f"loaded {len(DATA['ts']):,} ticks, {len(DATA['close']):,} bars in {time.time() - t0:.1f} s"
    )
    if args.check:
        r = simulate(9, 21, 200, 1, start_ns, end_ns)
        print({k: v for k, v in r.items() if k not in ("times", "values", "in_market")})
        return
    split_ns = start_ns + (end_ns - start_ns) * 7 // 10
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    page = LivePage(folder, args.port)
    print(f"live page: http://127.0.0.1:{args.port}/", flush=True)
    workers = max(1, int(os.cpu_count() * 0.8))
    rng = random.Random(args.seed)
    seen, population = {}, []
    while len(population) < args.population:
        g = random_genome(rng)
        if g not in population:
            population.append(g)
    best = None
    t1 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for round_ in range(1, args.rounds + 1):
            todo = [g for g in population if g not in seen]
            for genome, result in pool.imap_unordered(
                partial(evaluate, from_ns=start_ns, to_ns=split_ns), todo
            ):
                seen[genome] = result
                is_best = best is None or fitness(result) > fitness(seen[best])
                if is_best:
                    best = genome
                b = seen[best]
                page.status = {
                    "round": round_,
                    "rounds": args.rounds,
                    "done": len(seen),
                    "best_setting": f"{best[0]}/{best[1]}/{best[2]}, expiry {best[3]} min",
                    "best_net": f"{b['net_cents'] / 100:,.2f}",
                    "best_dd": f"{b['drawdown']:.2%}",
                    "best_trades": b["trades"],
                }
                page.add(genome, result, is_best)
            ranked = sorted(population, key=lambda g: fitness(seen[g]), reverse=True)
            print(
                f"round {round_}: best {best} net {seen[best]['net_cents'] / 100:,.2f} USD",
                flush=True,
            )
            if round_ == args.rounds:
                break
            nxt = ranked[:4]  # keep the 4 best
            while len(nxt) < args.population:
                a = max(rng.sample(population, 3), key=lambda g: fitness(seen[g]))
                b2 = max(rng.sample(population, 3), key=lambda g: fitness(seen[g]))
                c = child(a, b2, rng)
                if c not in nxt:
                    nxt.append(c)
            population = nxt
    search_s = time.time() - t1
    test = simulate(*best, split_ns, end_ns)
    page.status["test"] = (
        f"net {test['net_cents'] / 100:,.2f} USD, drawdown {test['drawdown']:.2%}, "
        f"trades {test['trades']}, win rate {test['wins'] / max(1, test['trades']):.2%}"
    )
    page.write()
    summary = {
        "span": [args.start, args.end],
        "split_ns": split_ns,
        "scenario": args.scenario,
        "payout": "95.2% fixed for every expiry (this test only)",
        "workers": workers,
        "backtests": len(seen),
        "search_seconds": round(search_s, 1),
        "best": {
            "fast": best[0],
            "middle": best[1],
            "slow": best[2],
            "expiry_min": best[3],
        },
        "search": {
            k: v
            for k, v in seen[best].items()
            if k not in ("times", "values", "in_market")
        },
        "held_back_test": {
            k: v for k, v in test.items() if k not in ("times", "values", "in_market")
        },
        "top10": [
            {
                "setting": list(g),
                **{
                    k: v
                    for k, v in seen[g].items()
                    if k not in ("times", "values", "in_market")
                },
            }
            for g in sorted(seen, key=lambda g: fitness(seen[g]), reverse=True)[:10]
        ],
    }
    (folder / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(summary["best"]),
        json.dumps(summary["search"]),
        json.dumps(summary["held_back_test"]),
    )
    print(
        f"{len(seen)} backtests on {workers} workers in {search_s:.1f} s; page stays up 10 min"
    )
    time.sleep(600)


if __name__ == "__main__":
    main()
