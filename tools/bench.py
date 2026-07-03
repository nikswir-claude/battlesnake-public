"""Measurement harness for the Battlesnake bot.

Runs batches of games through the Battlesnake CLI against one or more local
server instances, parses the results, and reports aggregate metrics
(winrate, turns survived, final length) so logic changes can be evaluated
objectively.

Typical uses
------------
Baseline of the served bot (solo survival + mirror duel):
    python tools/bench.py baseline --games 20

A/B two code variants (e.g. old vs new heuristic) in a duel:
    # server A serves current code, server B serves candidate code
    python tools/bench.py duel --games 30 \
        --snake A=http://localhost:8100 --snake B=http://localhost:8101

Notes
-----
* This script spawns its own ``backend.py`` server(s) via ``python backend.py``
  with a per-instance ``PORT`` and a per-instance ``LOGIC_MODULE`` env var, so
  you can point two instances at two different logic modules for A/B testing
  (see ``--spawn``). If you pass ``--snake NAME=URL`` for an already-running
  server, it is used as-is and not spawned.
* Metrics parsed reliably: turns played, winner / draw, per-snake survival and
  final length. Death-cause is intentionally omitted (the CLI does not expose
  the fatal move in a machine-readable frame); survival + winrate are enough to
  compare variants.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CLI = HERE / ("battlesnake.exe" if os.name == "nt" else "battlesnake")

_TURNS_RE = re.compile(r"Game completed after (\d+) turns")


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
@dataclass
class Server:
    """A spawned ``backend.py`` instance."""

    port: int
    logic_module: str = "logic"
    proc: Optional[subprocess.Popen] = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self) -> None:
        env = dict(os.environ, PORT=str(self.port), LOGIC_MODULE=self.logic_module)
        self.proc = subprocess.Popen(
            [sys.executable, "backend.py"],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def wait_healthy(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


# --------------------------------------------------------------------------- #
# Game execution + parsing
# --------------------------------------------------------------------------- #
@dataclass
class GameResult:
    turns: int
    winner: Optional[str]        # snake NAME, or None for draw / solo death
    is_draw: bool
    final_length: Dict[str, int] = field(default_factory=dict)
    survived: Dict[str, bool] = field(default_factory=dict)


def run_game(
    snakes: List[Tuple[str, str]],
    *,
    width: int,
    height: int,
    gametype: str,
    seed: int,
    timeout_ms: int,
    tmp_out: Path,
) -> GameResult:
    """Play one game and parse its result. ``snakes`` is a list of (name, url)."""
    cmd = [str(CLI), "play", "-W", str(width), "-H", str(height),
           "-g", gametype, "-r", str(seed), "-t", str(timeout_ms),
           "-o", str(tmp_out)]
    for name, url in snakes:
        cmd += ["-n", name, "-u", url]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    stderr = completed.stderr or ""

    m = _TURNS_RE.search(stderr)
    turns = int(m.group(1)) if m else -1

    winner: Optional[str] = None
    is_draw = False
    final_length: Dict[str, int] = {}
    survived: Dict[str, bool] = {name: False for name, _ in snakes}

    # Parse the output frames: last JSON line is the winner summary; the rest are
    # board states. We use the last board state to read final lengths, and track
    # which snakes are still present in the final board frame (survivors).
    last_summary: Optional[dict] = None
    last_board: Optional[dict] = None
    if tmp_out.exists():
        for line in tmp_out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "winnerId" in obj or "winnerName" in obj:
                last_summary = obj
            elif "board" in obj:
                last_board = obj

    if last_summary is not None:
        is_draw = bool(last_summary.get("isDraw", False))
        wname = last_summary.get("winnerName") or ""
        winner = wname or None

    if last_board is not None:
        for sn in last_board["board"]["snakes"]:
            final_length[sn["name"]] = sn["length"]
            survived[sn["name"]] = True

    return GameResult(
        turns=turns,
        winner=winner,
        is_draw=is_draw,
        final_length=final_length,
        survived=survived,
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarize(label: str, snake_names: List[str], results: List[GameResult]) -> dict:
    n = len(results)
    turns = [r.turns for r in results if r.turns >= 0]
    per_snake = {}
    for name in snake_names:
        wins = sum(1 for r in results if r.winner == name)
        draws = sum(1 for r in results if r.is_draw)
        finals = [r.final_length[name] for r in results if name in r.final_length]
        per_snake[name] = {
            "wins": wins,
            "winrate": round(wins / n, 4) if n else 0.0,
            "draws": draws,
            "avg_final_length": round(statistics.mean(finals), 2) if finals else 0.0,
        }
    summary = {
        "label": label,
        "games": n,
        "turns_mean": round(statistics.mean(turns), 1) if turns else 0.0,
        "turns_median": statistics.median(turns) if turns else 0,
        "turns_min": min(turns) if turns else 0,
        "turns_max": max(turns) if turns else 0,
        "draws": sum(1 for r in results if r.is_draw),
        "per_snake": per_snake,
    }
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['label']} ({summary['games']} games) ===")
    print(f"turns: mean={summary['turns_mean']} median={summary['turns_median']} "
          f"min={summary['turns_min']} max={summary['turns_max']}  draws={summary['draws']}")
    for name, s in summary["per_snake"].items():
        print(f"  {name:12s} winrate={s['winrate']:.3f} "
              f"wins={s['wins']} avg_len={s['avg_final_length']}")


# --------------------------------------------------------------------------- #
# Batch runners
# --------------------------------------------------------------------------- #
def run_batch(
    label: str,
    snakes: List[Tuple[str, str]],
    *,
    games: int,
    width: int,
    height: int,
    gametype: str,
    seed_start: int,
    timeout_ms: int,
) -> dict:
    tmp_out = HERE / f"_bench_{gametype}.jsonl"
    results: List[GameResult] = []
    names = [n for n, _ in snakes]
    for i in range(games):
        res = run_game(
            snakes, width=width, height=height, gametype=gametype,
            seed=seed_start + i, timeout_ms=timeout_ms, tmp_out=tmp_out,
        )
        results.append(res)
        print(f"[{label}] game {i + 1}/{games}: turns={res.turns} "
              f"winner={res.winner or ('draw' if res.is_draw else '-')}")
    tmp_out.unlink(missing_ok=True)
    summary = summarize(label, names, results)
    print_summary(summary)
    return summary


def parse_snake_arg(items: List[str]) -> List[Tuple[str, str]]:
    """Parse ``NAME=URL`` items into (name, url) tuples."""
    out = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--snake expects NAME=URL, got: {it}")
        name, url = it.split("=", 1)
        out.append((name, url))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Battlesnake measurement harness")
    p.add_argument("mode", choices=["baseline", "solo", "duel"])
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--width", type=int, default=11)
    p.add_argument("--height", type=int, default=11)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--timeout-ms", type=int, default=500)
    p.add_argument("--out", type=str, default=str(HERE / "metrics_baseline.json"))
    p.add_argument("--snake", action="append", default=[],
                   help="NAME=URL of an already-running server (repeatable). "
                        "If omitted, servers are spawned automatically.")
    p.add_argument("--logic", action="append", default=[],
                   help="LOGIC_MODULE name(s) to spawn for auto mode "
                        "(default: logic). Repeat for A/B, e.g. --logic logic "
                        "--logic logic_tailaware.")
    args = p.parse_args()

    if not CLI.exists():
        raise SystemExit(f"battlesnake CLI not found at {CLI}")

    spawned: List[Server] = []
    summaries: List[dict] = []
    try:
        if args.snake:
            snakes = parse_snake_arg(args.snake)
        else:
            # Spawn servers. For solo/baseline: 1 instance. For duel: 2 instances.
            logics = args.logic or ["logic"]
            n_instances = 1 if args.mode == "solo" else 2
            base_port = 8100
            urls: List[str] = []
            for i in range(n_instances):
                mod = logics[i] if i < len(logics) else logics[-1]
                srv = Server(port=base_port + i, logic_module=mod)
                srv.start()
                spawned.append(srv)
                urls.append(srv.url)
            for srv in spawned:
                if not wait_healthy(srv.url):
                    raise SystemExit(f"server on {srv.url} did not become healthy")
            if args.mode == "solo":
                snakes = [("bot", urls[0])]
            else:
                snakes = [("A", urls[0]), ("B", urls[1])]

        common = dict(games=args.games, width=args.width, height=args.height,
                      seed_start=args.seed_start, timeout_ms=args.timeout_ms)

        if args.mode in ("baseline", "solo"):
            solo_snakes = [snakes[0]] if len(snakes) > 1 else snakes
            summaries.append(run_batch("solo", solo_snakes, gametype="solo", **common))
        if args.mode in ("baseline", "duel"):
            duel_snakes = snakes if len(snakes) >= 2 else None
            if duel_snakes is None:
                print("\n[duel] skipped: need 2 snakes (pass --snake twice or use "
                      "auto-spawn without --snake).")
            else:
                summaries.append(run_batch("duel", duel_snakes, gametype="standard", **common))
    finally:
        for srv in spawned:
            srv.stop()

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote metrics to {out_path}")


if __name__ == "__main__":
    main()
