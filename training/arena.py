"""Head-to-head arena: new GBDT bot vs old linear model vs heuristic.

Plays many games in the simulator with randomized, symmetric starts and reports
win rates. This is the bottom-line hackathon metric — does the new policy
actually beat the old one over the board, not just match a label.

    python arena.py --games 200
"""

import argparse
import os
import random
import sys
from functools import partial
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features  # noqa: E402
import sim        # noqa: E402
import expert     # noqa: E402
from train import _LINEAR  # noqa: E402

_LIN_IDX = None


def _api_to_sim(api):
    """Rebuild an internal sim state from an api game_state (for the minimax bot)."""
    board = api["board"]
    snakes = []
    for s in board["snakes"]:
        body = [(p["x"], p["y"]) for p in s["body"]]
        snakes.append({"id": s["id"], "health": s["health"], "body": body,
                       "alive": True, "ate": False})
    food = [(f["x"], f["y"]) for f in board["food"]]
    return sim.make_state(board["width"], board["height"], snakes, food, api.get("turn", 0))


def make_minimax_bot(depth):
    """A real search bot (our teacher) as an arena opponent."""
    def bot(api):
        state = _api_to_sim(api)
        mv, _ = expert.expert_move(state, api["you"]["id"], depth=depth)
        return mv or "up"
    return bot


def make_gbdt_bot(model_path):
    booster = lgb.Booster(model_file=model_path)
    names = features.FEATURE_NAMES

    def bot(api):
        legal = _legal(api)
        if not legal:
            return "up"
        M = np.array([[features.candidate_features(api, m)[n] for n in names] for m in legal],
                     dtype=np.float32)
        scores = booster.predict(M)
        return legal[int(np.argmax(scores))]
    return bot


def linear_bot(api):
    legal = _legal(api)
    if not legal:
        return "up"
    m = _LINEAR
    best, bs = legal[0], -1e18
    for mv in legal:
        f = features.candidate_features(api, mv)
        z = sum(m["coef"][i] * (f[n] - m["mean"][i]) / m["std"][i]
                for i, n in enumerate(m["feature_names"]))
        if z > bs:
            bs, best = z, mv
    return best


def heuristic_bot(api):
    """Flood-fill + food heuristic (the original fallback), reimplemented on api state."""
    legal = _legal(api)
    if not legal:
        return "up"
    you = api["you"]
    board = api["board"]
    w, h = board["width"], board["height"]
    my_len = you["length"]
    best, bs = legal[0], -1e18
    for mv in legal:
        f = features.candidate_features(api, mv)
        score = f["open_space"] - 10000 * f["h2h_danger"]
        if you["health"] < 50:
            score += (w + h - f["nearest_food"]) * 2
        if score > bs:
            bs, best = score, mv
    return best


def _legal(api):
    board = api["board"]
    w, h = board["width"], board["height"]
    head = (api["you"]["head"]["x"], api["you"]["head"]["y"])
    occ = {(s["x"], s["y"]) for sn in board["snakes"] for s in sn["body"]}
    out = []
    for m, (dx, dy) in features.DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy
        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in occ:
            out.append(m)
    return out


def play(bot_a, bot_b, seed, w=11, h=11):
    rng = random.Random(seed)
    cells = [(x, y) for x in range(1, w - 1) for y in range(1, h - 1)]
    rng.shuffle(cells)
    a0, b0 = cells[0], next(c for c in cells if abs(c[0] - cells[0][0]) + abs(c[1] - cells[0][1]) >= 4)
    snakes = [sim.new_snake("A", [a0] * 3), sim.new_snake("B", [b0] * 3)]
    food = {(w // 2, h // 2)}
    empties = [c for c in cells if c not in (a0, b0)]
    rng.shuffle(empties)
    food.update(empties[:3])
    state = sim.make_state(w, h, snakes, list(food))
    bots = {"A": bot_a, "B": bot_b}
    while state["turn"] < 300 and len(sim.alive_snakes(state)) > 1:
        moves = {}
        for s in sim.alive_snakes(state):
            moves[s["id"]] = bots[s["id"]](sim.to_api(state, s["id"]))
        state = sim.step(state, moves)
    alive = sim.alive_snakes(state)
    if len(alive) == 1:
        return alive[0]["id"]
    return "draw"


# Bots live in module globals so worker processes build them once (booster load).
_BOTS = {}


def _init_worker(model_path):
    _BOTS["GBDT"] = make_gbdt_bot(model_path)
    _BOTS["linear"] = linear_bot
    _BOTS["heuristic"] = heuristic_bot
    _BOTS["minimax2"] = make_minimax_bot(2)
    _BOTS["minimax3"] = make_minimax_bot(3)


def _play_one(task):
    name_a, name_b, seed = task
    swap = seed % 2 == 1  # alternate seats for fairness
    ba, bb = (_BOTS[name_b], _BOTS[name_a]) if swap else (_BOTS[name_a], _BOTS[name_b])
    r = play(ba, bb, seed)
    if r == "draw":
        return "draw"
    winner_seat_is_a = (r == "A")
    if swap:
        return name_b if winner_seat_is_a else name_a
    return name_a if winner_seat_is_a else name_b


def match(name_a, name_b, games, pool):
    tasks = [(name_a, name_b, g) for g in range(games)]
    wins = {name_a: 0, name_b: 0, "draw": 0}
    for w in pool.imap_unordered(_play_one, tasks, chunksize=4):
        wins[w] += 1
    decisive = wins[name_a] + wins[name_b]
    wr = wins[name_a] / decisive if decisive else 0.5
    print(f"  {name_a:9s} vs {name_b:9s}: {wins[name_a]}-{wins[name_b]}-{wins['draw']} "
          f"(win rate {name_a}: {100 * wr:.1f}% of decisive)")
    return wr


def round_robin(bots, games, pool):
    """Every bot vs every other; print a ranking by overall win rate."""
    import itertools
    wr_sum = {b: 0.0 for b in bots}
    n = {b: 0 for b in bots}
    for a, b in itertools.combinations(bots, 2):
        wr = match(a, b, games, pool)
        wr_sum[a] += wr
        wr_sum[b] += 1 - wr
        n[a] += 1
        n[b] += 1
    print("\n=== RANKING (avg win rate across opponents) ===")
    ranking = sorted(bots, key=lambda x: -wr_sum[x] / max(n[x], 1))
    for i, b in enumerate(ranking, 1):
        print(f"  {i}. {b:10s} {100 * wr_sum[b] / max(n[b], 1):5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.txt"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--minimax", action="store_true", help="benchmark vs the minimax search bot")
    ap.add_argument("--roundrobin", action="store_true", help="full round-robin tournament + ranking")
    ap.add_argument("--bots", default="heuristic,linear,GBDT,minimax2,minimax3")
    args = ap.parse_args()
    print(f"[arena] {args.games} games per matchup, {args.workers} workers\n")
    with Pool(args.workers, initializer=_init_worker, initargs=(args.model,)) as pool:
        if args.roundrobin:
            round_robin([b.strip() for b in args.bots.split(",")], args.games, pool)
        elif args.minimax:
            match("GBDT", "minimax2", args.games, pool)
            match("GBDT", "minimax3", args.games, pool)
            match("linear", "minimax2", args.games, pool)
        else:
            match("GBDT", "linear", args.games, pool)
            match("GBDT", "heuristic", args.games, pool)
            match("linear", "heuristic", args.games, pool)


if __name__ == "__main__":
    main()
