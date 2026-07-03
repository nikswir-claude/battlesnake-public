"""Self-play data generator: minimax expert labels, shared features as X.

Runs many self-play games across all CPU cores until a wall-clock budget is hit.
Each turn, for every living snake, the depth-limited minimax expert picks the
best move (the label); every legal move becomes one training row with the
shared `features.candidate_features` vector. Epsilon-greedy play diversifies the
visited states while labels always come from the expert.

    python gen_data.py --minutes 15 --out data.npz
"""

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features  # noqa: E402  (shared with logic.py)
import sim        # noqa: E402
import expert     # noqa: E402

EPS = 0.15          # exploration rate
MAX_TURNS = 250

# (label, width, height, n_snakes, weight)
CONFIGS = [
    ("1v1", 11, 11, 2, 5),
    ("4p", 11, 11, 4, 3),
    ("3p", 11, 11, 3, 2),
    ("solo", 11, 11, 1, 1),
    ("small", 7, 7, 2, 2),
]
_WEIGHTED = [c for c in CONFIGS for _ in range(c[4])]


def _random_start(w, h, n, rng):
    cells = [(x, y) for x in range(1, w - 1) for y in range(1, h - 1)]
    rng.shuffle(cells)
    snakes, used = [], set()
    for i in range(n):
        # pick a start cell far from already-placed snakes
        spot = None
        for c in cells:
            if c in used:
                continue
            if all(abs(c[0] - u[0]) + abs(c[1] - u[1]) >= 3 for u in used):
                spot = c
                break
        if spot is None:
            spot = cells[i]
        used.add(spot)
        snakes.append(sim.new_snake(f"s{i}", [spot] * 3))
    # food: a few random empties + center
    food = {(w // 2, h // 2)}
    empties = [c for c in cells if c not in used]
    rng.shuffle(empties)
    food.update(empties[: max(2, n)])
    return sim.make_state(w, h, snakes, list(food))


def play_game(seed):
    rng = random.Random(seed)
    label, w, h, n, _ = rng.choice(_WEIGHTED)
    state = _random_start(w, h, n, rng)
    depth = 3 if n <= 2 else 2

    rows_X, rows_y, groups = [], [], []
    gid = 0
    while state["turn"] < MAX_TURNS and len(sim.alive_snakes(state)) >= (1 if n == 1 else 2):
        chosen = {}
        for s in sim.alive_snakes(state):
            sid = s["id"]
            legal = sim.agent_legal_moves(state, sid)
            if not legal:
                chosen[sid] = "up"
                continue
            best, _ = expert.expert_move(state, sid, depth=depth)
            if best is None:
                chosen[sid] = "up"
                continue
            api = sim.to_api(state, sid)
            g = seed * 100_000 + gid  # globally unique per (game, state)
            for m in legal:
                rows_X.append(features.feature_vector(api, m))
                rows_y.append(1 if m == best else 0)
                groups.append(g)
            gid += 1
            chosen[sid] = rng.choice(legal) if rng.random() < EPS else best
        state = sim.step(state, chosen)
    return rows_X, rows_y, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data.npz"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    budget = args.minutes * 60.0
    t0 = time.time()
    X, y, G = [], [], []
    games = 0

    print(f"[gen] budget={args.minutes}min workers={args.workers} features={len(features.FEATURE_NAMES)}")
    seeds = iter(range(10_000_000))
    with Pool(args.workers) as pool:
        for gx, gy, gg in pool.imap_unordered(play_game, seeds, chunksize=1):
            X.extend(gx)
            y.extend(gy)
            G.extend(gg)
            games += 1
            if games % 20 == 0:
                el = time.time() - t0
                print(f"[gen] {el:5.0f}s  games={games:5d}  rows={len(y):8d}  "
                      f"pos_rate={np.mean(y):.3f}  rows/s={len(y)/el:.0f}", flush=True)
            if time.time() - t0 > budget:
                pool.terminate()
                break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    G = np.asarray(G, dtype=np.int64)
    np.savez_compressed(args.out, X=X, y=y, groups=G, feature_names=np.array(features.FEATURE_NAMES))
    print(f"[gen] DONE games={games} rows={len(y)} states={len(np.unique(G))} "
          f"pos_rate={y.mean():.3f} shape={X.shape} elapsed={time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
