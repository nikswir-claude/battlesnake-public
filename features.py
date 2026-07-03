"""Per-move feature extraction, shared by training and the served bot.

Both `training/gen_data.py` and `logic.py` import `candidate_features` from here,
so the vector the model trains on is byte-for-byte the vector it scores at serve
time. `FEATURE_NAMES` fixes column order.
"""

from collections import deque
from typing import Dict, List, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0),
}
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))
_BIG = 10_000
HUNGRY_THRESHOLD = 50


def _in_bounds(p: Point, w: int, h: int) -> bool:
    return 0 <= p[0] < w and 0 <= p[1] < h


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    occ: Set[Point] = set()
    for s in snakes:
        for seg in s["body"]:
            occ.add((seg["x"], seg["y"]))
    return occ


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    danger: Set[Point] = set()
    for s in snakes:
        if s["id"] == my_id or s["length"] < my_length:
            continue
        eh = (s["head"]["x"], s["head"]["y"])
        for dx, dy in _NEIGHBORS:
            danger.add((eh[0] + dx, eh[1] + dy))
    return danger


def _flood_fill(start, occupied, w, h, limit):
    seen = {start}
    stack = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if nb in seen or not _in_bounds(nb, w, h) or nb in occupied:
                continue
            seen.add(nb)
            stack.append(nb)
    return count


def _bfs_dist(sources, blocked, w, h):
    dist = {}
    dq = deque()
    for s in sources:
        if s not in dist:
            dist[s] = 0
            dq.append(s)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if _in_bounds(nb, w, h) and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


# Fixed column order for the model matrix.
FEATURE_NAMES: List[str] = [
    # --- space / trap avoidance ---
    "space_capped", "open_space", "open_space_ratio", "flood_after_ratio",
    "escape", "reaches_tail", "voronoi", "voronoi_ratio",
    # --- combat ---
    "h2h_danger", "near_bigger_head", "near_enemy_head", "near_smaller_head",
    "can_win_h2h", "length_diff_max", "is_longest", "n_enemies",
    # --- food / health ---
    "health", "hungry", "food_score", "food_delta", "nearest_food", "is_food",
    "food_reach_first",
    # --- board geometry ---
    "wall_dist", "on_edge", "in_corner", "dist_to_center",
]


def candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing `move` from `state`. Assumes `move` is legal."""
    board = state["board"]
    you = state["you"]
    w, h = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]
    smaller_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] < my_length]
    max_enemy_len = max((s["length"] for s in enemies), default=0)

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, w, h)
    enemy_dist = _bfs_dist(enemy_heads, occupied, w, h) if enemy_heads else {}
    voronoi = sum(1 for c, md in my_dist.items() if md < enemy_dist.get(c, _BIG))
    free_cells = max(w * h - len(occupied), 1)

    # Tail reachability (anti-self-trap).
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, w, h)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1 for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), w, h)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    space_capped = _flood_fill(nxt, occupied, w, h, limit=my_length + 1)
    open_space = _flood_fill(nxt, occupied, w, h, limit=w * h)

    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD

    # Do we reach the nearest food before any enemy?
    food_reach_first = 0.0
    if foods:
        nf = min(foods, key=lambda f: _manhattan(nxt, f))
        my_fd = my_dist.get(nf, _BIG)
        en_fd = enemy_dist.get(nf, _BIG)
        food_reach_first = 1.0 if my_fd <= en_fd else 0.0

    near_smaller = min((_manhattan(nxt, s) for s in smaller_heads), default=w + h)
    can_win_h2h = 1.0 if near_smaller <= 1 else 0.0

    return {
        "space_capped": float(space_capped),
        "open_space": float(open_space),
        "open_space_ratio": open_space / float(w * h),
        "flood_after_ratio": space_capped / float(my_length + 1),
        "escape": float(escape),
        "reaches_tail": reaches_tail,
        "voronoi": float(voronoi),
        "voronoi_ratio": voronoi / float(free_cells),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, hd) for hd in bigger_heads), default=w + h)),
        "near_enemy_head": float(min((_manhattan(nxt, hd) for hd in enemy_heads), default=w + h)),
        "near_smaller_head": float(near_smaller),
        "can_win_h2h": can_win_h2h,
        "length_diff_max": float(my_length - max_enemy_len),
        "is_longest": 1.0 if my_length > max_enemy_len else 0.0,
        "n_enemies": float(len(enemies)),
        "health": health / 100.0,
        "hungry": 1.0 if hungry else 0.0,
        "food_score": float((w + h - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "nearest_food": float(nearest_next) if foods else float(w + h),
        "is_food": 1.0 if nxt in foods else 0.0,
        "food_reach_first": food_reach_first,
        "wall_dist": float(min(nxt[0], w - 1 - nxt[0], nxt[1], h - 1 - nxt[1])),
        "on_edge": 1.0 if (nxt[0] in (0, w - 1) or nxt[1] in (0, h - 1)) else 0.0,
        "in_corner": 1.0 if (nxt[0] in (0, w - 1) and nxt[1] in (0, h - 1)) else 0.0,
        "dist_to_center": abs(nxt[0] - (w - 1) / 2) + abs(nxt[1] - (h - 1) / 2),
    }


def feature_vector(state: Dict, move: str) -> List[float]:
    f = candidate_features(state, move)
    return [f[name] for name in FEATURE_NAMES]
