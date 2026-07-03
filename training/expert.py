"""Paranoid depth-limited minimax expert — the oracle that labels self-play data.

Much stronger than the served 1-ply model: it searches our move against the
most threatening enemy's response (other snakes move greedily and are frozen in
the search). Leaf eval rewards space control (Voronoi + flood fill), staying
connected to our tail, health, and length advantage.
"""

from collections import deque

from sim import NEIGH, agent_legal_moves, alive_snakes, blocked_cells, clone, get_snake, step

_BIG = 10_000
DEATH = -1_000_000.0


def _bfs(sources, blocked, w, h):
    dist = {}
    dq = deque()
    for s in sources:
        if s not in dist:
            dist[s] = 0
            dq.append(s)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in NEIGH.values():
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < w and 0 <= nb[1] < h and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _closest_enemy(state, my_id):
    me = get_snake(state, my_id)
    if me is None or not me["alive"]:
        return None
    hx, hy = me["body"][0]
    best, bestd = None, _BIG
    for s in state["snakes"]:
        if not s["alive"] or s["id"] == my_id:
            continue
        ex, ey = s["body"][0]
        d = abs(ex - hx) + abs(ey - hy)
        if d < bestd:
            best, bestd = s["id"], d
    return best


def greedy_move(state, sid):
    """Cheap 1-ply move: maximise open neighbours, break ties toward nearest food."""
    s = get_snake(state, sid)
    if s is None or not s["alive"]:
        return "up"
    legal = agent_legal_moves(state, sid)
    if not legal:
        return "up"
    hx, hy = s["body"][0]
    blocked = blocked_cells(state)
    w, h = state["w"], state["h"]
    foods = state["food"]
    best, bestscore = legal[0], -1e18
    for m in legal:
        dx, dy = NEIGH[m]
        nx, ny = hx + dx, hy + dy
        openn = sum(
            1 for ddx, ddy in NEIGH.values()
            if 0 <= nx + ddx < w and 0 <= ny + ddy < h and (nx + ddx, ny + ddy) not in blocked
        )
        score = openn * 10.0
        if foods:
            score -= min(abs(nx - fx) + abs(ny - fy) for (fx, fy) in foods)
        if score > bestscore:
            best, bestscore = m, score
    return best


def evaluate(state, my_id):
    me = get_snake(state, my_id)
    if me is None or not me["alive"]:
        return DEATH
    w, h = state["w"], state["h"]
    blocked = blocked_cells(state)
    head = me["body"][0]

    enemies = [s for s in state["snakes"] if s["alive"] and s["id"] != my_id]
    my_dist = _bfs([head], blocked - {me["body"][-1]}, w, h)
    enemy_heads = [s["body"][0] for s in enemies]
    enemy_dist = _bfs(enemy_heads, blocked, w, h) if enemy_heads else {}
    voronoi = sum(1 for c, md in my_dist.items() if md < enemy_dist.get(c, _BIG))

    space = len(my_dist)
    reaches_tail = 1.0 if me["body"][-1] in my_dist else 0.0
    my_len = len(me["body"])
    max_en = max((len(s["body"]) for s in enemies), default=0)

    score = 0.0
    score += 6.0 * voronoi
    score += 1.0 * space
    score += 40.0 * reaches_tail
    score += 12.0 * (my_len - max_en)   # growing / staying longest matters a lot
    score += 0.2 * me["health"]

    # Food gradient so trajectories actually eat and grow (diversifies states).
    if state["food"]:
        fd = min(abs(head[0] - fx) + abs(head[1] - fy) for (fx, fy) in state["food"])
        hunger_w = 3.0 if me["health"] < 40 else (1.0 if my_len <= max_en else 0.3)
        score -= hunger_w * fd
    if not enemies:  # solo survival: just keep space and health
        score += 2.0 * space
    return score


def _search(state, my_id, depth):
    me = get_snake(state, my_id)
    if me is None or not me["alive"]:
        return DEATH - depth  # dying sooner is worse
    if depth == 0 or len(alive_snakes(state)) <= 1:
        return evaluate(state, my_id)

    threat = _closest_enemy(state, my_id)
    others = [s["id"] for s in alive_snakes(state) if s["id"] != my_id and s["id"] != threat]
    my_moves = agent_legal_moves(state, my_id) or ["up"]

    best = -1e18
    for m in my_moves:
        t_moves = (agent_legal_moves(state, threat) or ["up"]) if threat else [None]
        worst = 1e18
        for tm in t_moves:
            joint = {my_id: m}
            if threat:
                joint[threat] = tm
            for o in others:
                joint[o] = greedy_move(state, o)
            ns = step(clone(state), joint)
            worst = min(worst, _search(ns, my_id, depth - 1))
            if worst <= best:  # alpha prune
                break
        best = max(best, worst)
    return best


def expert_move(state, my_id, depth=3):
    """Return (best_move, ranked_scores_dict) for `my_id`."""
    my_moves = agent_legal_moves(state, my_id)
    if not my_moves:
        return None, {}
    threat = _closest_enemy(state, my_id)
    others = [s["id"] for s in alive_snakes(state) if s["id"] != my_id and s["id"] != threat]

    scores = {}
    for m in my_moves:
        t_moves = (agent_legal_moves(state, threat) or ["up"]) if threat else [None]
        worst = 1e18
        for tm in t_moves:
            joint = {my_id: m}
            if threat:
                joint[threat] = tm
            for o in others:
                joint[o] = greedy_move(state, o)
            ns = step(clone(state), joint)
            worst = min(worst, _search(ns, my_id, depth - 1))
        scores[m] = worst
    best = max(scores, key=scores.get)
    return best, scores
