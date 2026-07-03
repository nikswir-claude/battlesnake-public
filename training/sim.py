"""Minimal Battlesnake standard-ruleset simulator for self-play data generation.

Fast, dependency-free engine used both by the minimax expert (`expert.py`) and
the self-play data generator (`gen_data.py`). Internal state is a plain dict so
it clones cheaply; `to_api` renders the exact JSON the Battlesnake engine POSTs
to `/move`, so features computed during training match features at serve time.
"""

NEIGH = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
MOVES = ("up", "down", "left", "right")


def new_snake(sid, body, health=100):
    return {"id": sid, "health": health, "body": [tuple(p) for p in body], "alive": True, "ate": False}


def make_state(w, h, snakes, food, turn=0):
    return {"w": w, "h": h, "snakes": snakes, "food": set(map(tuple, food)), "turn": turn}


def clone(state):
    return {
        "w": state["w"], "h": state["h"], "turn": state["turn"],
        "food": set(state["food"]),
        "snakes": [
            {"id": s["id"], "health": s["health"], "body": list(s["body"]),
             "alive": s["alive"], "ate": s["ate"]}
            for s in state["snakes"]
        ],
    }


def get_snake(state, sid):
    for s in state["snakes"]:
        if s["id"] == sid:
            return s
    return None


def alive_snakes(state):
    return [s for s in state["snakes"] if s["alive"]]


def blocked_cells(state):
    """Body cells that will still be solid next turn (tails of non-eating snakes free up)."""
    occ = set()
    for s in state["snakes"]:
        if not s["alive"]:
            continue
        body = s["body"]
        segs = body if s["ate"] else body[:-1]
        occ.update(segs)
    return occ


def all_body_cells(state):
    occ = set()
    for s in state["snakes"]:
        if s["alive"]:
            occ.update(s["body"])
    return occ


def agent_legal_moves(state, sid):
    """Moves that don't step off-board or into a cell that stays solid."""
    s = get_snake(state, sid)
    if s is None or not s["alive"]:
        return []
    hx, hy = s["body"][0]
    blocked = blocked_cells(state)
    w, h = state["w"], state["h"]
    res = []
    for m, (dx, dy) in NEIGH.items():
        nx, ny = hx + dx, hy + dy
        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in blocked:
            res.append(m)
    return res


def step(state, moves):
    """Advance one turn (mutates and returns `state`). `moves`: {sid: move_name}."""
    w, h = state["w"], state["h"]

    # 1. Move every living snake's head; drop health.
    for s in state["snakes"]:
        if not s["alive"]:
            continue
        m = moves.get(s["id"]) or "up"
        dx, dy = NEIGH[m]
        hx, hy = s["body"][0]
        s["body"] = [(hx + dx, hy + dy)] + s["body"]
        s["health"] -= 1
        s["ate"] = False

    # 2. Feeding: eat food (stay full, keep tail) or shrink.
    eaten = set()
    for s in state["snakes"]:
        if not s["alive"]:
            continue
        head = s["body"][0]
        if head in state["food"]:
            s["health"] = 100
            s["ate"] = True
            eaten.add(head)
        else:
            s["body"].pop()
    state["food"] -= eaten

    # 3. Eliminations: starvation + out-of-bounds first.
    dead = set()
    for s in state["snakes"]:
        if not s["alive"]:
            continue
        hx, hy = s["body"][0]
        if s["health"] <= 0 or not (0 <= hx < w and 0 <= hy < h):
            dead.add(s["id"])

    # 4. Collisions, resolved against all bodies simultaneously.
    for s in state["snakes"]:
        if not s["alive"] or s["id"] in dead:
            continue
        head = s["body"][0]
        if head in s["body"][1:]:
            dead.add(s["id"])
            continue
        for o in state["snakes"]:
            if not o["alive"] or o["id"] == s["id"]:
                continue
            if head in o["body"][1:]:
                dead.add(s["id"])
                break
            if head == o["body"][0] and len(s["body"]) <= len(o["body"]):
                dead.add(s["id"])
                break

    for s in state["snakes"]:
        if s["id"] in dead:
            s["alive"] = False

    state["turn"] += 1
    return state


def to_api(state, my_id):
    """Render the game_state JSON the engine sends, from `my_id`'s perspective."""
    snakes_api, you = [], None
    for s in state["snakes"]:
        if not s["alive"]:
            continue
        body = [{"x": x, "y": y} for (x, y) in s["body"]]
        snake = {
            "id": s["id"], "name": s["id"], "health": s["health"],
            "body": body, "head": body[0], "length": len(body),
            "latency": "0", "shout": "",
        }
        snakes_api.append(snake)
        if s["id"] == my_id:
            you = snake
    return {
        "turn": state["turn"],
        "board": {
            "width": state["w"], "height": state["h"],
            "food": [{"x": x, "y": y} for (x, y) in state["food"]],
            "hazards": [], "snakes": snakes_api,
        },
        "you": you,
    }
