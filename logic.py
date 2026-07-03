"""Model-backed move-selection logic for the Battlesnake.

The served policy scores each legal move with a gradient-boosted ranker and
returns the highest-scoring direction. The model is exported to a dependency-free
pure-Python form (``model_py.json``, flattened trees) so serving needs no
lightgbm/numpy — fast to load, light on memory, safe on a small free instance.
A flood-fill heuristic remains as a fallback and as a hard per-move deadline
guard, so a move is always returned well within the game's time limit.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up -> y+1, down -> y-1, left -> x-1, right -> x+1

Feature extraction lives in ``features.py`` and is shared with the training
pipeline, so the vector scored here is identical to the one trained on.
"""

import json
import os
import threading
from typing import Dict, List, Optional

from features import (
    DIRECTIONS,
    FEATURE_NAMES,
    candidate_features,
    _flood_fill,
    _head_to_head_cells,
    _in_bounds,
    _manhattan,
    _occupied_cells,
)

HUNGRY_THRESHOLD = 50
HEAD_TO_HEAD_PENALTY = 10_000
MOVE_BUDGET_S = 0.35  # hard per-move compute budget; fall back to heuristic if exceeded
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_py.json")

# Pure-Python model: {"names": [...], "trees": [[F,T,L,R,V], ...]}. Loaded lazily.
_MODEL = None
_LOAD_FAILED = False


def _get_model():
    """Load and cache the flattened tree model on first use. None on failure."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    try:
        with open(_MODEL_PATH) as f:
            raw = json.load(f)
        _MODEL = {"names": raw["feature_names"], "trees": raw["trees"]}
    except Exception:  # noqa: BLE001 - model load must never break the server
        _LOAD_FAILED = True
        _MODEL = None
    return _MODEL


def warmup() -> bool:
    """Best-effort model preload (call from /start). True if the model is ready."""
    return _get_model() is not None


def _score(fv: List[float], trees) -> float:
    """Sum of leaf values over all trees for one feature vector (pure Python)."""
    total = 0.0
    for F, T, L, R, V in trees:
        node = 0
        while F[node] != -1:
            node = L[node] if fv[F[node]] <= T[node] else R[node]
        total += V[node]
    return total


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.3.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the best move, guaranteed within the per-move budget.

    A fast, always-legal heuristic move is computed first; the model runs under a
    wall-clock budget in a worker thread. If the model finishes in time we use it,
    otherwise we return the heuristic move — so we never blow the game deadline.
    """
    safe = choose_move_heuristic(game_state)
    box: Dict[str, Optional[str]] = {}

    def _run():
        try:
            box["m"] = choose_move_model(game_state)
        except Exception:  # noqa: BLE001 - a model issue must never break gameplay
            box["m"] = None

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(MOVE_BUDGET_S)
    return box.get("m") or safe


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the GBDT; return the best. None if unavailable."""
    model = _get_model()
    if model is None:
        return None
    legal = _legal_moves(game_state)
    if not legal:
        return None
    names, trees = model["names"], model["trees"]
    best_move, best_score = None, float("-inf")
    for move in legal:
        feats = candidate_features(game_state, move)
        fv = [feats[n] for n in names]
        s = _score(fv, trees)
        if s > best_score:
            best_score, best_move = s, move
    return best_move


def choose_move_heuristic(game_state: Dict) -> str:
    """Flood-fill + food fallback. Always returns a legal-ish move."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length, health = you["length"], you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move, best_score = None, float("-inf")
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nxt, width, height) or nxt in occupied:
            continue
        score = float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1))
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2
        if score > best_score:
            best_score, best_move = score, move
    return best_move or "up"


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]
