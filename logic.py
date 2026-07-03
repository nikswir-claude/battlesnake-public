"""Model-backed move-selection logic for the Battlesnake.

The served policy scores each legal move with a gradient-boosted ranker
(LightGBM, trained by imitation of a depth-limited minimax expert) and returns
the highest-scoring direction. A compact flood-fill heuristic remains as a
fallback so gameplay always returns a legal move if the model is unavailable.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up -> y+1, down -> y-1, left -> x-1, right -> x+1

Feature extraction lives in ``features.py`` and is shared with the training
pipeline, so the vector scored here is identical to the one trained on.
"""

import os
from typing import Dict, List, Optional

import numpy as np

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
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt")

# Lazily-loaded LightGBM booster (None if unavailable -> heuristic fallback).
_BOOSTER = None
try:
    import lightgbm as _lgb

    if os.path.exists(_MODEL_PATH):
        _BOOSTER = _lgb.Booster(model_file=_MODEL_PATH)
except Exception:  # noqa: BLE001 - model load must never break the server
    _BOOSTER = None


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.2.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the model, with a heuristic fallback."""
    try:
        move = choose_move_model(game_state)
    except Exception:  # noqa: BLE001 - a model issue must never break gameplay
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the GBDT; return the best. None if unavailable."""
    if _BOOSTER is None:
        return None
    legal = _legal_moves(game_state)
    if not legal:
        return None
    matrix = np.array(
        [[candidate_features(game_state, m)[n] for n in FEATURE_NAMES] for m in legal],
        dtype=np.float32,
    )
    scores = _BOOSTER.predict(matrix)
    return legal[int(np.argmax(scores))]


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
