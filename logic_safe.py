"""Duel-safe candidate: model scoring with a hard head-to-head filter.

Baseline ``logic.py`` only *penalizes* head-to-head cells (a soft cost the
Voronoi term can outweigh), which makes two equal-length snakes walk into the
same contested cell and both die (mutual head-to-head). This module reuses the
exact same model, but *forbids* moves onto losing/tying head-to-head cells
whenever a safe alternative exists.

Not used in production by default; select it via ``LOGIC_MODULE=logic_safe``.
Baseline ``logic.py`` stays untouched so it remains the safe fallback.
"""

from typing import Dict, List, Optional

import logic
from logic import get_info  # noqa: F401  (re-export appearance for backend.py)


def _safe_legal(game_state: Dict) -> List[str]:
    """Legal moves whose target cell is NOT a losing/tying head-to-head cell."""
    board = game_state["board"]
    you = game_state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    # Cells adjacent to enemy heads of length >= ours: an h2h there we lose/tie.
    danger = logic._head_to_head_cells(board["snakes"], you["id"], you["length"])
    out: List[str] = []
    for move in logic._legal_moves(game_state):
        dx, dy = logic.DIRECTIONS[move]
        if (head[0] + dx, head[1] + dy) not in danger:
            out.append(move)
    return out


def _best_of(game_state: Dict, moves: List[str]) -> Optional[str]:
    """Highest model-scoring move among ``moves`` (reuses logic._MODEL)."""
    if not moves:
        return None
    model = logic._MODEL
    names = model["feature_names"]
    mean = model["mean"]
    std = model["std"]
    coef = model["coef"]
    best_move, best_score = None, float("-inf")
    for move in moves:
        feats = logic._candidate_features(game_state, move)
        score = model["intercept"]
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def choose_move(game_state: Dict) -> str:
    """Prefer safe moves; fall back to any legal move, then to the heuristic."""
    try:
        move = _best_of(game_state, _safe_legal(game_state))
        if move is None:  # cornered: every legal move risks a lost h2h
            move = _best_of(game_state, logic._legal_moves(game_state))
        if move is not None:
            return move
    except Exception:  # noqa: BLE001 - a scoring issue must never break gameplay
        pass
    return logic.choose_move_heuristic(game_state)
