"""Battlesnake HTTP server for the model inference bot.

Implements the four endpoints the Battlesnake game engine calls:
  GET  /        -> snake appearance + metadata
  POST /start   -> a game has started
  POST /move    -> return our next move for this turn
  POST /end     -> a game has ended
  POST /log     -> (internal) log game state for data collection
"""

import json
import logging
import os
from pathlib import Path

from flask import Flask, request

from logic import choose_move, get_info

app = Flask("battlesnake")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("battlesnake")

# Data collection buffer
_game_data: list = []
_current_game: dict = {}


def _get_data_dir() -> Path:
    return Path(__file__).parent / "experiments" / "data"


def _save_game_data():
    """Append collected game data to a JSONL file."""
    if not _game_data:
        return
    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "games.jsonl"
    with open(path, "a") as f:
        for entry in _game_data:
            f.write(json.dumps(entry) + "\n")
    log.info("💾 Saved %d turns to %s", len(_game_data), path)
    _game_data.clear()


@app.get("/")
def on_info():
    return get_info()


@app.post("/start")
def on_start():
    global _current_game
    game_state = request.get_json()
    _current_game = {
        "game_id": game_state["game"]["id"],
        "turns": [],
    }
    log.info("GAME START %s", game_state["game"]["id"])
    return "ok"


@app.post("/move")
def on_move():
    game_state = request.get_json()
    move = choose_move(game_state)
    log.info("MOVE turn=%s -> %s", game_state["turn"], move)

    # Store turn data for later collection
    _current_game["turns"].append({
        "turn": game_state["turn"],
        "game_state": game_state,
        "move": move,
    })
    _game_data.append({
        "game_id": _current_game["game_id"],
        "turn": game_state["turn"],
        "game_state": game_state,
        "move": move,
    })

    return {"move": move}


@app.post("/end")
def on_end():
    global _current_game
    game_state = request.get_json()
    log.info("GAME END %s", game_state["game"]["id"])
    _save_game_data()
    _current_game = {}
    return "ok"


@app.post("/log")
def on_log():
    """Internal endpoint to force-save collected data."""
    _save_game_data()
    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log.info("Starting Battlesnake server on port %s", port)
    app.run(host="0.0.0.0", port=port)
