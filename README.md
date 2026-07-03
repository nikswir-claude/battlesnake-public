# Battlesnake ML Inference Bot

A [Battlesnake](https://play.battlesnake.com) written in Python and Flask. This
version uses a pretrained model checkpoint to choose moves.

## What It Does

Each turn, `logic.py`:

- Gets legal moves for the current board.
- Calculates per-move features.
- Scores each move with a pure-Python linear model.
- Returns the highest-scoring move.

## Training Pipeline

You can improve the snake by collecting game data and retraining the model:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (run real games)
python experiments/run_games.py --games 50

# 3. Train a new model
python train.py experiments/data/collected_data.parquet

# 4. Update the embedded model
python update_model.py model.json

# 5. Test the improved snake
python backend.py
```

See [`experiments/README.md`](experiments/README.md) for details.

## Files

- `backend.py` — Battlesnake HTTP server with `/`, `/start`, `/move`, `/end`, and `/log`.
- `logic.py` — embedded checkpoint, feature extraction, move scoring, and fallback logic.
- `train.py` — train a new linear model from collected data.
- `update_model.py` — update the embedded model in `logic.py`.
- `experiments/` — data collection and game running scripts.
- `requirements.txt` — runtime dependencies.
- `render.yaml` — Render deployment config.

## Run Locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backend.py
```

Test your battlesnake with the Battlesnake CLI:

```bash
battlesnake play -W 11 -H 11 \
  -n ml -u http://localhost:8000 \
  -g solo \
  -v -c -d 300
```

## Deploy to Render

1. Push this repo to GitHub.
2. In the [Render dashboard](https://dashboard.render.com): **New -> Blueprint**,
   connect the repo. Render reads `render.yaml` and provisions a free web
   service running `gunicorn backend:app`.
   - Or **New -> Web Service** manually with build command
     `pip install -r requirements.txt` and start command
     `gunicorn backend:app --bind 0.0.0.0:$PORT`.
3. Wait for the deploy to go live. Note the public URL, e.g.
   `https://battlesnake-xxxx.onrender.com`.
4. Visit that URL in a browser — you should see the appearance JSON.

## Register on Battlesnake

1. Create an account at [play.battlesnake.com](https://play.battlesnake.com).
2. **Create Battlesnake** -> paste your Render URL as the server URL.
3. Now you can use it in a game!
