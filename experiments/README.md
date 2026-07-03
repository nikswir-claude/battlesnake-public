# Experiments

This directory contains scripts for collecting training data and improving the Battlesnake model.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Collect Training Data

#### Option A: Generate Synthetic Data (for testing)

```bash
python collect_data.py
```

This generates 100 synthetic games and saves them to `experiments/data/`.

#### Option B: Run Real Games

First, start the server:

```bash
python ../backend.py
```

Then in another terminal, run games with the Battlesnake CLI:

```bash
battlesnake play -W 11 -H 11 \
  -n MLBot \
  -u http://localhost:8000 \
  -g solo \
  -o experiments/data/game.jsonl
```

Or use the automated runner:

```bash
python run_games.py --games 50 --port 8000
```

### 3. Train a New Model

```bash
python ../train.py experiments/data/battlesnake_data_*.csv
```

Or with parquet:

```bash
python ../train.py experiments/data/collected_data.parquet
```

This creates `model.json` in the project root.

### 4. Update the Embedded Model

```bash
python ../update_model.py model.json
```

This updates the `_EMBEDDED_MODEL` dict in `logic.py` with the new weights.

### 5. Test the Improved Snake

```bash
python ../backend.py
```

Then in another terminal:

```bash
battlesnake play -W 11 -H 11 \
  -n MLBot \
  -u http://localhost:8000 \
  -g solo \
  -v -c -d 300
```

## Files

- `collect_data.py` — Data collection utilities
- `run_games.py` — Automated game runner using Battlesnake CLI
- `data/` — Collected game data (CSV, Parquet, JSONL)

## Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Run Games  │ -> │ Collect Data │ -> │   Train     │ -> │ Update Model │
│  (CLI)      │    │ (JSONL/CSV)  │    │ (sklearn)   │    │ (logic.py)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```
