"""Run Battlesnake games locally and collect training data.

This script:
1. Starts the Flask server in a subprocess
2. Runs N games using the Battlesnake CLI
3. Collects game data from the server's JSONL log
4. Processes the data into training format

Usage:
    python experiments/run_games.py --games 50 --port 8000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from logic import _candidate_features, _legal_moves, DIRECTIONS, choose_move_heuristic


def process_game_log(log_path: Path) -> list:
    """Process a game JSONL file into training rows."""
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            game_state = entry["game_state"]
            chosen_move = entry["move"]

            legal_moves = _legal_moves(game_state)
            features_by_move = {}
            for m in legal_moves:
                features = _candidate_features(game_state, m)
                features_by_move[m] = features

            you = game_state["you"]
            board = game_state["board"]

            row = {
                "game_id": entry["game_id"],
                "turn": entry["turn"],
                "move": chosen_move,
                "chosen": True,
                "reward": 0.0,  # Will be computed later
                "health": you["health"],
                "length": you["length"],
                "food_count": len(board["food"]),
                "enemy_count": len([s for s in board["snakes"] if s["id"] != you["id"]]),
            }
            for k, v in features_by_move[chosen_move].items():
                row[f"feat_{k}"] = v

            rows.append(row)

            # Also add non-chosen moves as negative examples
            for m in legal_moves:
                if m == chosen_move:
                    continue
                row_neg = {
                    "game_id": entry["game_id"],
                    "turn": entry["turn"],
                    "move": m,
                    "chosen": False,
                    "reward": 0.0,
                    "health": you["health"],
                    "length": you["length"],
                    "food_count": len(board["food"]),
                    "enemy_count": len([s for s in board["snakes"] if s["id"] != you["id"]]),
                }
                for k, v in features_by_move[m].items():
                    row_neg[f"feat_{k}"] = v
                rows.append(row_neg)

    return rows


def run_games(n_games: int, port: int, width: int = 11, height: int = 11, seed: int = None):
    """Run N games and collect data."""
    # Start the Flask server
    server_proc = subprocess.Popen(
        [sys.executable, "backend.py"],
        env={**__import__("os").environ, "PORT": str(port)},
        cwd=Path(__file__).parent.parent,
    )
    print(f"🚀 Started server on port {port} (PID {server_proc.pid})")

    try:
        # Wait for server to start
        time.sleep(3)

        # Check if battlesnake CLI is available
        try:
            subprocess.run(["battlesnake", "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Battlesnake CLI not found. Install it first:")
            print("   go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest")
            print("   Or download from: https://github.com/BattlesnakeOfficial/rules/releases")
            return []

        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for i in range(n_games):
            output_file = data_dir / f"game_{i:04d}.jsonl"
            seed_val = seed + i if seed else None

            cmd = [
                "battlesnake", "play",
                "-W", str(width),
                "-H", str(height),
                "-n", "MLBot",
                "-u", f"http://localhost:{port}",
                "-g", "solo",
                "-o", str(output_file),
            ]
            if seed_val is not None:
                cmd.extend(["-r", str(seed_val)])

            print(f"🎮 Game {i+1}/{n_games}...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ⚠️  Game failed: {result.stderr[:200]}")
                continue

            if output_file.exists():
                rows = process_game_log(output_file)
                all_rows.extend(rows)
                print(f"  ✅ Collected {len(rows)} rows")

        return all_rows

    finally:
        server_proc.terminate()
        server_proc.wait()
        print(f"🛑 Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Run Battlesnake games and collect data")
    parser.add_argument("--games", type=int, default=10, help="Number of games to run")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--width", type=int, default=11, help="Board width")
    parser.add_argument("--height", type=int, default=11, help="Board height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default="experiments/data/collected_data.parquet", help="Output file")
    args = parser.parse_args()

    import pandas as pd
    rows = run_games(args.games, args.port, args.width, args.height, args.seed)

    if not rows:
        print("❌ No data collected")
        return

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"\n✅ Saved {len(df)} rows to {out_path}")
    print(f"   Games: {df['game_id'].nunique()}")
    print(f"   Chosen moves: {df['chosen'].sum()}")


if __name__ == "__main__":
    main()
