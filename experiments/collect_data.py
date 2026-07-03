"""Сбор данных для обучения модели Battlesnake."""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

# Импортируем логику из вашего проекта
import sys
sys.path.append(str(Path(__file__).parent.parent))
from logic import _candidate_features, _legal_moves, DIRECTIONS, choose_move_heuristic

class DataCollector:
    """Собирает данные о ходах змейки для обучения модели."""
    
    def __init__(self, save_dir: str = "experiments/data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.games_data = []
        
    def collect_game(self, game_state: Dict, move: str, reward: float, 
                     game_id: int, turn: int) -> Dict:
        """Собирает данные для одного хода."""
        # Получаем признаки для всех легальных ходов
        legal_moves = _legal_moves(game_state)
        features_by_move = {}
        
        for m in legal_moves:
            features = _candidate_features(game_state, m)
            features_by_move[m] = features
        
        # Сохраняем данные
        return {
            'game_id': game_id,
            'turn': turn,
            'chosen_move': move,
            'reward': reward,
            'legal_moves': legal_moves,
            'features': features_by_move,
            'health': game_state['you']['health'],
            'length': game_state['you']['length'],
            'food_count': len(game_state['board']['food']),
            'enemy_count': len([s for s in game_state['board']['snakes'] 
                               if s['id'] != game_state['you']['id']])
        }
    
    def save_data(self, filename: str = None):
        """Сохраняет собранные данные в CSV и Parquet."""
        if not self.games_data:
            print("Нет данных для сохранения.")
            return
        
        # Преобразуем в DataFrame
        rows = []
        for game in self.games_data:
            # Для каждого легального хода создаём строку
            for move in game['legal_moves']:
                features = game['features'][move]
                row = {
                    'game_id': game['game_id'],
                    'turn': game['turn'],
                    'move': move,
                    'chosen': (move == game['chosen_move']),
                    'reward': game['reward'],
                    'health': game['health'],
                    'length': game['length'],
                    'food_count': game['food_count'],
                    'enemy_count': game['enemy_count'],
                }
                # Добавляем все признаки
                row.update({f'feat_{k}': v for k, v in features.items()})
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Сохраняем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"battlesnake_data_{timestamp}"
        
        csv_path = self.save_dir / f"{filename}.csv"
        parquet_path = self.save_dir / f"{filename}.parquet"
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        
        print(f"✅ Сохранено {len(df)} строк данных")
        print(f"   CSV: {csv_path}")
        print(f"   Parquet: {parquet_path}")
        
        return df


def simulate_game(collector: DataCollector, game_id: int, 
                  max_turns: int = 200, verbose: bool = False) -> float:
    """Симулирует одну игру и собирает данные."""
    # Простая симуляция — здесь нужна интеграция с реальной игрой
    # В реальности вы будете использовать Battlesnake CLI или API
    
    from battlesnake import Game  # В реальности нужно подключить симулятор
    
    total_reward = 0
    game = Game()  # Инициализация игры (заглушка)
    
    for turn in range(max_turns):
        # Получаем текущее состояние (заглушка)
        game_state = game.get_state()
        
        # Выбираем ход (используем вашу текущую логику)
        from logic import choose_move
        move = choose_move(game_state)
        
        # Выполняем ход и получаем награду
        new_state, reward, done = game.step(move)
        
        # Собираем данные
        data = collector.collect_game(
            game_state, move, reward, game_id, turn
        )
        collector.games_data.append(data)
        
        total_reward += reward
        
        if verbose:
            print(f"Тур {turn}: ход {move}, награда {reward}")
        
        if done:
            break
    
    return total_reward


# ============================================================
# ⚡ БЫСТРЫЙ СТАРТ
# ============================================================

def quick_collect(n_games: int = 100, save: bool = True):
    """Быстрый сбор данных — запускает N игр."""
    collector = DataCollector()
    
    print(f"🚀 Запуск {n_games} игр для сбора данных...")
    
    # В реальности здесь нужно подключить симулятор
    # Сейчас просто создаём пример данных
    
    for game_id in range(n_games):
        # Генерируем пример данных (заглушка)
        game_data = {
            'game_id': game_id,
            'turn': 0,
            'chosen_move': 'up',
            'reward': 1.0,
            'legal_moves': ['up', 'down', 'left', 'right'],
            'features': {
                'up': {'space_capped': 10, 'open_space': 50, 'voronoi': 20, 'reaches_tail': 1.0, 'escape': 3, 'h2h_danger': 0.0, 'near_bigger_head': 5.0, 'near_enemy_head': 3.0, 'wall_dist': 2.0, 'food_score': 0.0, 'food_delta': 1.0, 'is_food': 0.0, 'dist_to_center': 3.0},
                'down': {'space_capped': 8, 'open_space': 40, 'voronoi': 15, 'reaches_tail': 0.0, 'escape': 2, 'h2h_danger': 1.0, 'near_bigger_head': 3.0, 'near_enemy_head': 2.0, 'wall_dist': 1.0, 'food_score': 0.0, 'food_delta': 0.5, 'is_food': 0.0, 'dist_to_center': 4.0},
                'left': {'space_capped': 9, 'open_space': 45, 'voronoi': 18, 'reaches_tail': 0.0, 'escape': 2, 'h2h_danger': 0.0, 'near_bigger_head': 4.0, 'near_enemy_head': 2.5, 'wall_dist': 1.5, 'food_score': 0.0, 'food_delta': 0.0, 'is_food': 0.0, 'dist_to_center': 3.5},
                'right': {'space_capped': 7, 'open_space': 35, 'voronoi': 12, 'reaches_tail': 0.0, 'escape': 1, 'h2h_danger': 0.0, 'near_bigger_head': 6.0, 'near_enemy_head': 4.0, 'wall_dist': 0.5, 'food_score': 0.0, 'food_delta': -1.0, 'is_food': 0.0, 'dist_to_center': 2.5},
            },
            'health': 100,
            'length': 3,
            'food_count': 5,
            'enemy_count': 1
        }
        collector.games_data.append(game_data)
        
        if (game_id + 1) % 10 == 0:
            print(f"  Собрано {game_id + 1} игр")
    
    if save:
        df = collector.save_data()
        return df, collector
    
    return collector


if __name__ == "__main__":
    # Пример запуска
    print("🐍 Сбор данных для обучения змейки")
    print("=" * 50)
    
    # Быстрый сбор 100 игр
    df, collector = quick_collect(n_games=100, save=True)
    
    print("\n📊 Статистика:")
    print(f"  Всего записей: {len(df)}")
    print(f"  Игр: {df['game_id'].nunique()}")
    print(f"  Ходов: {df['turn'].max() + 1}")
    print(f"  Уникальных ходов: {df['move'].unique().tolist()}")