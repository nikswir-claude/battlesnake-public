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
from logic import _candidate_features, _legal_moves, DIRECTIONS, choose_move_heuristic, _occupied_cells, _head_to_head_cells, _flood_fill, _in_bounds, _manhattan, HUNGRY_THRESHOLD, _BIG, _NEIGHBORS

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


def simulate_game(collector: DataCollector, game_state: Dict, move: str,
                  reward: float, game_id: int, turn: int) -> None:
    """Собирает данные для одного хода из реального состояния игры."""
    data = collector.collect_game(game_state, move, reward, game_id, turn)
    collector.games_data.append(data)


# ============================================================
# ⚡ БЫСТРЫЙ СТАРТ
# ============================================================

def quick_collect(n_games: int = 100, save: bool = True):
    """Быстрый сбор данных — генерирует синтетические данные для тестирования."""
    import random
    
    collector = DataCollector()
    
    print(f"🚀 Генерация {n_games} синтетических игр для тестирования...")
    
    width, height = 11, 11
    
    for game_id in range(n_games):
        # Генерируем случайное состояние игры
        head_x = random.randint(2, width - 3)
        head_y = random.randint(2, height - 3)
        health = random.randint(20, 100)
        length = random.randint(3, 10)
        
        # Создаём синтетическое состояние
        game_state = {
            'board': {
                'width': width,
                'height': height,
                'food': [
                    {'x': random.randint(0, width - 1), 'y': random.randint(0, height - 1)}
                    for _ in range(3)
                ],
                'snakes': [
                    {
                        'id': 'us',
                        'head': {'x': head_x, 'y': head_y},
                        'body': [
                            {'x': head_x - i, 'y': head_y} for i in range(length)
                        ],
                        'length': length,
                    }
                ]
            },
            'you': {
                'id': 'us',
                'head': {'x': head_x, 'y': head_y},
                'body': [
                    {'x': head_x - i, 'y': head_y} for i in range(length)
                ],
                'length': length,
                'health': health,
            }
        }
        
        # Выбираем ход
        move = choose_move_heuristic(game_state)
        
        # Собираем данные
        data = collector.collect_game(game_state, move, 0.0, game_id, 0)
        collector.games_data.append(data)
        
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