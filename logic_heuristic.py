"""Serve the heuristic fallback directly (for A/B benchmarking).

Points ``choose_move`` at ``choose_move_heuristic`` so the bench harness can pit
the served linear model against its own heuristic fallback. Not used in
production; select it via ``LOGIC_MODULE=logic_heuristic``.
"""

from logic import choose_move_heuristic as choose_move  # noqa: F401
from logic import get_info  # noqa: F401
