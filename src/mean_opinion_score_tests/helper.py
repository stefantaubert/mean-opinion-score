from pathlib import Path

import numpy as np

from mean_opinion_score.calculation import get_ci95, get_mos


def get_test_resource_path(name: str) -> Path:
  path = Path(f"src/mean_opinion_score_tests/res/{name}")
  assert path.is_file()
  return path
