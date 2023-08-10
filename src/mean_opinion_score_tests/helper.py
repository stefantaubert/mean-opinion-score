import json
from pathlib import Path

import numpy as np


def get_test_resource_path(name: str) -> Path:
  path = Path(f"src/mean_opinion_score_tests/res/{name}")
  assert path.is_file()
  return path


def load_ratings_from_json(path: Path) -> np.ndarray:
  with open(path, "r", encoding="utf-8") as f:
    ratings_list = json.load(f)

  ratings = np.array(ratings_list, dtype=np.float32)
  ratings[ratings == 0] = np.nan
  return ratings
