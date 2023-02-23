from pathlib import Path


def get_test_resource_path(name: str) -> Path:
  path = Path(f"src/mean_opinion_score_tests/res/{name}")
  assert path.is_file()
  return path
