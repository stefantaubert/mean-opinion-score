import json
from pathlib import Path

import numpy as np

from mean_opinion_score.calculation import get_ci95, get_mos
from mean_opinion_score_tests.helper import get_test_resource_path


def load_ratings_from_json(path: Path) -> np.ndarray:
  with open(path, "r", encoding="utf-8") as f:
    ratings_list = json.load(f)

  ratings = np.array(ratings_list, dtype=np.float32)
  ratings[ratings == 0] = np.nan
  return ratings


def test_blizzard_crowdmos1_hp():
  path = get_test_resource_path("blizzard_crowdmos1_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos_ci95(ratings)

  np.testing.assert_allclose(result, [
    [
      4.908, 2.930328, 2.9626555, 2.580247, 2.3690987, 2.9387755, 3.3471074, 2.7206478, 3.792683, 3.0365853, 2.02834, 2.979339, 2.877551, 2.282258, 2.322314, 3.904, 2.701613, 2.498008
    ],
    [
      0.11618882, 0.4321971, 0.50067484, 0.45428753, 0.45001453,
      0.39287296, 0.4284437, 0.40747246, 0.39982477, 0.44571114,
      0.39014196, 0.43808025, 0.41212726, 0.41697392, 0.40702996,
      0.3145306, 0.51542485, 0.37011412
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_hp():
  path = get_test_resource_path("blizzard_crowdmos2_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos_ci95(ratings)

  np.testing.assert_allclose(result, [
    [
      4.921941, 2.8311965, 3.0, 2.811159, 2.4888394, 3.0592105, 3.145055, 2.6876357, 3.665962, 3.178022, 2.0172787, 3.0334077, 2.8394794, 2.124731, 2.249453, 3.9308856, 2.74375, 2.6858406
    ],
    [
      0.051524, 0.4005236, 0.34172955, 0.34118438, 0.399796,
      0.29253745, 0.3689839, 0.28519785, 0.3422088, 0.3130137,
      0.3288277, 0.30906132, 0.30260444, 0.32713827, 0.3639018,
      0.27469155, 0.33445048, 0.30748326
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_ls():
  path = get_test_resource_path("blizzard_crowdmos2_ls.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos_ci95(ratings)

  np.testing.assert_allclose(result, [
    [
      4.8125, 2.8155339, 3.3615024, 3.1534884, 2.495238, 3.162162, 3.2102804, 2.6439025, 3.6517413, 3.55, 2.1857142, 3.3640554, 2.9581394, 2.088372, 2.2488263, 3.9812207, 3.5727699, 3.0536585
    ],
    [
      0.15285364, 0.58587676, 0.44803184, 0.5102908, 0.528,
      0.39689648, 0.4911777, 0.41779906, 0.45866576, 0.41913238,
      0.37948135, 0.38489953, 0.38505414, 0.36011842, 0.4903041,
      0.37959763, 0.3461114, 0.48029682
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_paid_participants():
  path = get_test_resource_path("blizzard_paid_participants.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos_ci95(ratings)

  np.testing.assert_allclose(result, [
    [
      4.8875, 2.8625, 2.8375, 2.4375, 2.2625, 2.7125, 3.5625, 2.475, 3.9375, 3.0, 2.1375, 2.9875, 2.3875, 2.225, 2.5125, 4.175, 2.025, 2.1125
    ],
    [
      0.08334564, 0.27342203, 0.29472297, 0.24695815, 0.24384049,
      0.23937851, 0.22638696, 0.263661, 0.20880696, 0.24739781,
      0.19995913, 0.22837521, 0.2803301, 0.23874208, 0.2529408,
      0.17403981, 0.23581083, 0.21096095
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_online_volunteers():
  path = get_test_resource_path("blizzard_online_volunteers.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos_ci95(ratings)

  np.testing.assert_allclose(result, [
    [
      4.903226, 3.1935484, 2.935484, 2.8064516, 2.8064516, 3.096774, 3.1935484, 2.7096775, 4.0, 3.2258065, 2.3225806, 2.612903, 2.3225806, 2.6451614, 2.3548386, 4.096774, 2.3225806, 2.451613
    ],
    [
      0.11203187, 0.29539856, 0.3842204, 0.39001012, 0.42414582,
      0.4009665, 0.3900101, 0.3990989, 0.34703195, 0.44846082,
      0.32479122, 0.41596445, 0.43458575, 0.37952507, 0.3542759,
      0.26102635, 0.4237934, 0.39495915
    ]
  ], rtol=1e-7, atol=1e-8)


def compute_alg_mos(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty(n_algorithms, dtype=np.float32)
  for algo_i in range(n_algorithms):
    result[algo_i] = get_mos(ratings[algo_i])
  return result


def compute_alg_ci95(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty(n_algorithms, dtype=np.float32)
  for algo_i in range(n_algorithms):
    result[algo_i] = get_ci95(ratings[algo_i])
  return result


def compute_alg_mos_ci95(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty((2, n_algorithms), dtype=np.float32)
  result[0, :] = compute_alg_mos(ratings)
  result[1, :] = compute_alg_ci95(ratings)
  return result
