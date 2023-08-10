import numpy as np

from mean_opinion_score.calculation import get_ci95
from mean_opinion_score_tests.helper import get_test_resource_path, load_ratings_from_json


def test_blizzard_crowdmos1_hp():
  path = get_test_resource_path("blizzard_crowdmos1_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.11618882, 0.4321971, 0.50067484, 0.45428753, 0.45001453,
    0.39287296, 0.4284437, 0.40747246, 0.39982477, 0.44571114,
    0.39014196, 0.43808025, 0.41212726, 0.41697392, 0.40702996,
    0.3145306, 0.51542485, 0.37011412
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_hp():
  path = get_test_resource_path("blizzard_crowdmos2_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.051524, 0.4005236, 0.34172955, 0.34118438, 0.399796,
    0.29253745, 0.3689839, 0.28519785, 0.3422088, 0.3130137,
    0.3288277, 0.30906132, 0.30260444, 0.32713827, 0.3639018,
    0.27469155, 0.33445048, 0.30748326
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_ls():
  path = get_test_resource_path("blizzard_crowdmos2_ls.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.15285364, 0.58587676, 0.44803184, 0.5102908, 0.528,
    0.39689648, 0.4911777, 0.41779906, 0.45866576, 0.41913238,
    0.37948135, 0.38489953, 0.38505414, 0.36011842, 0.4903041,
    0.37959763, 0.3461114, 0.48029682
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_paid_participants():
  path = get_test_resource_path("blizzard_paid_participants.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.08334564, 0.27342203, 0.29472297, 0.24695815, 0.24384049,
    0.23937851, 0.22638696, 0.263661, 0.20880696, 0.24739781,
    0.19995913, 0.22837521, 0.2803301, 0.23874208, 0.2529408,
    0.17403981, 0.23581083, 0.21096095
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_online_volunteers():
  path = get_test_resource_path("blizzard_online_volunteers.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.11203187, 0.29539856, 0.3842204, 0.39001012, 0.42414582,
    0.4009665, 0.3900101, 0.3990989, 0.34703195, 0.44846082,
    0.32479122, 0.41596445, 0.43458575, 0.37952507, 0.3542759,
    0.26102635, 0.4237934, 0.39495915
  ], rtol=1e-7, atol=1e-8)


def compute_alg_ci95(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty(n_algorithms, dtype=np.float32)
  for algo_i in range(n_algorithms):
    result[algo_i] = get_ci95(ratings[algo_i])
  return result
