import numpy as np

from mean_opinion_score.calculation import get_mos
from mean_opinion_score_tests.helper import get_test_resource_path, load_ratings_from_json


def test_blizzard_crowdmos1_hp():
  path = get_test_resource_path("blizzard_crowdmos1_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos(ratings)

  np.testing.assert_allclose(result, [
    4.908, 2.930328, 2.9626555, 2.580247, 2.3690987, 2.9387755, 3.3471074, 2.7206478, 3.792683, 3.0365853, 2.02834, 2.979339, 2.877551, 2.282258, 2.322314, 3.904, 2.701613, 2.498008
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_hp():
  path = get_test_resource_path("blizzard_crowdmos2_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos(ratings)

  np.testing.assert_allclose(result, [
    4.921941, 2.8311965, 3.0, 2.811159, 2.4888394, 3.0592105, 3.145055, 2.6876357, 3.665962, 3.178022, 2.0172787, 3.0334077, 2.8394794, 2.124731, 2.249453, 3.9308856, 2.74375, 2.6858406
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_ls():
  path = get_test_resource_path("blizzard_crowdmos2_ls.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos(ratings)

  np.testing.assert_allclose(result, [
    4.8125, 2.8155339, 3.3615024, 3.1534884, 2.495238, 3.162162, 3.2102804, 2.6439025, 3.6517413, 3.55, 2.1857142, 3.3640554, 2.9581394, 2.088372, 2.2488263, 3.9812207, 3.5727699, 3.0536585
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_paid_participants():
  path = get_test_resource_path("blizzard_paid_participants.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos(ratings)

  np.testing.assert_allclose(result, [
    4.8875, 2.8625, 2.8375, 2.4375, 2.2625, 2.7125, 3.5625, 2.475, 3.9375, 3.0, 2.1375, 2.9875, 2.3875, 2.225, 2.5125, 4.175, 2.025, 2.1125
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_online_volunteers():
  path = get_test_resource_path("blizzard_online_volunteers.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_mos(ratings)

  np.testing.assert_allclose(result, [
    4.903226, 3.1935484, 2.935484, 2.8064516, 2.8064516, 3.096774, 3.1935484, 2.7096775, 4.0, 3.2258065, 2.3225806, 2.612903, 2.3225806, 2.6451614, 2.3548386, 4.096774, 2.3225806, 2.451613
  ], rtol=1e-7, atol=1e-8)


def compute_alg_mos(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty(n_algorithms, dtype=np.float32)
  for algo_i in range(n_algorithms):
    result[algo_i] = get_mos(ratings[algo_i])
  return result
