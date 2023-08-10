import numpy as np

from mean_opinion_score.calculation import get_ci95_default
from mean_opinion_score_tests.helper import get_test_resource_path, load_ratings_from_json


def test_blizzard_crowdmos1_hp():
  path = get_test_resource_path("blizzard_crowdmos1_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.0421344, 0.14138715, 0.15142073, 0.13476837, 0.14331244,
    0.13098319, 0.13475642, 0.12738818, 0.1250711, 0.1339533,
    0.11844325, 0.13718148, 0.12606472, 0.12401591, 0.12517108,
    0.11514077, 0.15034962, 0.1246323
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_hp():
  path = get_test_resource_path("blizzard_crowdmos2_hp.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.02484817, 0.09491657, 0.08836678, 0.08876229, 0.09624269,
    0.08481681, 0.09519985, 0.08010752, 0.0854466, 0.08621473,
    0.08549163, 0.09014896, 0.08397444, 0.08289323, 0.08742789,
    0.07803175, 0.08978353, 0.08665785
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_ls():
  path = get_test_resource_path("blizzard_crowdmos2_ls.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.05542524, 0.1634123, 0.1417076, 0.15522435, 0.15742232,
    0.13509941, 0.15691558, 0.13351098, 0.15590177, 0.14155106,
    0.13709177, 0.12907982, 0.1323009, 0.12543218, 0.14426807,
    0.12945496, 0.12886404, 0.14700325
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_paid_participants():
  path = get_test_resource_path("blizzard_paid_participants.json")
  ratings = load_ratings_from_json(path)

  result = compute_alg_ci95(ratings)

  np.testing.assert_allclose(result, [
    0.07742594, 0.25400198, 0.27379003, 0.22941771, 0.2265215,
    0.22237642, 0.21030763, 0.24493425, 0.19397625, 0.22982614,
    0.18575685, 0.21215464, 0.26041937, 0.22178519, 0.23497547,
    0.16167846, 0.21906215, 0.19597726
  ], rtol=1e-7, atol=1e-8)


def compute_alg_ci95(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty(n_algorithms, dtype=np.float32)
  for algo_i in range(n_algorithms):
    result[algo_i] = get_ci95_default(ratings[algo_i])
  return result
