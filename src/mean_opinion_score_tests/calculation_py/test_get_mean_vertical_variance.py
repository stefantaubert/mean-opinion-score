import numpy as np

from mean_opinion_score.calculation import get_mean_vertical_variance


def test_component():
  _ = np.nan
  Z = np.array([
    [5, 4, 1, 2, _, 5, 1, 3, 4, 4, _, 1],
    [4, 5, 2, 1, 4, _, 2, 2, 4, 5, 3, _],
    [5, 5, 1, 1, 5, _, 1, 3, 5, 5, 2, _],
    [3, _, 1, 2, 4, _, 1, _, 3, 4, 2, _],
    [_, 4, _, _, _, _, _, 2, _, _, _, _],
    [_, _, _, _, _, _, _, _, _, _, _, _],
  ])

  result = get_mean_vertical_variance(Z)

  np.testing.assert_allclose(result, 0.30069447, rtol=1e-07, atol=1e-08)


def test_empty():
  _ = np.nan
  Z = np.full((0, 0), fill_value=np.nan)

  result = get_mean_vertical_variance(Z)

  assert np.isnan(result)


def test_one_worker_zero_ratings__returns_NaN():
  _ = np.nan
  Z = np.full((1, 0), fill_value=np.nan)

  result = get_mean_vertical_variance(Z)

  assert np.isnan(result)


def test_one_worker_one_rating__returns_NaN():
  _ = np.nan
  Z = np.array([
    [5],
  ])

  result = get_mean_vertical_variance(Z)

  assert np.isnan(result)
