import numpy as np

from mean_opinion_score.mos_variance import compute_mos

_ = np.nan


def test_component():
  Z = np.array([
    [5, 4, 1, 2, _, 5, 1, 3, 4, 4, _, 1, _],
    [4, 5, 2, 1, 4, _, 2, 2, 4, 5, 3, _, _],
    [5, 5, 1, 1, 5, _, 1, 3, 5, 5, 2, _, _],
    [5, _, 1, 2, 4, _, 1, _, 4, 4, 2, _, _],
    [_, 4, _, _, _, _, _, 2, _, _, _, _, _],
    [_, _, _, _, _, _, _, _, _, _, _, _, _],
  ])

  result = compute_mos(Z)

  assert result == 3.1


def test_empty__returns_nan():
  Z = np.array([[]])

  result = compute_mos(Z)

  assert np.isnan(result)


def test_1x1_nan__returns_nan():
  Z = np.array([[np.nan]])

  result = compute_mos(Z)

  assert np.isnan(result)


def test_2x1_all_nan__returns_nan():
  Z = np.array([[np.nan], [np.nan]])

  result = compute_mos(Z)

  assert np.isnan(result)


def test_2x2_all_nan__returns_nan():
  Z = np.array([[np.nan, np.nan], [np.nan, np.nan]])

  result = compute_mos(Z)

  assert np.isnan(result)


def test_2x2_nan_nan_nan_1__returns_1():
  Z = np.array([[np.nan, np.nan], [np.nan, 1]])

  result = compute_mos(Z)

  assert result == 1
