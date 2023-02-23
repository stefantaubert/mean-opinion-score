import numpy as np

from mean_opinion_score.calculation import get_v_mu


def test_component():
  _ = np.nan
  Z = np.array([
    [5, 4, 1, 2, _, 5, 1, 3, 4, 4, _, 1, _],
    [4, 5, 2, 1, 4, _, 2, 2, 4, 5, 3, _, _],
    [5, 5, 1, 1, 5, _, 1, 3, 5, 5, 2, _, _],
    [5, _, 1, 2, 4, _, 1, _, 4, 4, 2, _, _],
    [_, 4, _, _, _, _, _, 2, _, _, _, _, _],
    [_, _, _, _, _, _, _, _, _, _, _, _, _],
  ])

  result = get_v_mu(Z)

  assert result == 0.25230010602623226


def test_v_su_not_NaN_and_v_wu_not_NaN():
  _ = np.nan
  Z = np.array([
    [5, 4],
    [4, 5],
  ])

  result = get_v_mu(Z)

  assert result == 0.0625


def test_v_su_NaN_and_v_wu_not_NaN():
  _ = np.nan
  Z = np.array([
    [5, 4],
  ])

  result = get_v_mu(Z)

  assert result == 0.125


def test_v_su_not_NaN_and_v_wu_NaN():
  _ = np.nan
  Z = np.array([
    [5],
    [4],
  ])

  result = get_v_mu(Z)

  assert result == 0.125


def test_v_su_NaN_and_v_wu_NaN_and_Z_not_empty__returns_NaN():
  Z = np.array([[1]])

  result = get_v_mu(Z)

  assert np.isnan(result)


def test_v_su_NaN_and_v_wu_NaN_and_Z_empty__returns_NaN():
  Z = np.array([[]])

  result = get_v_mu(Z)

  assert np.isnan(result)
