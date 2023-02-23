# ------------------------
# Calculation was taken from:
# Ribeiro, F., Florêncio, D., Zhang, C., & Seltzer, M. (2011). CrowdMOS: An approach for crowdsourcing mean opinion score studies. 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2416–2419. https://doi.org/10.1109/ICASSP.2011.5946971
# ------------------------

from math import sqrt

import numpy as np
from scipy.stats import t


def get_mos(Z: np.ndarray) -> float:
  if np.isnan(Z).all():
    return np.nan
  mos = np.nanmean(Z)
  return mos


def get_ci95(Z: np.ndarray) -> float:
  # Computes the 95% confidence interval using the sum of 3 Gaussian models.
  v_mu = get_v_mu(Z)
  t = matlab_tinv(0.5 * (1 + 0.95), min(Z.shape) - 1)
  ci95 = t * sqrt(v_mu)
  return ci95


def get_v_mu(Z: np.ndarray) -> float:
  # Determines the variance of the mean opinion score, given a matrix of ratings.
  # Unknown ratings are represented by NaN.
  #
  # Z: a N-by-M matrix of ratings, where the rows are subjects and columns are sentences
  # We assume that
  #
  # 	Z_ij = mu + x_i + y_j + eps_ij, where
  #
  # mu is the mean opinion score (given by np.nanmean(Z))
  # x_i ~ N(0, sigma_w^2), with sigma_w^2 modeling worker variation
  # y_j ~ N(0, sigma_s^2), with sigma_y^2 modeling sentence variation
  # eps_ij ~ N(0, sigma_u^2), with sigma_u^2 modeling worker uncertainty
  #
  # The returned value v_mu is Var[mu].

  v_su = get_su(Z)  # v_su  = v_s + v_u
  v_wu = get_wu(Z)  # v_wu  = v_w + v_u
  v_swu = get_swu(Z)  # v_swu = v_s + v_w + v_u
  W: np.ndarray = ~np.isnan(Z)
  M_s: np.ndarray = np.sum(W, axis=0)
  N_w: np.ndarray = np.sum(W, axis=1)

  if not np.isnan(v_su) and not np.isnan(v_wu):
    return get_v_mu_from_v_su_v_wu_and_v_swu(v_su, v_wu, v_swu, M_s, N_w)

  if np.isnan(v_su) and not np.isnan(v_wu):
    return get_v_mu_from_v_wu_and_v_swu(v_wu, v_swu, N_w)

  if not np.isnan(v_su) and np.isnan(v_wu):
    return get_v_mu_from_v_su_and_v_swu(v_su, v_swu, M_s)

  assert np.isnan(v_su)
  assert np.isnan(v_wu)

  if not np.isnan(v_swu):
    return get_v_mu_from_v_swu(v_swu, W)

  return np.nan


def get_v_mu_from_v_su_v_wu_and_v_swu(v_su: float, v_wu: float, v_swu: float, M_s: np.ndarray, N_w: np.ndarray) -> float:
  assert not np.isnan(v_su)
  assert not np.isnan(v_wu)
  assert not np.isnan(v_swu)

  assert np.sum(M_s) == np.sum(N_w)
  T = np.sum(M_s)

  a = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
  ])
  b = np.array([
    v_su,
    v_wu,
    v_swu,
  ], dtype=np.float32)

  x = np.linalg.solve(a, b)
  x = np.maximum(x, 0)

  v_s = x[0]
  v_w = x[1]
  v_u = x[2]

  v_mu = \
      v_s * np.sum(M_s**2) / T**2 + \
      v_w * np.sum(N_w**2) / T**2 + \
      v_u / T

  return v_mu


def get_v_mu_from_v_wu_and_v_swu(v_wu: float, v_swu: float, N_w: np.ndarray) -> float:
  assert not np.isnan(v_wu)
  assert not np.isnan(v_swu)

  T = np.sum(N_w)

  a = np.array([
    [0, 1],
    [1, 1],
  ])
  b = np.array([
    v_wu,
    v_swu,
  ])

  x = np.linalg.solve(a, b)
  x = np.maximum(x, 0)

  v_w = x[0]
  v_u = x[1]

  v_mu = \
      v_w * np.sum(N_w**2) / T**2 + \
      v_u / T
  return v_mu


def get_v_mu_from_v_su_and_v_swu(v_su: float, v_swu: float, M_s: np.ndarray) -> float:
  assert not np.isnan(v_su)
  assert not np.isnan(v_swu)

  T = np.sum(M_s)

  a = np.array([
    [0, 1],
    [1, 1],
  ])
  b = np.array([
    v_su,
    v_swu,
  ])

  x = np.linalg.solve(a, b)
  x = np.maximum(x, 0)

  v_s = x[0]
  v_u = x[1]

  v_mu = \
      v_s * np.sum(M_s**2) / T**2 + \
      v_u / T

  return v_mu


def get_v_mu_from_v_swu(v_swu: float, W: np.ndarray) -> float:
  assert not np.isnan(v_swu)

  T = np.sum(W)

  v_mu = v_swu / T

  return v_mu


def get_su(Z: np.ndarray) -> float:
  return get_mean_vertical_variance(Z.T)


def get_wu(Z: np.ndarray) -> float:
  return get_mean_vertical_variance(Z)


def get_swu(Z: np.ndarray) -> float:
  return get_custom_variance(Z)


def get_mean_vertical_variance(Z: np.ndarray) -> float:
  assert len(Z.shape) == 2
  vertical_dim = Z.shape[1]
  vertical_variances = np.full(
    vertical_dim,
    fill_value=np.nan,
    dtype=np.float32,
  )
  for index in range(vertical_dim):
    vertical_variances[index] = get_custom_variance(Z[:, index])

  if np.isnan(vertical_variances).all():
    return np.nan
  mean_variance = np.nanmean(vertical_variances)
  return mean_variance


def get_custom_variance(vec: np.ndarray) -> float:
  result = np.nan
  if get_non_nan_count(vec) > 1:
    result = np.nanvar(vec)
  return result


def get_non_nan_count(vec: np.ndarray) -> int:
  result = np.sum(~np.isnan(vec))
  return result


def matlab_tinv(p: float, df: int) -> float:
  result = -t.isf(p, df)
  return result
