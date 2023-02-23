import numpy as np

from mean_opinion_score.mos_variance import non_nan_count


def test_1_dim_component():
  vec = np.array([5, 3, np.nan, 4, np.nan, np.nan, 6, 1])
  result = non_nan_count(vec)
  assert result == 5


def test_1_dim_empty__returns_0():
  vec = np.array([])
  result = non_nan_count(vec)
  assert result == 0


def test_1_dim_0_num_1_nan__returns_0():
  vec = np.array([np.nan])
  result = non_nan_count(vec)
  assert result == 0


def test_1_dim_1_num_0_nan__returns_1():
  vec = np.array([1])
  result = non_nan_count(vec)
  assert result == 1


def test_1_dim_1_num_1_nan__returns_1():
  vec = np.array([1, np.nan])
  result = non_nan_count(vec)
  assert result == 1


def test_1_dim_2_num_1_nan__returns_2():
  vec = np.array([1, 1, np.nan])
  result = non_nan_count(vec)
  assert result == 2


def test_2_dim_component():
  vec = np.array([[5, 3], [np.nan, 4], [np.nan, np.nan], [6, 1]])
  result = non_nan_count(vec)
  assert result == 5


def test_2_dim_empty__returns_0():
  vec = np.array([[]])
  result = non_nan_count(vec)
  assert result == 0


def test_2_dim_1_num_0_nan__returns_1():
  vec = np.array([[1]])
  result = non_nan_count(vec)
  assert result == 1


def test_2_dim_2_num_1_nan__returns_2():
  vec = np.array([[1, 1, np.nan]])
  result = non_nan_count(vec)
  assert result == 2
