import math
import sys

from mean_opinion_score.calculation import matlab_tinv


def test_095_from_one_to_hundered_returns_correct_values():
  correct_results = [
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
    2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
    2.0395, 2.0369, 2.0345, 2.0322, 2.0301, 2.0281, 2.0262, 2.0244, 2.0227, 2.0211,
    2.0195, 2.0181, 2.0167, 2.0154, 2.0141, 2.0129, 2.0117, 2.0106, 2.0096, 2.0086,
    2.0076, 2.0066, 2.0057, 2.0049, 2.0040, 2.0032, 2.0025, 2.0017, 2.0010, 2.0003,
    1.9996, 1.9990, 1.9983, 1.9977, 1.9971, 1.9966, 1.9960, 1.9955, 1.9949, 1.9944,
    1.9939, 1.9935, 1.9930, 1.9925, 1.9921, 1.9917, 1.9913, 1.9908, 1.9905, 1.9901,
    1.9897, 1.9893, 1.9890, 1.9886, 1.9883, 1.9879, 1.9876, 1.9873, 1.9870, 1.9867,
    1.9864, 1.9861, 1.9858, 1.9855, 1.9853, 1.9850, 1.9847, 1.9845, 1.9842, 1.9840,
  ]

  for deg_freedom, correct_result in zip(range(1, len(correct_results) + 1), correct_results):
    correct_results = matlab_tinv(0.5 * (1 + 0.95), deg_freedom)
    assert round(correct_results, 4) == correct_result


def test_095_200__returns_1_9718962236316089():
  result = matlab_tinv(0.5 * (1 + 0.95), 200)
  assert result == 1.9718962236316089


def test_py_36_37__095_inf__returns_1e100():
  result = matlab_tinv(0.5 * (1 + 0.95), math.inf)

  assert sys.version_info.major == 3
  if 6 <= sys.version_info.minor < 8:
    assert result == 1e+100


def test_py_38_39_310_311__095_inf__returns_1_959963984540054():
  result = matlab_tinv(0.5 * (1 + 0.95), math.inf)

  assert sys.version_info.major == 3
  if 8 <= sys.version_info.minor < 12:
    assert result == 1.959963984540054


def test_095_zero__returns_NaN():
  result = matlab_tinv(0.5 * (1 + 0.95), 0)
  assert math.isnan(result)
