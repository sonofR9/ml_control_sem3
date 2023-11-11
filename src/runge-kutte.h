#pragma once
// module;

#include "global.h"

#include <array>
#include <cmath>
#include <iostream>

// export module runge_kutte;

// export import <array>;

namespace optimization {
template <int T, StateSpaceFunction<T> F>
Vector<T> RungeKutteStep(double startT, const Vector<T>& startX, F fun,
                         double interestT, double delta = 0.001) {
  double curT{startT};
  Vector<T> curX{startX};

  while (curT < interestT) {
    auto k1 = fun(curX, curT);
    auto k2 = fun(curX + delta / 2 * k1, curT + delta / 2);
    auto k3 = fun(curX + delta / 2 * k2, curT + delta / 2);
    auto k4 = fun(curX + delta * k3, curT + delta);

    curX += delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    curT += delta;
  }
  return curX;
}

/**
 * @tparam T number of state coordinates
 * @return std::array<std::vector<double>, T + 1> array of vectors with values
 * of each state coordinate and time
 */
template <int T, StateSpaceFunction<T> F>
std::array<std::vector<double>, T + 1> SolveDiffEqRungeKutte(
    double startT, const Vector<T>& startX, F fun, double lastT,
    double delta = 0.001) {
  std::array<std::vector<double>, T + 1> result;
  double curT{startT};
  Vector<T> curX{startX};

  for (int i{0}; i < T; ++i) result[i].push_back(curX[i]);
  result[T].push_back(curT);

  while (curT < lastT - kEps) {
    curX = RungeKutteStep(curT, curX,
                          std::function<Vector<3>(Vector<3>, double)>(fun),
                          curT + delta, delta);
    curT += delta;
    for (int i{0}; i < T; ++i) result[i].push_back(curX[i]);
    result[T].push_back(curT);
  }

  return result;
}
// }
}  // namespace optimization

