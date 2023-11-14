#pragma once

#include "global.h"

#include <concepts>
#include <iostream>

namespace optimization {
/**
 * @brief
 *
 * @tparam T smth like Vector<N, typename smth>
 * @tparam G
 * @param qMin
 * @param qMax
 * @param grad
 * @param ksi
 * @return T
 */
template <typename T, Regular1OutFunction<T> F, GradientFunction<T> G>
T GradientDescent(const T& qMin, const T& qMax, F functional, G grad,
                  double ksi = 0.5, int maxIter = 1e5) {
  constexpr double kg{0.618034};

  auto qStar{qMin + ksi * (qMax - qMin)};
  auto q1{qStar};
  auto q2{qStar};

  for (int i{0}; i < maxIter; ++i) {
    const auto& gradStar{grad(qStar)};

    const auto golden = [qMin, qMax](const double kGold, const T& q,
                                     const T& grad) -> T {
      T result;
      for (int i{0}; i < T::size(); ++i) {
        const auto dq{q[i] + kGold * grad[i]};
        if (dq > qMax[i]) {
          result[i] = qMax[i];
        } else if (dq < qMin[i]) {
          result[i] = qMin[i];
        } else {
          result[i] = dq;
        }
      }
      return result;
    };

    auto qWave{golden(1, qStar, -gradStar)};
    q1 = golden(1 - kg, qStar, qWave - qStar);
    q2 = golden(kg, qStar, qWave - qStar);
    auto fQ1{functional(q1)};
    auto fQ2{functional(q2)};

    int iter{0};
    while (norm(q2 - q1) > kEps) {
      // std::cout << iter << " " << norm(q2 - q1) << std::endl;
      if (fQ1 < fQ2) {
        qWave = q2;
        q2 = q1;
        fQ2 = fQ1;
        q1 = golden(1 - kg, qStar, qWave - qStar);
        fQ1 = functional(q1);
      } else {
        qStar = q1;
        q1 = q2;
        fQ1 = fQ2;
        q2 = golden(kg, qStar, qWave - qStar);
        fQ2 = functional(q2);
      }
      ++iter;
    }
  }
  return (q1 + q2) / 2;
}

template <typename T, Regular1OutFunction<T> G>
T GradientDescent(const T& qMin, const T& qMax, G functional, double ksi = 0.5,
                  int maxIter = 1e5) {
  auto gradFunc = [&](const T& q) -> T {
    constexpr double dq{norm(qMax - qMin) / 1e6};
    T result{};
    T tmp{};
    for (int i{0}; i < T::size(); ++i) {
      tmp[i] = dq;
      result[i] = functional(q + tmp);
      tmp[i] = 0;
    }
    return result;
  };
  return GradientDescent(qMin, qMax, functional, gradFunc, ksi, maxIter);
}

}  // namespace optimization
