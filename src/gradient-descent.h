#pragma once

#include "global.h"

#include <concepts>

namespace optimization {
template <typename T, GradientFunction<T> G>
T GradientDescent(const T& qMin, const T& qMax, G grad, double ksi = 0.5) {
  constexpr double kg{0.618034};

  auto qStar{ksi * (qMin + qMax)};
  const auto& gradStar{grad(qStar)};

  const auto golden = [qMin, qMax](const double kGold, const T& q,
                                   const T& grad) -> T {
    T result;
    for (int i{0}; i < T::size(); ++i) {
      const auto dq{q[i] + kGold * grad[i]};
      if (dq > qMax) {
        result[i] = qMax;
      } else if (dq < qMin) {
        result[i] = qMin;
      } else {
        result[i] = dq;
      }
      return result;
    }
  };

  auto qWave{golden(1, qStar, -gradStar)};
  auto q1{golden(1 - kg, qStar, qWave - qStar)};
  auto q2{golden(kg, qStar, qWave - qStar)};
  auto gradQ1{grad(q1)};
  auto gradQ2{grad(q2)};

  while (norm(q2 - q1) > kEps) {
    if (gradQ1 < gradQ2) {
      qWave = q2;
      q2 = q1;
      gradQ2 = gradQ1;
      q1 = golden(1 - kg, qStar, qWave - qStar);
      gradQ1 = grad(q1);
    } else {
      qWave = q1;
      q1 = q2;
      gradQ1 = gradQ2;
      q2 = golden(kg, qStar, qWave - qStar);
      gradQ2 = grad(q2);
    }
  }
  return (q1 + q2) / 2;
}

template <typename T, Regular1OutFunction<T> G>
T GradientDescent(const T& qMin, const T& qMax, G grad, double ksi = 0.5) {
  auto gradFunc = [&](const T& q) -> T {
    constexpr double dq{norm(qMax - qMin) / 1e6};
    T result{};
    T tmp{};
    for (int i{0}; i < T::size(); ++i) {
      tmp[i] = dq;
      result[i] = grad(q + tmp);
      tmp[i] = 0;
    }
    return result;
  };
  return GradientDescent(qMin, qMax, gradFunc, ksi);
}

}  // namespace optimization
