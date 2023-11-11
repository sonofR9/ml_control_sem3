#pragma once

#include "global.h"

#include <concepts>

namespace optimization {
template <typename F, int N>
concept GradientFunction = requires(F func, const Vector<N>& inp) {
                             { func(inp) } -> std::same_as<Vector<N>>;
                           };

template <typename F, int N>
concept Regular1OutFunction = requires(F func, const Vector<N>& inp) {
                                { func(inp) } -> std::same_as<double>;
                              };

template <int N, GradientFunction<N> G>
Vector<N> GradientDescent(const Vector<N>& qMin, const Vector<N>& qMax, G grad,
                          double ksi = 0.5) {
  constexpr double kg{0.618034};

  auto qStar{ksi * (qMin + qMax)};
  const auto& gradStar{grad(qStar)};

  const auto golden = [qMin, qMax](const double kGold, const Vector<N>& q,
                                   const Vector<N>& grad) -> Vector<N> {
    Vector<N> result;
    for (int i{0}; i < N; ++i) {
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

template <int N, Regular1OutFunction<N> G>
Vector<N> GradientDescent(const Vector<N>& qMin, const Vector<N>& qMax, G grad,
                          double ksi = 0.5) {
  auto gradFunc = [&](const Vector<N>& q) -> Vector<N> {
    constexpr double dq{norm(qMax - qMin) / 1e6};
    Vector<N> result{};
    Vector<N> tmp{};
    for (int i{0}; i < N; ++i) {
      tmp[i] = dq;
      result[i] = grad(q + tmp);
      tmp[i] = 0;
    }
    return result;
  };
  return GradientDescent(qMin, qMax, gradFunc, ksi);
}

}  // namespace optimization
