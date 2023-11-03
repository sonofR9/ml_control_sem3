#pragma once
// module;

#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>

// export module runge_kutte;

// export import <array>;

namespace runge_kutte {
constexpr double kEps = 1e-10;

// --------------------------------StatePoint start---------------------------
// export {
/**
 * @brief class to represent state of system using T variables
 *
 * @tparam T number of state variables
 */
template <int T>
struct StatePoint {
  StatePoint() = default;
  ~StatePoint() = default;
  StatePoint(const StatePoint&) = default;
  StatePoint(StatePoint&&) = default;
  StatePoint& operator=(const StatePoint&) = default;
  StatePoint& operator=(StatePoint&&) = default;

  StatePoint(std::initializer_list<double> list) {
    if (list.size() != T)
      throw std::length_error("Initializer list size dffers from data size");
    auto el = list.begin();
    for (int i{0}; i < T; ++i) data_[i] = *(el++);
  }

  double& operator[](int i) {
    return data_[i];
  }
  double operator[](int i) const {
    return data_[i];
  }

 private:
  std::array<double, T> data_;
};

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <int T>
using StateDerivativesPoint = StatePoint<T>;

template <int T>
StatePoint<T> operator+(const StatePoint<T>& self, const StatePoint<T>& other) {
  StatePoint<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <int T>
StatePoint<T>& operator+=(StatePoint<T>& self, const StatePoint<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] += other[i];
  }
  return self;
}

template <int T>
StatePoint<T> operator-(const StatePoint<T>& self, const StatePoint<T>& other) {
  StatePoint<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = other[i] - self[i];
  }
  return result;
}

template <int T>
StatePoint<T>& operator-=(StatePoint<T>& self, const StatePoint<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] -= other[i];
  }
  return self;
}

template <int T, typename M>
StatePoint<T> operator*(const StatePoint<T>& self, M multiplier) {
  StatePoint<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = multiplier * self[i];
  }
  return result;
}
template <int T, typename M>
StatePoint<T> operator*(M multiplier, const StatePoint<T>& self) {
  return self * multiplier;
}
template <int T>
StatePoint<T>& operator*=(StatePoint<T>& self, const StatePoint<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] *= other[i];
  }
  return self;
}

template <int T, typename M>
StatePoint<T> operator/(const StatePoint<T>& self, M divider) {
  StatePoint<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <int N>
bool operator==(const StatePoint<N>& lhs, const StatePoint<N>& rhs) {
  constexpr double eps{1e-3};
  for (int i{0}; i < N; ++i) {
    if (fabs(lhs[i] - rhs[i]) > eps) {
      return false;
    }
  }
  return true;
}

template <int N>
bool operator!=(const StatePoint<N>& lhs, const StatePoint<N>& rhs) {
  return !(lhs == rhs);
}
// }
// --------------------------------StatePoint end------------------------------
// export {
template <typename F, int T>
concept StateSpaceFunction = requires(F func, StatePoint<T> point,
                                      double time) {
  { func(point, time) } -> std::same_as<StateDerivativesPoint<T>>;
};

template <int T, StateSpaceFunction<T> F>
StatePoint<T> RungeKutteStep(double startT, const StatePoint<T>& startX, F fun,
                             double interestT, double delta = 0.001) {
  double curT{startT};
  StatePoint<T> curX{startX};

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
    double startT, const StatePoint<T>& startX, F fun, double lastT,
    double delta = 0.001) {
  std::array<std::vector<double>, T + 1> result;
  double curT{startT};
  StatePoint<T> curX{startX};

  for (int i{0}; i < T; ++i) result[i].push_back(curX[i]);
  result[T].push_back(curT);

  while (curT < lastT - kEps) {
    curX = RungeKutteStep(
        curT, curX, std::function<StatePoint<3>(StatePoint<3>, double)>(fun),
        curT + delta, delta);
    curT += delta;
    for (int i{0}; i < T; ++i) result[i].push_back(curX[i]);
    result[T].push_back(curT);
  }

  return result;
}
// }
}  // namespace runge_kutte

template <int T>
std::ostream& operator<<(std::ostream& stream,
                         const runge_kutte::StatePoint<T>& state) {
  for (int i{0}; i < T; ++i) {
    stream << state[i] << " ";
  }
  return stream;
}
