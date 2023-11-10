#pragma once
// module;

#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>

// export module runge_kutte;

// export import <array>;

namespace optimization {
constexpr double kEps = 1e-10;

// --------------------------------StatePoint start---------------------------
// export {
/**
 * @brief class to represent state of system using T variables
 *
 * @tparam T number of state variables
 */
template <int T>
struct Vector {
  Vector() = default;
  ~Vector() = default;
  Vector(const Vector&) = default;
  Vector(Vector&&) = default;
  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) = default;

  Vector(std::initializer_list<double> list) {
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

template <int T>
using StatePoint = Vector<T>;

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <int T>
using StateDerivativesPoint = Vector<T>;

template <int T>
Vector<T> operator+(const Vector<T>& self, const Vector<T>& other) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <int T>
Vector<T>& operator+=(Vector<T>& self, const Vector<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] += other[i];
  }
  return self;
}

template <int T>
Vector<T> operator-(const Vector<T>& self, const Vector<T>& other) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = other[i] - self[i];
  }
  return result;
}

template <int T>
Vector<T>& operator-=(Vector<T>& self, const Vector<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] -= other[i];
  }
  return self;
}

template <int T, typename M>
Vector<T> operator*(const Vector<T>& self, M multiplier) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = multiplier * self[i];
  }
  return result;
}
template <int T, typename M>
Vector<T> operator*(M multiplier, const Vector<T>& self) {
  return self * multiplier;
}
template <int T>
Vector<T>& operator*=(Vector<T>& self, const Vector<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] *= other[i];
  }
  return self;
}

template <int T, typename M>
Vector<T> operator/(const Vector<T>& self, M divider) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <int N>
bool operator==(const Vector<N>& lhs, const Vector<N>& rhs) {
  constexpr double eps{1e-3};
  for (int i{0}; i < N; ++i) {
    if (fabs(lhs[i] - rhs[i]) > eps) {
      return false;
    }
  }
  return true;
}

template <int N>
bool operator!=(const Vector<N>& lhs, const Vector<N>& rhs) {
  return !(lhs == rhs);
}
// }
// --------------------------------StatePoint end------------------------------
// export {
template <typename F, int T>
concept StateSpaceFunction = requires(F func, Vector<T> point, double time) {
  { func(point, time) } -> std::same_as<StateDerivativesPoint<T>>;
};

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

template <int T>
std::ostream& operator<<(std::ostream& stream,
                         const optimization::Vector<T>& state) {
  for (int i{0}; i < T; ++i) {
    stream << state[i] << " ";
  }
  return stream;
}
