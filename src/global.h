#pragma once

#include "static-tensor.h"

#include <functional>
#include <numeric>
#include <random>

namespace optimization {
// constexpr double kEps = 1e-10;

template <int N>
using StatePoint = StaticTensor<N>;

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <int N>
using StateDerivativesPoint = StaticTensor<N>;

template <int N, typename T>
double norm(StaticTensor<N, T> self) {
  return std::sqrt(
      std::transform_reduce(self.begin(), self.end(), 0.0, std::plus<>(),
                            [](const T& val) { return val * val; }));
}

double norm(StaticTensor<100, double> self) {
  return std::sqrt(std::transform_reduce(self.begin(), self.end(), 0.0,
                                         std::plus{},
                                         [](double val) { return val * val; }));
}

extern unsigned int seed;

template <typename F, int N>
concept StateSpaceFunction =
    requires(F func, StaticTensor<N> point, double time) {
      { func(point, time) } -> std::same_as<StateDerivativesPoint<N>>;
    };

template <typename F, typename T>
concept GradientFunction = requires(F func, const T& inp) {
  { func(inp) } -> std::same_as<T>;
};

template <typename F, typename T>
concept Regular1OutFunction = requires(F func, const T& inp) {
  { func(inp) } -> std::same_as<double>;
};

struct DoubleGenerator {
  static double get() {
    static std::mt19937 gen(seed);
    static std::uniform_real_distribution<> dis(-100, 100);
    return dis(gen);
  }

  static double absLimit() {
    return 100;
  }
};

template <uint64_t D>
struct IntGenerator {
  static int get() {
    static std::mt19937 gen(seed);
    static std::uniform_int_distribution<> dis(0, D - 1);
    return dis(gen);
  }
};

struct Probability {
  static double get() {
    static std::mt19937 gen(seed);
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
  }
};
}  // namespace optimization
