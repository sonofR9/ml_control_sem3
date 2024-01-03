#pragma once

#include "tensor.h"

#include <functional>
#include <numeric>
#include <random>

namespace optimization {
// constexpr double kEps = 1e-10;

template <typename T, class Alloc>
using StatePoint = Tensor<T, Alloc>;

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <typename T, class Alloc>
using StateDerivativesPoint = Tensor<T, Alloc>;

extern unsigned int seed;

template <typename F, typename T, class Alloc>
concept StateSpaceFunction =
    requires(F func, StatePoint<T, Alloc> point, double time) {
      { func(point, time) } -> std::same_as<StateDerivativesPoint<T, Alloc>>;
    };

template <typename F, typename T>
concept GradientFunction = requires(F func, const T& inp) {
  { func(inp) } -> std::same_as<T>;
};

template <typename F, typename T>
concept Regular1OutFunction = requires(F func, const T& inp) {
  { func(inp) } -> std::same_as<double>;
};

template <typename T>
concept Iterable = requires(const T& self) {
  { self.begin() } -> std::convertible_to<typename T::iterator>;
  { self.end() } -> std::convertible_to<typename T::iterator>;
};

double norm(const Iterable auto&& self) {
  return std::sqrt(std::transform_reduce(
      self.begin(), self.end(), 0.0, std::plus<>(),
      [](const auto& val) -> double { return val * val; }));
}

// --------------------------random numbers----------------------------------
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
