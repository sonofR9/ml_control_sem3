#pragma once

#include "tensor.h"

#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>

namespace optimization {
// constexpr double kEps = 1e-10;

// ----------------------------- state point etc ------------------------------
template <typename T, class Alloc>
using StatePoint = Tensor<T, Alloc>;

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <typename T, class Alloc>
using StateDerivativesPoint = Tensor<T, Alloc>;

template <typename F, typename T, class Alloc>
concept StateSpaceFunction =
    requires(F func, StatePoint<T, Alloc> point, double time) {
      { func(point, time) } -> std::same_as<StateDerivativesPoint<T, Alloc>>;
    };

template <typename F, typename T, class Alloc>
concept StateSpaceFunctionPreallocated =
    requires(F func, StatePoint<T, Alloc> point, double time,
             StateDerivativesPoint<T, Alloc>& result) {
      { func(point, time, result) } -> std::same_as<void>;
    };

template <typename F, typename T, class Alloc>
concept StateSpaceFunctionAll = StateSpaceFunction<F, T, Alloc> ||
                                StateSpaceFunctionPreallocated<F, T, Alloc>;

template <typename F, typename T>
concept GradientFunction = requires(F func, const T& inp) {
  { func(inp) } -> std::same_as<T>;
};

template <typename F, typename T>
concept Regular1OutFunction = requires(F func, const T& inp) {
  { func(inp) } -> std::same_as<double>;
};

// ----------------------------- print ---------------------------------------
template <typename F>
concept PrintFunction =
    requires(F func, std::size_t iteration, double functional) {
      { func(iteration, functional) } -> std::same_as<void>;
    };

inline void coutPrint(std::size_t iteration, double functional) {
  std::cout << std::format("\33[2K\riter {} functional {:.5f}", iteration,
                           functional)
            << std::flush;
}

// ----------------------------- norm ---------------------------------------
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

// ----------------------- Preallocated callable -----------------------------
template <typename F, typename U1, typename T, class Alloc>
concept CallableOneArgPreallocatedResult =
    requires(F fun, U1 inp, Tensor<T, Alloc>& preallocatedResult) {
      { fun(inp, preallocatedResult) } -> std::same_as<void>;
    };

template <typename F, typename U1, typename U2, typename T, class Alloc>
concept CallableTwoArgsPreallocatedResult =
    requires(F fun, U1 inp1, U2 inp2, Tensor<T, Alloc>& preallocatedResult) {
      { fun(inp1, inp2, preallocatedResult) } -> std::same_as<void>;
    };

// --------------------------random numbers----------------------------------
struct SharedGenerator {
  static std::mt19937 gen;
};

struct DoubleGenerator {
  static double get() {
    static std::uniform_real_distribution<> dis(-100, 100);
    return dis(SharedGenerator::gen);
  }

  static double absLimit() {
    return 100;
  }
};

template <uint64_t D>
struct IntGenerator {
  static int get() {
    static std::uniform_int_distribution<> dis(0, D - 1);
    return dis(SharedGenerator::gen);
  }
};

struct VaryingIntGenerator {
  static int get(int min, int max) {
    std::uniform_int_distribution<> dis(min, max);
    return dis(SharedGenerator::gen);
  }
};

struct Probability {
  static double get() {
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(SharedGenerator::gen);
  }
};
}  // namespace optimization
