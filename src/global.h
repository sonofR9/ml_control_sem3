/* Copyright (C) 2023-2024 Novak Alexander
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    requires(F fun, U1 inp,
             Tensor<std::remove_cvref_t<T>, Alloc>& preallocatedResult) {
      { fun(inp, preallocatedResult) } -> std::same_as<void>;
    } &&
    !std::is_invocable_v<F, U1, Tensor<std::remove_cvref_t<T>, Alloc>&&>;

template <typename F, typename U1, typename U2, typename T, class Alloc>
concept CallableTwoArgsPreallocatedResult =
    requires(F fun, U1 inp1, U2 inp2,
             Tensor<std::remove_cvref_t<T>, Alloc>& preallocatedResult) {
      { fun(inp1, inp2, preallocatedResult) } -> std::same_as<void>;
    } &&
    !std::is_invocable_v<F, U1, U2, Tensor<std::remove_cvref_t<T>, Alloc>&&>;

template <typename F, typename T, typename R, typename... U>
concept CallableNArgsPreallocatedResult =
    requires(F fun, std::remove_cv_t<T>& preallocatedResult, U... u) {
      { fun(u..., preallocatedResult) } -> std::same_as<R>;
    } && !std::is_invocable_v<F, U..., std::remove_cvref_t<T>&&>;

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
