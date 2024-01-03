#pragma once

#include "global.h"
#include "tensor.h"

#include <vector>
namespace optimization {
template <typename T, class Alloc, StateSpaceFunctionAll<T, Alloc> F>
Tensor<T, Alloc> rungeKutteStep(double startT, const Tensor<T, Alloc>& startX,
                                F fun, double interestT, double delta = 0.001) {
  double curT{startT};
  auto curX{startX};

  const auto delta2{delta / 2};
  const auto delta6{delta / 6};

  // thread_local static auto k = Tensor<T, Alloc>(curX.size());
  // thread_local static auto tmp{k};
  while (curT < interestT) {
    // bringing outside and preallocating does nothing
    if constexpr (CallableTwoArgsPreallocatedResult<
                      decltype(fun), decltype(curX), double, T, Alloc>) {
      auto k = Tensor<T, Alloc>(curX.size());
      fun(curX, curT, k);
      auto tmp{k};
      fun(curX + delta2 * k, curT + delta2, k);
      tmp += 2 * k;
      fun(curX + delta2 * k, curT + delta2, k);
      tmp += 2 * k;
      fun(curX + delta * k, curT + delta, k);
      tmp += k;
      curX += delta6 * tmp;
      curT += delta;
    } else {
      auto k = fun(curX, curT);
      auto tmp{k};
      k = fun(curX + delta2 * k, curT + delta2);
      tmp += 2 * k;
      k = fun(curX + delta2 * k, curT + delta2);
      tmp += 2 * k;
      k = fun(curX + delta * k, curT + delta);
      tmp += k;
      curX += delta6 * tmp;
      curT += delta;
    }
  }
  return curX;
}

/**
 * @tparam T type
 * @return std::vector<std::vector<T>> array of vectors with values of each
 * state coordinate and time. Given input =Tensor with shape (N...), output
 * will have shape (N+1, M) where M is the number of steps
 */
template <typename T, class Alloc, StateSpaceFunctionAll<T, Alloc> F,
          class VectorAlloc = std::allocator<std::vector<T, Alloc>>>
std::vector<std::vector<T, Alloc>, VectorAlloc> solveDiffEqRungeKutte(
    double startT, const Tensor<T, Alloc>& startX, F fun, double lastT,
    double delta = 0.001) {
  auto result =
      std::vector<std::vector<T, Alloc>, VectorAlloc>(startX.size() + 1);
  for (auto& vec : result) {
    vec.reserve(static_cast<std::size_t>((lastT - startT) / delta));
  }
  double curT{startT};
  Tensor<T, Alloc> curX{startX};

  auto addPointToResult = [&result](const Tensor<T, Alloc>& point,
                                    double curT) -> void {
    for (std::size_t i{0}; i < result.size() - 1; ++i) {
      result[i].push_back(point[i]);
    }
    result[result.size() - 1].push_back(curT);
  };

  addPointToResult(curX, curT);

  while (curT < lastT - kEps) {
    curX = rungeKutteStep(
        curT, curX,
        std::function<Tensor<T, Alloc>(Tensor<T, Alloc>, double)>(
            fun),  // std::function<Tensor<3>(Tensor<3>, double)
        curT + delta, delta);
    curT += delta;
    addPointToResult(curX, curT);
  }

  return result;
}
}  // namespace optimization
