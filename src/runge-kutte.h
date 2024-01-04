#pragma once

#include "global.h"
#include "tensor.h"
#include "utils.h"

#include <vector>
namespace optimization {

template <typename F, typename T, class Alloc>
concept RungeKutteSaveFun = requires(F fun, Tensor<T, Alloc> x, double t) {
  { fun(x, t) };
};

template <typename T, class Alloc>
void emptySave(const Tensor<T, Alloc>&, double) {
}

template <typename T, class Alloc, StateSpaceFunctionAll<T, Alloc> F,
          RungeKutteSaveFun<T, Alloc> SaveF>
Tensor<T, Alloc> rungeKutteStep(
    double startT, const Tensor<T, Alloc>& startX, F fun, double interestT,
    double delta = 0.001, SaveF save = [](const Tensor<T, Alloc>&, double) {}) {
  double curT{startT};
  auto curX{startX};

  const auto delta2{delta / 2};
  const auto delta6{delta / 6};

  thread_local static auto k = Tensor<T, Alloc>(curX.size());
  thread_local static auto tmp{k};
  while (curT < interestT) {
    // bringing outside and preallocating does nothing
    if constexpr (CallableTwoArgsPreallocatedResult<
                      decltype(fun), decltype(curX), double, T, Alloc>) {
      k = Tensor<T, Alloc>(curX.size());
      fun(curX, curT, k);
      tmp = k;
      fun(curX + delta2 * k, curT + delta2, k);
      tmp += 2 * k;
      fun(curX + delta2 * k, curT + delta2, k);
      tmp += 2 * k;
      fun(curX + delta * k, curT + delta, k);
      tmp += k;
      curX += delta6 * tmp;
      curT += delta;
      save(curX, curT);
    } else {
      k = fun(curX, curT);
      tmp = k;
      k = fun(curX + delta2 * k, curT + delta2);
      tmp += 2 * k;
      k = fun(curX + delta2 * k, curT + delta2);
      tmp += 2 * k;
      k = fun(curX + delta * k, curT + delta);
      tmp += k;
      curX += delta6 * tmp;
      curT += delta;
      save(curX, curT);
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

  auto addPointToResult = [&result](const Tensor<T, Alloc>& point,
                                    double curT) -> void {
    const auto lastIndex{result.size() - 1};
    for (std::size_t i{0}; i < lastIndex; ++i) {
      result[i].push_back(point[i]);
    }
    result[lastIndex].push_back(curT);
  };

  addPointToResult(startX, startT);

  rungeKutteStep(startT, startX,
                 std::function<Tensor<T, Alloc>(Tensor<T, Alloc>, double)>(
                     fun),  // std::function<Tensor<3>(Tensor<3>, double)
                 lastT - kEps, delta, addPointToResult);

  return result;
}
}  // namespace optimization
