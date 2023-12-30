#include "global.h"
#include "runge-kutte.h"

#include <concepts>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>


namespace optimization {
class ControlFunction {
 public:
  double operator()(double t) {
    return u_.lower_bound(t)->second;
  }

  void Insert(double t, double val) {
    u_.insert({t, val});
  }

 private:
  std::map<double, double> u_;
};

template <typename F, int N, int M>
concept FindMaximumFunction =
    requires(F func, const Tensor<N>& x, const Tensor<N>& psi, double time) {
      { func(x, psi, time) } -> std::same_as<Tensor<M>>;
    };

template <typename F, int N, int M>
concept ConjugateFunction =
    requires(F func, const Tensor<N>& x, const Tensor<M>& u,
             const Tensor<N>& psi, double time) {
      { func(x, u, psi, time) } -> std::same_as<StateDerivativesPoint<N>>;
    };

template <int M, int N, StateSpaceFunction<N> MS, ConjugateFunction<N, N> CS,
          FindMaximumFunction<N, M> FM>
ControlFunction SolveUsingPontryagin(const Tensor<N>& x0, const Tensor<N>& psi0,
                                     MS main, CS conjugate, FM findMaximum,
                                     const Tensor<N>& xf) {
  ControlFunction result;

  double dt{1e-2};

  double t{0};
  Tensor<M> u;
  auto x{x0};
  auto psi{psi0};
  while (x != xf) {
    u = findMaximum(x, psi, t);
    psi = RungeKutteStep(t, psi, std::bind_front(conjugate, x, u), t + dt, dt);
    x = RungeKutteStep(t, x, main, t + dt, dt);
    result.Insert(t, u);
  }
  return result;
}

}  // namespace optimization
