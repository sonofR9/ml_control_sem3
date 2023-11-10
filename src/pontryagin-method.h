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
concept FindMaximumFunction = requires(F func, const Vector<N>& x,
                                       const Vector<N>& psi, double time) {
  { func(x, psi, time) } -> std::same_as<Vector<M>>;
};

template <typename F, int N, int M>
concept ConjugateFunction = requires(F func, const Vector<N>& x,
                                     const Vector<M>& u, const Vector<N>& psi,
                                     double time) {
  { func(x, u, psi, time) } -> std::same_as<StateDerivativesPoint<N>>;
};

template <int M, int N, StateSpaceFunction<N> MS, ConjugateFunction<N, N> CS,
          FindMaximumFunction<N, M> FM>
ControlFunction SolveUsingPontryagin(const Vector<N>& x0, const Vector<N>& psi0,
                                     MS main, CS conjugate, FM findMaximum,
                                     const Vector<N>& xf) {
  ControlFunction result;

  double dt{1e-2};

  double t{0};
  Vector<M> u;
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
