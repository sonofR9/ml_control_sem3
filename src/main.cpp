#include "runge-kutte.h"
#include "two-wheel-robot.h"
// import runge_kutte1;
#include "pontryagin-method.h"
#include "global.h"

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace optimization;

template <int N, StateSpaceFunction<N> MS, ConjugateFunction<N, N> CS,
          FindMaximumFunction<N, 2> FM>
using PontryaginSolver = decltype(SolveUsingPontryagin<2, N, MS, CS, FM>);

int main() {
  // struct Lambda {
  //  public:
  //   Lambda(int a, int& b) : a{a}, b{b} {
  //   }

  //   double operator()(double t) const {
  //     return 1;
  //   }

  //  private:
  //   int a;
  //   int& b;
  // };
  // int a = 0;
  // int b = a;
  // auto u1 = [a, &b](double t) mutable -> double {
  //   a = 4;
  //   b = 5;
  //   return 1;
  // };

  // auto u2 = [](double t) -> double { return 0; };

  auto u = [](double t) -> optimization::StatePoint<2> { return {1, 0}; };
  constexpr double umin{-1};
  constexpr double umax{1};

  two_wheeled_robot::Model robot(u, 2, 1);
  auto conjugate = [](const Vector<3>& x, const Vector<2>& u,
                      const Vector<3>& psi,
                      double time) -> StateDerivativesPoint<3> {
    return {0, 0,
            -psi[0] * sin(x[2]) * (u[0] + u[1]) / 2 +
                psi[1] * cos(x[2]) * (u[0] + u[1]) / 2};
  };
  auto findMaximum = [](const Vector<3>& x, const Vector<3>& psi,
                        double) -> Vector<2> {
    double ul;
    double ur;
    if (psi[0] * cos(x[2]) + psi[1] * sin(x[2]) + psi[2] > 0) {
      ul = umax;
    } else {
      ul = umin;
    }
    if (psi[0] * cos(x[2]) + psi[1] * sin(x[2]) - psi[2] > 0) {
      ur = umax;
    } else {
      ur = umin;
    }
    return {ul, ur};
  };

  double delta{0.01};
  Vector<3> x0{0, 0, 0};
  Vector<3> psi0{
      0,
      0,
  };
  Vector<3> xf{0, 0, 0};
  double curT{0};
  // double endT{100};
  // const auto solvedFun = SolveDiffEqRungeKutte(curT, curX, robot, endT,
  // delta);

  const auto solvedFcn =
      PontryaginSolver(x0, psi0, robot, conjugate, findMaximum, xf);

  plt::figure();
  // plt::plot(solvedFun[0], solvedFun[1]);
  plt::show();

  return 0;
}
