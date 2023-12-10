#include "gradient-descent.h"
#include "runge-kutte.h"
#include "two-wheel-robot.h"
// import runge_kutte1;
#include "evolution-optimization.h"
#include "global.h"
#include "particle-sworm.h"
#include "pontryagin-method.h"

#include <cmath>

#ifdef MATPLOTLIB
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

using namespace optimization;

template <int N, StateSpaceFunction<N> MS, ConjugateFunction<N, N> CS,
          FindMaximumFunction<N, 2> FM>
using PontryaginSolver = decltype(SolveUsingPontryagin<2, N, MS, CS, FM>);

void testGradientDescent() {
  const Vector<5> qMin{-3, -3, -3, -3, -3};
  const Vector<5> qMax{30, 30, 30, 30, 30};
  // func = (q0-5)^2 + (q1)^2 + (q2)^2 + (q3-5)^2 + (q4-10)^2
  auto grad = [](const Vector<5>& q) -> Vector<5> {
    return {2 * (q[0] - 5), 2 * q[1], 2 * q[2], 2 * (q[3] - 5),
            2 * (q[4] - 10)};
  };
  auto functional = [](const Vector<5>& q) -> double {
    return std::pow(q[0] - 5, 2) + std::pow(q[1], 2) + std::pow(q[2], 2) +
           std::pow(q[3] - 5, 2) + std::pow(q[4] - 10, 2);
  };
  const auto& res{GradientDescent(qMin, qMax, functional, grad, 0.5, 1e4)};
  std::cout << "GradientDescent: [" << res << "] True: [5 0 0 5 10]"
            << std::endl;
}

void testPontryagin() {
  // auto u = [](double t) -> optimization::StatePoint<2> { return {1, 0}; };
  // constexpr double umin{-1};
  // constexpr double umax{1};

  // two_wheeled_robot::Model robot(u, 2, 1);
  // auto conjugate = [](const Vector<3>& x, const Vector<2>& u,
  //                     const Vector<3>& psi,
  //                     double time) -> StateDerivativesPoint<3> {
  //   return {0, 0,
  //           -psi[0] * sin(x[2]) * (u[0] + u[1]) / 2 +
  //               psi[1] * cos(x[2]) * (u[0] + u[1]) / 2};
  // };
  // auto findMaximum = [](const Vector<3>& x, const Vector<3>& psi,
  //                       double) -> Vector<2> {
  //   double ul;
  //   double ur;
  //   if (psi[0] * cos(x[2]) + psi[1] * sin(x[2]) + psi[2] > 0) {
  //     ul = umax;
  //   } else {
  //     ul = umin;
  //   }
  //   if (psi[0] * cos(x[2]) + psi[1] * sin(x[2]) - psi[2] > 0) {
  //     ur = umax;
  //   } else {
  //     ur = umin;
  //   }
  //   return {ul, ur};
  // };

  // double delta{0.01};
  // Vector<3> x0{0, 0, 0};
  // Vector<3> psi0{0, 0, 0};
  // Vector<3> xf{0, 0, 0};
  // double curT{0};
  // double endT{100};
  // const auto solvedFun = SolveDiffEqRungeKutte(curT, curX, robot, endT,
  // delta);

  // const auto solvedFcn =
  // PontryaginSolver(x0, psi0, robot, conjugate, findMaximum, xf);
}

void testEvolution() {
  auto fitness = [](const Vector<5>& q) -> double {
    return 1 + std::pow(q[0] - 5, 2) + std::pow(q[1], 2) + std::pow(q[2], 2) +
           std::pow(q[3] - 5, 2) + std::pow(q[4] - 10, 2);
  };
  Evolution<5, 1000, 1000, decltype(fitness), 100> solver(fitness, -100, 100);
  const auto best{solver.solve(200)};
  std::cout << "Evolution: [" << best << "] True: [5 0 0 5 10]" << std::endl;
}

void testParticle() {
  auto fitness = [](const Vector<5>& q) -> double {
    return 1 + std::pow(q[0] - 5, 2) + std::pow(q[1], 2) + std::pow(q[2], 2) +
           std::pow(q[3] - 5, 2) + std::pow(q[4] - 10, 2);
  };
  GrayWolfAlgorithm<5, decltype(fitness), 500, 5> solver(fitness, 100);
  const auto best{solver.solve(800)};
  std::cout << "Gray wolf: [" << best << "] True: [5 0 0 5 10]" << std::endl;
}

void modelTestEvolution() {
  // auto u = [](double t) -> optimization::StatePoint<2> { return {1, 0}; };
  // constexpr double umin{-1};
  // constexpr double umax{1};

  // two_wheeled_robot::Model robot(u, 2, 1);
  // auto conjugate = [](const Vector<3>& x, const Vector<2>& u,
  //                     const Vector<3>& psi,
  //                     double time) -> StateDerivativesPoint<3> {
  //   return {0, 0,
  //           -psi[0] * sin(x[2]) * (u[0] + u[1]) / 2 +
  //               psi[1] * cos(x[2]) * (u[0] + u[1]) / 2};
  // };
  // auto findMaximum = [](const Vector<3>& x, const Vector<3>& psi,
  //                       double) -> Vector<2> {
  //   double ul;
  //   double ur;
  //   if (psi[0] * cos(x[2]) + psi[1] * sin(x[2]) + psi[2] > 0) {
  //     ul = umax;
  //   } else {
  //     ul = umin;
  //   }
  //   if (psi[0] * cos(x[2]) + psi[1] * sin(x[2]) - psi[2] > 0) {
  //     ur = umax;
  //   } else {
  //     ur = umin;
  //   }
  //   return {ul, ur};
  // };

  // double delta{0.01};
  // Vector<3> x0{0, 0, 0};
  // Vector<3> psi0{0, 0, 0};
  // Vector<3> xf{0, 0, 0};
  // double curT{0};
  // double endT{100};
  // const auto solvedFun = SolveDiffEqRungeKutte(curT, curX, robot, endT,
  // delta);

  // const auto solvedFcn =
  // PontryaginSolver(x0, psi0, robot, conjugate, findMaximum, xf);
}

int main() {
  testPontryagin();
  testGradientDescent();
  testEvolution();
  testParticle();
  // plt::figure();
  // plt::plot(solvedFun[0], solvedFun[1]);
  // plt::show();

  return 0;
}
