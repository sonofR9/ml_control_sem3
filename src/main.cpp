#include "gradient-descent.h"
#include "runge-kutte.h"
#include "two-wheel-robot.h"
// import runge_kutte1;
#include "evolution-optimization.h"
#include "global.h"
#include "model.h"
#include "particle-sworm.h"
#include "pontryagin-method.h"

#include <chrono>
#include <cmath>
#include <fstream>

#ifdef MATPLOTLIB
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

namespace optimization {
int seed = 50;
}

void writeTrajectoryToFiles(
    const std::array<std::vector<double>, 4>& trajectory) {
  std::ofstream fileX("trajectory_x.txt");
  std::ofstream fileY("trajectory_y.txt");

  if (!fileX.is_open() || !fileY.is_open()) {
    throw std::runtime_error("Error opening files");
  }

  for (size_t i = 0; i < trajectory[0].size(); ++i) {
    fileX << trajectory[0][i] << std::endl;
    fileY << trajectory[1][i] << std::endl;
  }

  fileX.close();
  fileY.close();
}

using namespace optimization;

template <int N, StateSpaceFunction<N> MS, ConjugateFunction<N, N> CS,
          FindMaximumFunction<N, 2> FM>
using PontryaginSolver = decltype(SolveUsingPontryagin<2, N, MS, CS, FM>);

void testGradientDescent() {
  const Tensor<5> qMin{-3, -3, -3, -3, -3};
  const Tensor<5> qMax{30, 30, 30, 30, 30};
  // func = (q0-5)^2 + (q1)^2 + (q2)^2 + (q3-5)^2 + (q4-10)^2
  auto grad = [](const Tensor<5>& q) -> Tensor<5> {
    return {2 * (q[0] - 5), 2 * q[1], 2 * q[2], 2 * (q[3] - 5),
            2 * (q[4] - 10)};
  };
  auto functional = [](const Tensor<5>& q) -> double {
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
  // auto conjugate = [](const Tensor<3>& x, const Tensor<2>& u,
  //                     const Tensor<3>& psi,
  //                     double time) -> StateDerivativesPoint<3> {
  //   return {0, 0,
  //           -psi[0] * sin(x[2]) * (u[0] + u[1]) / 2 +
  //               psi[1] * cos(x[2]) * (u[0] + u[1]) / 2};
  // };
  // auto findMaximum = [](const Tensor<3>& x, const Tensor<3>& psi,
  //                       double) -> Tensor<2> {
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
  // Tensor<3> x0{0, 0, 0};
  // Tensor<3> psi0{0, 0, 0};
  // Tensor<3> xf{0, 0, 0};
  // double curT{0};
  // double endT{100};
  // const auto solvedFun = SolveDiffEqRungeKutte(curT, curX, robot, endT,
  // delta);

  // const auto solvedFcn =
  // PontryaginSolver(x0, psi0, robot, conjugate, findMaximum, xf);
}

void testEvolution() {
  auto fitness = [](const Tensor<5>& q) -> double {
    return 1 + std::pow(q[0] - 5, 2) + std::pow(q[1], 2) + std::pow(q[2], 2) +
           std::pow(q[3] - 5, 2) + std::pow(q[4] - 10, 2);
  };
  Evolution<5, 1000, 1000, decltype(fitness), 100> solver(fitness, -100, 100);
  const auto best{solver.solve(200)};
  std::cout << "Evolution: [" << best << "] True: [5 0 0 5 10]" << std::endl;
}

void testParticle() {
  auto fitness = [](const Tensor<5>& q) -> double {
    return 1 + std::pow(q[0] - 5, 2) + std::pow(q[1], 2) + std::pow(q[2], 2) +
           std::pow(q[3] - 5, 2) + std::pow(q[4] - 10, 2);
  };
  GrayWolfAlgorithm<5, decltype(fitness), 500, 5> solver(fitness, 100);
  const auto best{solver.solve(800)};
  std::cout << "Gray wolf: [" << best << "] True: [5 0 0 5 10]" << std::endl;
}

template <int N, int P = 10 * N>
void testGraySin(int iter) {
  auto fitness = [](const Tensor<1>& q) -> double {
    return 1 + std::sin(q[0]);
  };
  GrayWolfAlgorithm<1, decltype(fitness), P, N> solver(fitness, 100);
  const auto best{solver.solve(iter)};
  std::cout << "<" << N << ">, ", iter,
      ": " << best << " fit " << fitness(best) << std::endl;
}

void testGraySin() {
  std::cout << "gray with different number of best wolfs\n <wolfs> iters: "
               "best_val fit fitness_val";
}

template <int N>
void modelTestEvolution(int iters, double tMax, double dt) {
  auto start = std::chrono::high_resolution_clock::now();

  using namespace two_wheeled_robot;
  const auto adap = [tMax, dt](const Tensor<2 * N, double>& solverResult) {
    return functional<N>(solverResult, tMax, dt);
  };
  Evolution<2 * N, 1000, 1000, decltype(adap), 500> solver(adap, -10, 10);
  const auto best{solver.solve(iters)};
  std::cout << "model: [" << best
            << "] functional: " << functional<N>(best, tMax, dt) << std::endl;

  const auto trajectory{getTrajectoryFromControl<N>(best, tMax)};
  // std::cout << "--------------------\n\n\n";
  // for (std::size_t i{0}; i < trajectory[0].size(); ++i) {
  //   std::cout << "x: " << trajectory[0][i] << " y: " << trajectory[1][i]
  //             << "\n";
  // }
  writeTrajectoryToFiles(trajectory);

  auto end = std::chrono::high_resolution_clock::now();

#ifdef NDEBUG
  std::cout << "Release build" << std::endl;
#else
  std::cout << "Debug build" << std::endl;
#endif
  std::cout << "Time of excecution Evolution: " << (end - start).count() / 1e9
            << " s" << std::endl;
}

template <int N>
void modelTestGrey(int iters, double tMax, double dt) {
  auto start = std::chrono::high_resolution_clock::now();

  using namespace two_wheeled_robot;
  const auto adap = [tMax, dt](const Tensor<2 * N, double>& solverResult) {
    return functional<N>(solverResult, tMax, dt);
  };
  GrayWolfAlgorithm<2 * N, decltype(adap), 512, 3> solver(adap, 10);
  const auto best{solver.solve(iters)};
  std::cout << "model: [" << best
            << "] functional: " << functional<N>(best, tMax, dt) << std::endl;

  const auto trajectory{getTrajectoryFromControl<N>(best, tMax)};
  // std::cout << "--------------------\n\n\n";
  // for (std::size_t i{0}; i < trajectory[0].size(); ++i) {
  //   std::cout << "x: " << trajectory[0][i] << " y: " << trajectory[1][i]
  //             << "\n";
  // }
  writeTrajectoryToFiles(trajectory);

  auto end = std::chrono::high_resolution_clock::now();

#ifdef NDEBUG
  std::cout << "Release build" << std::endl;
#else
  std::cout << "Debug build" << std::endl;
#endif
  std::cout << "Time of excecution gray wolf: " << (end - start).count() / 1e9
            << " s" << std::endl;
}

int main(int argc, char** argv) {
  int iter{500};
  double tMax{100};
  double dt{0.1};
  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};
    if (arg == "--seed") {
      if (i + 1 < argc) {
        try {
          seed = std::stoi(argv[i + 1]);
          std::cout << "seed provided: " << seed << std::endl;
        } catch (std::invalid_argument& e) {
          std::cerr << "Error: Invalid seed value provided." << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: Missing seed value after --seed flag."
                  << std::endl;
        return 1;
      }
    } else if (arg == "--iter") {
      if (i + 1 < argc) {
        try {
          iter = std::stoi(argv[i + 1]);
        } catch (std::invalid_argument& e) {
          std::cerr << "Error: Invalid iter value provided." << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: Missing iter value after --iter flag."
                  << std::endl;
        return 1;
      }
    } else if (arg == "--tmax") {
      if (i + 1 < argc) {
        try {
          tMax = std::stod(argv[i + 1]);
        } catch (std::invalid_argument& e) {
          std::cerr << "Error: Invalid tMax value provided." << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: Missing tMax value after --tMax flag."
                  << std::endl;
        return 1;
      }
    } else if (arg == "--dt") {
      if (i + 1 < argc) {
        try {
          dt = std::stod(argv[i + 1]);
        } catch (std::invalid_argument& e) {
          std::cerr << "Error: Invalid dt value provided." << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: Missing dt value after --dt flag." << std::endl;
        return 1;
      }
    } else if (arg == "-help") {
      std::cout << "Usage: program --seed <seed_value> --iter <iter_value>\n"
                   "Options:\n"
                   "     --seed <seed_value>: Specify a seed value for the "
                   "random number generator.\n"
                   "     --iter <iter_value>: Specify number of iterations for "
                   "model optimization algorithm.\n"
                   "     --tmax <tmax_value>: Specify the maximum time value "
                   "for the model.\n"
                   "     --dt <dt_value>: Specify the time step value for the "
                   "model.\n"
                   "     --help: Display this help message."
                << std::endl;
      return 0;
    }
  }

  // testPontryagin();
  // testGradientDescent();
  // testEvolution();
  // testParticle();
  modelTestEvolution<100>(iter, tMax, dt);
  // modelTestGrey<20>(iter, tMax, dt);
  // plt::figure();
  // plt::plot(solvedFun[0], solvedFun[1]);
  // plt::show();

  return 0;
}
