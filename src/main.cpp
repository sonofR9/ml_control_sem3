// #include "gradient-descent.h"
// #include "runge-kutte.h"
// #include "two-wheel-robot.h"

#include "evolution-optimization.h"
#include "global.h"
#include "model.h"
#include "options.h"
#include "particle-sworm.h"
// #include "pontryagin-method.h"
#include "tensor.h"

#include <cassert>
#include <chrono>
#include <fstream>

namespace optimization {
unsigned int seed = 50;
}

template <class Alloc, class VectorAlloc>
void writeTrajectoryToFiles(
    const std::vector<std::vector<double, Alloc>, VectorAlloc>& trajectory) {
  assert((trajectory.size() == 4));

  std::ofstream fileX("trajectory_x.txt");
  std::ofstream fileY("trajectory_y.txt");

  if (!fileX.is_open() || !fileY.is_open()) {
    throw std::runtime_error("Error opening files");
  }

  for (size_t i = 0; i < trajectory[0].size(); ++i) {
    fileX << trajectory[0][i] << "\n";
    fileY << trajectory[1][i] << "\n";
  }

  fileX.close();
  fileY.close();
}

using namespace optimization;

template <class Alloc>
void modelTestEvolution(std::size_t paramsCount, int iters, double tMax,
                        double dt) {
  auto start = std::chrono::high_resolution_clock::now();

  using namespace two_wheeled_robot;
  const auto adap = [paramsCount, tMax,
                     dt](const Tensor<double, Alloc>& solverResult) {
    assert((solverResult.size() == 2 * paramsCount));
    return functional<double, Alloc>(solverResult, tMax, dt);
  };
  Evolution<1000, 1000, Alloc, decltype(adap), 500> solver(
      adap, 2 * paramsCount, -10, 10);
  const auto best{solver.solve(iters)};
  std::cout << "model: [" << best
            << "] functional: " << functional<double, Alloc>(best, tMax, dt)
            << "\n";

  const auto trajectory{getTrajectoryFromControl<double, Alloc>(best, tMax)};
  writeTrajectoryToFiles(trajectory);

  auto end = std::chrono::high_resolution_clock::now();

#ifdef NDEBUG
  std::cout << "Release build\n";
#else
  std::cout << "Debug build\n";
#endif
  std::cout << "Time of excecution Evolution: " << (end - start).count() / 1e9
            << " s\n";
}

template <class Alloc>
void modelTestGrey(std::size_t paramsCount, int iters, double tMax, double dt) {
  auto start = std::chrono::high_resolution_clock::now();

  using namespace two_wheeled_robot;
  const auto adap = [paramsCount, tMax,
                     dt](const Tensor<double, Alloc>& solverResult) {
    assert((solverResult.size() == 2 * paramsCount));
    return functional<double, Alloc>(solverResult, tMax, dt);
  };
  GrayWolfAlgorithm<Alloc, decltype(adap), 512, 3> solver(adap, 2 * paramsCount,
                                                          10);
  const auto best{solver.solve(iters)};
  std::cout << "model: [" << best
            << "] functional: " << functional<double, Alloc>(best, tMax, dt)
            << "\n";

  const auto trajectory{getTrajectoryFromControl<double, Alloc>(best, tMax)};
  writeTrajectoryToFiles(trajectory);

  auto end = std::chrono::high_resolution_clock::now();

#ifdef NDEBUG
  std::cout << "Release build\n";
#else
  std::cout << "Debug build\n";
#endif
  std::cout << "Time of excecution gray wolf: " << (end - start).count() / 1e9
            << " s\n";
}

int main(int argc, const char** argv) try {
  const auto& options{optimization::parseOptions(argc, argv)};
  const double tMax{options.tMax};
  const double dt{options.integrationDt};
  const int iter{options.iter};
  seed = options.seed;
  std::size_t paramsCount{options.controlOptions.numOfParams};

  switch (options.method) {
  case optimization::GlobalOptions::Method::kEvolution:
    modelTestEvolution<std::allocator<double>>(paramsCount, iter, tMax, dt);
  case optimization::GlobalOptions::Method::kGrayWolf:
    modelTestGrey<std::allocator<double>>(paramsCount, iter, tMax, dt);
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << "\n";
  return 1;
} catch (...) {
  return 1;
}
