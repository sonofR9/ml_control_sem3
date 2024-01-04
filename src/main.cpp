// #include "gradient-descent.h"
// #include "runge-kutte.h"
// #include "two-wheel-robot.h"
// #include "pontryagin-method.h"

#include "allocator.h"
#include "evolution-optimization.h"
#include "global.h"
#include "model.h"
#include "options.h"
#include "particle-sworm.h"
#include "tensor.h"
#include "utils.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <cassert>
#include <chrono>
#include <fstream>

namespace optimization {
unsigned int seed = 50;
}

constexpr double kMaxDiff{1};

class TimeMeasurer {
 public:
  explicit TimeMeasurer(std::string name)
      : start_{std::chrono::high_resolution_clock::now()},
        name_{std::move(name)} {
  }

  ~TimeMeasurer() {
    std::cout << "Execution time of " << name_ << ": "
              << 1e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start_)
                            .count()
              << " s\n";
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};

using namespace optimization;

template <class Alloc>
void modelTestEvolution(const optimization::GlobalOptions& options) {
  const double tMax{options.tMax};
  const double dt{options.integrationDt};
  const int iters{options.iter};
  std::size_t controlStepsCount{options.controlOptions.numOfParams};
  std::size_t paramsCount{controlStepsCount * 2};

  Tensor<double, Alloc> init{};
  if (!options.controlSaveFile.empty() && !options.clearSaveBeforeStart) {
    // create file if it does not exist
    { std::ofstream file(options.controlSaveFile, std::ofstream::app); }
    std::ifstream file(options.controlSaveFile);
    try {
      boost::archive::text_iarchive ia(file);
      ia >> init;
    } catch (const boost::archive::archive_exception& e) {
      std::cout << e.what() << "\n";
    }
    init.resize(paramsCount);
  }

  TimeMeasurer tm("Evolution");

  using namespace two_wheeled_robot;
  const auto adap = [paramsCount, tMax,
                     dt](const Tensor<double, Alloc>& solverResult) {
    assert((solverResult.size() == paramsCount));
    return functional<double, Alloc>(solverResult, tMax, dt);
  };
  Evolution<1000, 1000, Alloc, decltype(adap), 500> solver(
      adap, paramsCount,
      {.min = options.controlOptions.uMin, .max = options.controlOptions.uMax},
      {.mutation = options.evolutionOpt.mutationRate,
       .crossover = options.evolutionOpt.crossoverRate});
  if (!init.empty()) {
    solver.setBaseline(init, kMaxDiff);
  }
  const auto best{solver.solve(iters)};

  if (!options.controlSaveFile.empty()) {
    std::ofstream file(options.controlSaveFile,
                       std::ofstream::out | std::ofstream::trunc);
    boost::archive::text_oarchive oa(file);
    oa << best;
  }

  const auto trajectory{getTrajectoryFromControl<double, Alloc>(best, tMax)};
  writeTrajectoryToFiles(trajectory);
}

template <class Alloc>
void modelTestGray(const optimization::GlobalOptions& options) {
  const double tMax{options.tMax};
  const double dt{options.integrationDt};
  const int iters{options.iter};
  std::size_t controlStepsCount{options.controlOptions.numOfParams};
  std::size_t paramsCount{controlStepsCount * 2};

  Tensor<double, Alloc> init{};
  if (!options.controlSaveFile.empty() && !options.clearSaveBeforeStart) {
    // create file if it does not exist
    { std::ofstream file(options.controlSaveFile, std::ofstream::app); }
    std::ifstream file(options.controlSaveFile);
    try {
      boost::archive::text_iarchive ia(file);
      ia >> init;
    } catch (const boost::archive::archive_exception& e) {
      std::cout << e.what() << "\n";
    }
    init.resize(paramsCount);
  }

  TimeMeasurer tm("gray wolf");

  using namespace two_wheeled_robot;
  const auto adap = [paramsCount, tMax,
                     dt](const Tensor<double, Alloc>& solverResult) {
    assert((solverResult.size() == paramsCount));
    return functional<double, Alloc>(solverResult, tMax, dt);
  };
  GrayWolfAlgorithm<Alloc, decltype(adap), 512, 3> solver(adap, paramsCount,
                                                          10);
  if (!init.empty()) {
    solver.setBaseline(init, kMaxDiff);
  }
  const auto best{solver.solve(iters)};

  if (!options.controlSaveFile.empty()) {
    std::ofstream file(options.controlSaveFile,
                       std::ofstream::out | std::ofstream::trunc);
    boost::archive::text_oarchive oa(file);
    oa << best;
  }

  const auto trajectory{getTrajectoryFromControl<double, Alloc>(best, tMax)};
  writeTrajectoryToFiles(trajectory);
}

int main(int argc, const char** argv) try {
  using namespace optimization;
  const auto& options{parseOptions(argc, argv)};

  seed = options.seed;

  switch (options.method) {
  case GlobalOptions::Method::kEvolution:
    modelTestEvolution<RepetitiveAllocator<double>>(options);
    break;
  case GlobalOptions::Method::kGrayWolf:
    modelTestGray<RepetitiveAllocator<double>>(options);
    break;
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << "\n";
  return 1;
} catch (...) {
  return 1;
}
