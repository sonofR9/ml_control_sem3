#pragma once

#include "evolution-optimization.h"
#include "functional.h"
#include "global.h"
#include "options.h"
#include "particle-sworm.h"
#include "tensor.h"

#include <cassert>
#include <chrono>
#include <fstream>


namespace optimization {

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

template <template <typename> class Alloc,
          PrintFunction Printer = decltype(&coutPrint)>
Tensor<double, Alloc<double>> modelTestEvolution(
    const optimization::GlobalOptions& options, Printer printer = &coutPrint) {
  const double tMax{options.tMax};
  const double dt{options.integrationDt};
  const int iters{options.iter};
  std::size_t controlStepsCount{options.controlOptions.numOfParams};
  std::size_t paramsCount{controlStepsCount * 2};

  Tensor<double, Alloc<double>> init{};
  if (!options.controlSaveFile.empty() && !options.clearSaveBeforeStart) {
    init = readSaveFile<double, Alloc<double>>(options.controlSaveFile);
    init.resize(paramsCount);
  }

  TimeMeasurer tm("Evolution");

  using namespace two_wheeled_robot;
  const auto adap = [paramsCount, tMax,
                     dt](const Tensor<double, Alloc<double>>& solverResult) {
    assert((solverResult.size() == paramsCount));
    return functional<double, Alloc<double>>(solverResult, tMax, dt);
  };
  Evolution<1000, 1000, Alloc, decltype(adap), Printer> solver(
      adap, paramsCount,
      {.min = options.controlOptions.uMin, .max = options.controlOptions.uMax},
      static_cast<std::size_t>(options.evolutionOpt.populationSize),
      {.mutation = options.evolutionOpt.mutationRate,
       .crossover = options.evolutionOpt.crossoverRate},
      printer);
  if (!init.empty()) {
    solver.setBaseline(init, kMaxDiff);
  }
  const auto best{solver.solve(iters)};

  if (!options.controlSaveFile.empty()) {
    writeToSaveFile(options.controlSaveFile, best);
  }

  const auto trajectory{
      getTrajectoryFromControl<double, Alloc<double>>(best, tMax)};
  writeTrajectoryToFiles(trajectory);

  return best;
}

template <template <typename> class Alloc,
          PrintFunction Printer = decltype(&coutPrint)>
Tensor<double, Alloc<double>> modelTestGray(
    const optimization::GlobalOptions& options, Printer printer = &coutPrint) {
  const double tMax{options.tMax};
  const double dt{options.integrationDt};
  const int iters{options.iter};
  std::size_t controlStepsCount{options.controlOptions.numOfParams};
  std::size_t paramsCount{controlStepsCount * 2};

  Tensor<double, Alloc<double>> init{};
  if (!options.controlSaveFile.empty() && !options.clearSaveBeforeStart) {
    init = readSaveFile<double, Alloc<double>>(options.controlSaveFile);
    init.resize(paramsCount);
  }

  TimeMeasurer tm("gray wolf");

  using namespace two_wheeled_robot;
  const auto adap = [paramsCount, tMax,
                     dt](const Tensor<double, Alloc<double>>& solverResult) {
    assert((solverResult.size() == paramsCount));
    return functional<double, Alloc<double>>(solverResult, tMax, dt);
  };
  GrayWolfAlgorithm<Alloc, decltype(adap), Printer> solver(
      adap, paramsCount, 10,
      {.populationSize = static_cast<std::size_t>(options.wolfOpt.wolfNum),
       .bestNum = static_cast<std::size_t>(options.wolfOpt.numBest)},
      printer);
  if (!init.empty()) {
    solver.setBaseline(init, kMaxDiff);
  }
  const auto best{solver.solve(iters)};

  if (!options.controlSaveFile.empty()) {
    writeToSaveFile(options.controlSaveFile, best);
  }

  const auto trajectory{
      getTrajectoryFromControl<double, Alloc<double>>(best, tMax)};
  writeTrajectoryToFiles(trajectory);

  return best;
}

}  // namespace optimization
