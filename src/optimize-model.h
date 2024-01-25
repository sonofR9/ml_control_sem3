/* Copyright (C) 2023-2024 Novak Alexander
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "allocator.h"
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
  Functional functional{{.time = options.functionalOptions.coefTime,
                         .terminal = options.functionalOptions.coefTerminal,
                         .obstacle = options.functionalOptions.coefObstacle},
                        options.functionalOptions.terminalTolerance,
                        options.functionalOptions.circles};
  const auto adap = [paramsCount, tMax, dt, &functional](
                        const Tensor<double, Alloc<double>>& solverResult) {
    assert((solverResult.size() == paramsCount));
    return functional.operator()<double, Alloc<double>>(solverResult, tMax, dt);
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
      getTrajectoryFromControl<double, Alloc<double>>(best, tMax, dt)};
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
  Functional functional{{.time = options.functionalOptions.coefTime,
                         .terminal = options.functionalOptions.coefTerminal,
                         .obstacle = options.functionalOptions.coefObstacle},
                        options.functionalOptions.terminalTolerance,
                        options.functionalOptions.circles};
  const auto adap = [paramsCount, tMax, dt, &functional](
                        const Tensor<double, Alloc<double>>& solverResult) {
    assert((solverResult.size() == paramsCount));
    return functional.operator()<double, Alloc<double>>(solverResult, tMax, dt);
  };

  GrayWolfAlgorithm<Alloc, decltype(adap), Printer> solver(
      adap, paramsCount,
      {.max = options.controlOptions.uMax, .min = options.controlOptions.uMin},
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
      getTrajectoryFromControl<double, Alloc<double>>(best, tMax, dt)};
  writeTrajectoryToFiles(trajectory);

  return best;
}

extern template Tensor<double, RepetitiveAllocator<double>>
modelTestEvolution<RepetitiveAllocator>(const optimization::GlobalOptions&,
                                        decltype(&coutPrint));
extern template Tensor<double, RepetitiveAllocator<double>>
modelTestGray<RepetitiveAllocator>(const optimization::GlobalOptions&,
                                   decltype(&coutPrint));

extern template Tensor<double, RepetitiveAllocator<double>> modelTestEvolution<
    RepetitiveAllocator, std::function<void(std::size_t, double)>>(
    const optimization::GlobalOptions&,
    std::function<void(std::size_t, double)>);
extern template Tensor<double, RepetitiveAllocator<double>>
modelTestGray<RepetitiveAllocator, std::function<void(std::size_t, double)>>(
    const optimization::GlobalOptions&,
    std::function<void(std::size_t, double)>);

}  // namespace optimization
