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

// #include "static-tensor.h"

#include <filesystem>
#include <vector>

namespace optimization {
struct GrayWolfOptions {
  int wolfNum;
  int numBest;
};

struct EvolutionOptions {
  int populationSize;
  double mutationRate;
  double crossoverRate;
};

struct ControlOptions {
  std::size_t numOfParams;
  // TODO(novak) may be also accept vector
  double uMin;
  double uMax;
  // TODO(novak) Tensor
  std::vector<double> initialState;
  std::vector<double> targetState;
};

struct CircleData {
  double x;
  double y;
  double r;
};

struct FunctionalOptions {
  double coefTime;
  double coefTerminal;
  double coefObstacle;

  double terminalTolerance;

  std::vector<CircleData> circles;
};

struct GlobalOptions {
  std::string configFile;

  double tMax;
  double integrationDt;
  FunctionalOptions functionalOptions;

  ControlOptions controlOptions;
  std::string controlSaveFile;
  bool clearSaveBeforeStart;

  enum class Method {
    kGrayWolf,
    kEvolution
  } method;
  GrayWolfOptions wolfOpt;
  EvolutionOptions evolutionOpt;
  int iter;

  unsigned int seed;
};

constexpr std::string methodToName(optimization::GlobalOptions::Method method) {
  switch (method) {
  case optimization::GlobalOptions::Method::kGrayWolf:
    return "wolf";
  case optimization::GlobalOptions::Method::kEvolution:
    return "evolution";
  }
}

GlobalOptions parseOptions(int argc, const char** argv,
                           const std::string& configPathFromQt = {}) noexcept;
void writeConfig(const GlobalOptions& options,
                 const std::filesystem::path& file = {"config.ini"});
}  // namespace optimization
