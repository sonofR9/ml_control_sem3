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
  // TODO(novak)
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

struct GlobalOptions {
  std::string configFile;

  double tMax;
  double integrationDt;
  // TODO(novak)
  double solutionTolerance;

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

  int printStep;
};

GlobalOptions parseOptions(int argc, const char** argv,
                           const std::string& configPathFromQt = {}) noexcept;
void writeConfig(const GlobalOptions& options,
                 const std::filesystem::path& file = {"config.ini"});
}  // namespace optimization
