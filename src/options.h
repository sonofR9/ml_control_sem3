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
  std::vector<CircleData> circles;
};

struct GlobalOptions {
  std::string configFile;

  double tMax;
  double integrationDt;
  // TODO(novak) tolerance
  double solutionTolerance;

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
