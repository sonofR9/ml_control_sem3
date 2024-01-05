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

#ifdef BUILD_WITH_QT
#include "main-window.h"

#include <QApplication>
#include <QDir>
#include <QSettings>
#include <QStandardPaths>

#include <cstdlib>

// #ifdef WINDOWS
// #include <windows.h>
// #endif

#endif

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

template <template <typename> class Alloc>
void modelTestEvolution(const optimization::GlobalOptions& options) {
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
  Evolution<1000, 1000, Alloc, decltype(adap)> solver(
      adap, paramsCount,
      {.min = options.controlOptions.uMin, .max = options.controlOptions.uMax},
      static_cast<std::size_t>(options.evolutionOpt.populationSize),
      {.mutation = options.evolutionOpt.mutationRate,
       .crossover = options.evolutionOpt.crossoverRate});
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
}

template <template <typename> class Alloc>
void modelTestGray(const optimization::GlobalOptions& options) {
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
  GrayWolfAlgorithm<Alloc, decltype(adap)> solver(
      adap, paramsCount, 10,
      {.populationSize = static_cast<std::size_t>(options.wolfOpt.wolfNum),
       .bestNum = static_cast<std::size_t>(options.wolfOpt.numBest)});
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
}

#ifdef BUILD_WITH_QT
int commonMain(int argc, const char** argv) {
  const auto configsPath{QDir::homePath() + "/" + kAppFolder};
  if (!QDir(configsPath).exists()) {
    if (!QDir().mkdir(configsPath)) {
      std::cerr << std::format(
          "Directory {} does not exist and can not be created.\nPlease create "
          "it manually.\n",
          configsPath.toStdString());
      return 1;
    }
  }
  QSettings settings(configsPath + kConfigPathFile, QSettings::IniFormat);
  const auto configFilePath{settings.value("config_file_path").toString()};

  auto options{parseOptions(argc, argv, configFilePath.toStdString())};
  settings.setValue("config_file_path", QString(options.configFile.c_str()));

  seed = options.seed;

  int qtArgc{1};
  char** qtArgv{new char*[1]};
  qtArgv[0] = const_cast<char*>(argv[0]);

  QApplication app{qtArgc, qtArgv};
  MainWindow window{options};
  window.show();

  const auto res{QApplication::exec()};

  RepetitiveAllocator<double> alloc{};
  alloc.deallocateAll();
  delete[] qtArgv;

  return res;
}
#endif

int main(int argc, char** argvCmd) try {
  using namespace optimization;

  const char** argv = const_cast<const char**>(argvCmd);

#ifdef BUILD_WITH_QT
  return commonMain(argc, argv);
#else
  auto options{parseOptions(argc, argv)};

  seed = options.seed;

  switch (options.method) {
  case GlobalOptions::Method::kEvolution:
    modelTestEvolution<RepetitiveAllocator>(options);
    break;
  case GlobalOptions::Method::kGrayWolf:
    modelTestGray<RepetitiveAllocator>(options);
    break;
  }
  RepetitiveAllocator<double> alloc{};
  alloc.deallocateAll();
  return 0;
#endif
} catch (const std::exception& e) {
  std::cerr << e.what() << "\n";
  return 1;
} catch (...) {
  return 1;
}

// #if defined(WINDOWS) and defined(BUILD_WITH_QT)
// void wcharToCharArr(wchar_t* argv[], char** char_argv, int argc) {
//   for (int i = 0; i < argc; i++) {
//     size_t converted = 0;
//     char_argv[i] = new char[wcslen(argv[i]) + 1];
//     wcstombs_s(&converted, char_argv[i], wcslen(argv[i]) + 1, argv[i],
//                _TRUNCATE);
//   }
// }

// int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
//                    LPSTR lpCmdLine, int nCmdShow) try {
//   int argc = 0;  // Count the arguments
//   wchar_t** argv_wide = CommandLineToArgvW(GetCommandLineW(), &argc);

//   char** char_argv = new char*[argc];
//   wcharToCharArr(argv_wide, char_argv, argc);

//   const auto res{commonMain(argc, const_cast<const char**>(char_argv))};

//   for (int i = 0; i < argc; i++) {
//     delete[] char_argv[i];
//   }
//   delete[] char_argv;
//   LocalFree(argv_wide);
//   return res;
// } catch (const std::exception& e) {
//   std::cerr << e.what() << "\n";
//   return 1;
// } catch (...) {
//   return 1;
// }
// #endif
