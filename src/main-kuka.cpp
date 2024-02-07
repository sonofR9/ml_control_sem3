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

#include "allocator.h"
#include "gateway-transmission.h"
#include "global.h"
#include "optimize-kuka-robot.h"
#include "options.h"
#include "tensor.h"
#include "utils.h"

#include <chrono>

#ifdef BUILD_WITH_QT
#include "main-window.h"

#include <QApplication>
#include <QDir>
#include <QSettings>
#include <QStandardPaths>

#include <cstdlib>

#endif

using namespace optimization;
using namespace kuka;

constexpr const char* kTrajectoryFilePath{"trajectory.txt"};

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

  SharedGenerator::gen.seed(options.seed);

  int qtArgc{1};
  char** qtArgv{new char*[1]};
  qtArgv[0] = const_cast<char*>(argv[0]);

  QApplication app{qtArgc, qtArgv};
  MainWindow window{options};
  window.show();

  const auto res{QApplication::exec()};

  delete[] qtArgv;

  return res;
}
#endif

int main(int argc, char** argvCmd) try {
  using namespace optimization;
  using namespace std::chrono_literals;

  const char** argv = const_cast<const char**>(argvCmd);

#undef BUILD_WITH_QT
#ifdef BUILD_WITH_QT
  return commonMain(argc, argv);
#else
  std::size_t lastSequentialNumber{0};
  while (true) {
    auto [trajectoryRaw, sequentialNumber] = readDataFromFile<Tensor<
        Tensor<double, RepetitiveAllocator<double> >,
        RepetitiveAllocator<Tensor<double, RepetitiveAllocator<double> > > > >(
        kTrajectoryFilePath);
    Tensor<double, RepetitiveAllocator<double> > trajectory{
        kNumDof * trajectoryRaw.size()};
    for (std::size_t i{0}; i < trajectoryRaw.size(); ++i) {
      for (std::size_t j{0}; j < trajectoryRaw[i].size(); ++j) {
        trajectory[i * kNumDof + j] = trajectoryRaw[i][j];
      }
    }
    if (sequentialNumber == lastSequentialNumber) {
      std::this_thread::sleep_for(1ms);
    }
    auto options{parseOptions(argc, argv)};

    SharedGenerator::gen.seed(options.seed);

    switch (options.method) {
    case GlobalOptions::Method::kEvolution: {
      auto best{
          kuka::modelTestEvolution<RepetitiveAllocator>(options, trajectory)};
      writeDataToFile(kTrajectoryFilePath, best, ++lastSequentialNumber);
      break;
    }
    case GlobalOptions::Method::kGrayWolf: {
      auto best{kuka::modelTestGray<RepetitiveAllocator>(options, trajectory)};
      writeDataToFile(kTrajectoryFilePath, best, ++lastSequentialNumber);
      break;
    }
    }
  }
  return 0;
#endif
} catch (const std::exception& e) {
  std::cerr << e.what() << "\n";
  return 1;
} catch (...) {
  return 1;
}
