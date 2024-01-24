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
#include "global.h"
#include "optimize-wheel-robot.h"
#include "options.h"
#include "tensor.h"
#include "utils.h"

#ifdef BUILD_WITH_QT
#include "main-window.h"

#include <QApplication>
#include <QDir>
#include <QSettings>
#include <QStandardPaths>

#include <cstdlib>

#endif

using namespace optimization;

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

  SharedGenerator::gen.seed(options.seed);

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
