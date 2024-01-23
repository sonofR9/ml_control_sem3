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

#include "options.h"

#include "utils.h"

#include <boost/program_options.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

amespace {
  template <typename T>
  constexpr T getValue(const boost::program_options::variables_map& vm,
                       const std::string& name) {
    if (vm.contains(name)) {
      return vm[name].as<T>();
    }
    return {};
  }

  std::string vectorToCommaSeparatedString(const std::vector<double>& values) {
    std::stringstream stream;
    std::copy(values.begin(), values.end(),
              std::ostream_iterator<double>(stream, ","));
    return stream.str();
  }

  void vectorToSeparateLines(const std::string& fileName, std::string tag) {
    if (tag.back() != '=') {
      tag += '=';
    }
    std::ifstream inputFile(fileName);
    std::stringstream modifiedLines;  // Store modified lines temporarily

    std::string line;
    while (std::getline(inputFile, line)) {
      if (line.starts_with(tag)) {
        // Split the values and write separate lines to the stringstream
        std::istringstream iss(line.substr(tag.size()));  // Skip startingWord
        std::string value;
        while (std::getline(iss, value, ',')) {
          modifiedLines << tag << value << std::endl;
        }
      } else {
        modifiedLines << line << std::endl;  // Copy other lines directly
      }
    }

    inputFile.close();

    std::ofstream outputFile(fileName);
    outputFile << modifiedLines.str();
    outputFile.close();
  }
}  // namespace

namespace optimization {
GlobalOptions parseOptions(int argc, const char** argv,
                           const std::string& configPathFromQt) noexcept try {
  namespace po = boost::program_options;

  // clang-format off
  po::options_description desc("Global Options");
  desc.add_options()
      ("help,h", "display help")
      // General options
      ("tmax,t", po::value<double>()->default_value(10.0), "Maximum simulation time")
      ("dt", po::value<double>()->default_value(0.01), "Integration time step")
      // functional options
      ("functional.coefTime", po::value<double>()->default_value(1), 
          "Time coefficient in functional")
      ("functional.coefTerminal", po::value<double>()->default_value(1),
          "Terminal position coefficient in functional")
      ("functional.coefObstacle", po::value<double>()->default_value(1),
          "Obstacle coefficient in functional")
      ("functional.terminalTolerance", po::value<double>()->default_value(0.1), 
          "Tolerance of terminal solution")
      ("functional.circleX",
          po::value<std::vector<double>>()->multitoken()->default_value({2.5, 7.5}),
          "Circle centers x coordinates (comma-separated list or separate "
          "entries in config file). Should be the same amount as circleY, circleR")
      ("functional.circleY",
          po::value<std::vector<double>>()->multitoken()->default_value({2.5, 7.5}),
          "Circle centers y coordinates (comma-separated list or separate "
          "entries in config file). Should be the same amount as circleX, circleR")
      ("functional.circleR",
          po::value<std::vector<double>>()->multitoken()->default_value({2, 2}),
          "Circle radii (comma-separated list or separate "
          "entries in config file). Should be the same amount as circleX, circleY")
      // Control options
      ("control.number", po::value<std::size_t>()->default_value(30),
          "Number of control parameters")
      ("control.min", po::value<double>()->default_value(-10.0),
          "Minimum control value")
      ("control.max", po::value<double>()->default_value(10.0),
          "Maximum control value")
      ("control.initial",
          po::value<std::vector<double>>()->multitoken()->default_value({10,10,0}),
          "Initial state values (comma-separated list or separate entries in config file)")
      ("control.target",
          po::value<std::vector<double>>()->multitoken()->default_value({0, 0, 0}),
          "Target state values (comma-separated list or separate entries in config file)")
      // Optimization method options
      ("method", po::value<std::string>()->default_value("wolf"),
          "Optimization method (wolf or evolution)")
      ("wolf.num", po::value<int>()->default_value(1000),
          "Number of wolves for Gray Wolf method")
      ("wolf.best", po::value<int>()->default_value(3),
          "Number of best wolves for Gray Wolf method")
      ("evolution.population", po::value<int>()->default_value(500),
          "Population size for Evolution method")
      ("evolution.mutation", po::value<double>()->default_value(0.1),
          "Mutation rate for Evolution method")
      ("evolution.crossover", po::value<double>()->default_value(0.8),
          "Crossover rate for Evolution method")
      ("iter,i", po::value<int>()->default_value(100), "Number of iterations")
      // Other options
      ("file,f", po::value<std::string>(),
          "Control save file (optimization will start with control saved here and "
          "optimized control will be saved here)")
      ("clear", po::bool_switch()->default_value(false),
          "Clear control save file before start")
      ("seed,s", po::value<unsigned int>()->default_value(std::random_device{}()),
          "Random seed (type: unsigned int) (if empty will be random)")
      ("configFile,c", po::value<std::string>(), "Configuration file")
  ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    std::exit(0);
  }

  GlobalOptions options;

  if (vm.count("configFile") != 0U) {
    const auto& fileName{vm["configFile"].as<std::string>()};
    std::ifstream ifs(fileName);
    if (!ifs.is_open()) {
      std::cout << std::format("Config file '{}' does not exist\n", fileName);
      std::exit(1);
    }

    options.configFile = fileName;
    po::store(po::parse_config_file(ifs, desc), vm);
  } else if (!configPathFromQt.empty()) {
    std::ifstream ifs(configPathFromQt);
    if (!ifs.is_open()) {
      std::cout << std::format("Config file '{}' does not exist\n",
                               configPathFromQt);
      std::exit(1);
    }
    options.configFile = configPathFromQt;
    po::store(po::parse_config_file(ifs, desc), vm);
  }

  po::notify(vm);

  // Populate general options
  options.tMax = getValue<double>(vm, "tmax");
  options.integrationDt = getValue<double>(vm, "dt");

  // Populate functional options
  options.functionalOptions.coefTime =
      getValue<double>(vm, "functional.coefTime");
  options.functionalOptions.coefTerminal =
      getValue<double>(vm, "functional.coefTerminal");
  options.functionalOptions.coefObstacle =
      getValue<double>(vm, "functional.coefObstacle");
  options.functionalOptions.terminalTolerance =
      getValue<double>(vm, "functional.terminalTolerance");
  const auto circlesX = getValue<std::vector<double>>(vm, "functional.circleX");
  const auto circlesY = getValue<std::vector<double>>(vm, "functional.circleY");
  const auto circlesR = getValue<std::vector<double>>(vm, "functional.circleR");
  if (circlesX.size() != circlesY.size() ||
      circlesX.size() != circlesR.size()) {
    std::cout << std::format(
        "Circles count does not match. Number of x coordinates provided is {}, "
        "y coordinates is {}, and radii is {}\n",
        circlesX.size(), circlesY.size(), circlesR.size());
    std::exit(1);
  }
  for (std::size_t i = 0; i < circlesX.size(); ++i) {
    options.functionalOptions.circles.emplace_back(circlesX[i], circlesY[i],
                                                   circlesR[i]);
  }

  // Populate control options
  options.controlOptions.numOfParams =
      getValue<std::size_t>(vm, "control.number");
  options.controlOptions.uMin = getValue<double>(vm, "control.min");
  options.controlOptions.uMax = getValue<double>(vm, "control.max");
  options.controlOptions.initialState =
      getValue<std::vector<double>>(vm, "control.initial");
  options.controlOptions.targetState =
      getValue<std::vector<double>>(vm, "control.target");

  // Populate optimization method options
  const auto& method{getValue<std::string>(vm, "method")};
  if (method == methodToName(GlobalOptions::Method::kGrayWolf)) {
    options.method = GlobalOptions::Method::kGrayWolf;
  } else if (method == methodToName(GlobalOptions::Method::kEvolution)) {
    options.method = GlobalOptions::Method::kEvolution;
  } else {
    std::cout << std::format(
        "Invalid method name '{}' was provided.\nValid names: {}, {}", method,
        methodToName(GlobalOptions::Method::kGrayWolf),
        methodToName(GlobalOptions::Method::kEvolution));
    std::exit(2);
  }

  options.wolfOpt.wolfNum = getValue<int>(vm, "wolf.num");
  options.wolfOpt.numBest = getValue<int>(vm, "wolf.best");

  options.evolutionOpt.populationSize =
      getValue<int>(vm, "evolution.population");
  options.evolutionOpt.mutationRate =
      getValue<double>(vm, "evolution.mutation");
  options.evolutionOpt.crossoverRate =
      getValue<double>(vm, "evolution.crossover");

  options.iter = getValue<int>(vm, "iter");

  // Populate other options
  options.controlSaveFile = getValue<std::string>(vm, "file");
  options.clearSaveBeforeStart = getValue<bool>(vm, "clear");

  options.seed = getValue<unsigned int>(vm, "seed");

  if (options.clearSaveBeforeStart) {
    std::ofstream ofs(options.controlSaveFile,
                      std::ofstream::out | std::ofstream::trunc);
  }

  return options;
} catch (const boost::program_options::error& e) {
  std::cout << std::format("Failed to parse config file: {}\n", e.what());
  std::exit(1);
}

void writeConfig(const GlobalOptions& options,
                 const std::filesystem::path& file) try {
  boost::property_tree::ptree pt;

  // General options
  pt.put("tmax", options.tMax);
  pt.put("dt", options.integrationDt);

  pt.put("functional.coefTime", options.functionalOptions.coefTime);
  pt.put("functional.coefTerminal", options.functionalOptions.coefTerminal);
  pt.put("functional.coefObstacle", options.functionalOptions.coefObstacle);
  pt.put("functional.terminalTolerance",
         options.functionalOptions.terminalTolerance);
  std::vector<double> circlesX;
  std::vector<double> circlesY;
  std::vector<double> circlesR;
  for (const auto& circle : options.functionalOptions.circles) {
    circlesX.push_back(circle.x);
    circlesY.push_back(circle.y);
    circlesR.push_back(circle.r);
  }
  pt.put("functional.circleX", vectorToCommaSeparatedString(circlesX));
  pt.put("functional.circleY", vectorToCommaSeparatedString(circlesY));
  pt.put("functional.circleR", vectorToCommaSeparatedString(circlesR));

  pt.put("control.initial",
         vectorToCommaSeparatedString(options.controlOptions.initialState));
  pt.put("control.target",
         vectorToCommaSeparatedString(options.controlOptions.targetState));
  pt.put("control.number", options.controlOptions.numOfParams);
  pt.put("control.min", options.controlOptions.uMin);
  pt.put("control.max", options.controlOptions.uMax);

  // Optimization method options
  pt.put("method", methodToName(options.method));

  pt.put("wolf.num", options.wolfOpt.wolfNum);
  pt.put("wolf.best", options.wolfOpt.numBest);

  pt.put("evolution.population", options.evolutionOpt.populationSize);
  pt.put("evolution.mutation", options.evolutionOpt.mutationRate);
  pt.put("evolution.crossover", options.evolutionOpt.crossoverRate);

  pt.put("iter", options.iter);

  // Other options
  pt.put("seed", options.seed);
  pt.put("file", options.controlSaveFile);
  pt.put("clear", options.clearSaveBeforeStart);

  // Write to file
  boost::property_tree::write_ini(file.string(), pt);
  vectorToSeparateLines(file.string(), "circleX");
  vectorToSeparateLines(file.string(), "circleY");
  vectorToSeparateLines(file.string(), "circleR");
  vectorToSeparateLines(file.string(), "initial");
  vectorToSeparateLines(file.string(), "target");
} catch (const std::exception& e) {
  std::cerr << std::format("Error writing configuration: {}\n", e.what());
}
}  // namespace optimization
