#include "options.h"

#include "utils.h"

#include <boost/program_options.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <format>
#include <fstream>
#include <iostream>
#include <random>

namespace {
constexpr std::string methodToName(optimization::GlobalOptions::Method method) {
  switch (method) {
  case optimization::GlobalOptions::Method::kGrayWolf:
    return "wolf";
  case optimization::GlobalOptions::Method::kEvolution:
    return "evolution";
  }
}

template <typename T>
constexpr T getValue(const boost::program_options::variables_map& vm,
                     const std::string& name) {
  if (vm.contains(name)) {
    return vm[name].as<T>();
  }
  return {};
}
}  // namespace

namespace optimization {
GlobalOptions parseOptions(int argc, const char** argv) noexcept try {
  namespace po = boost::program_options;

  // clang-format off
  po::options_description desc("Global Options");
  desc.add_options()
      ("help,h", "display help")
      // General options
      ("tmax,t", po::value<double>()->default_value(10.0), "Maximum simulation time")
      ("dt", po::value<double>()->default_value(0.01), "Integration time step")
      // Control options
      ("control.number", po::value<std::size_t>()->default_value(30),
          "Number of control parameters")
      ("control.min", po::value<double>()->default_value(-10.0),
          "Minimum control value")
      ("control.max", po::value<double>()->default_value(10.0),
          "Maximum control value")
      ("control.initial",
          po::value<std::vector<double>>()->multitoken()->default_value({10,10,0}),
          "Initial state values (comma-separated list)")
      ("control.target",
          po::value<std::vector<double>>()->multitoken()->default_value({0, 0, 0}),
          "Target state values (comma-separated list)")
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
          "Random seed (type: unsigned int) (default: random)")
      ("printStep", po::value<int>()->default_value(5),
          "Print results every N steps")
      ("configFile,c", po::value<std::string>(), "Configuration file")
  ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    std::exit(0);
  }

  if (vm.count("configFile") != 0U) {
    const auto& fileName{vm["configFile"].as<std::string>()};
    std::ifstream ifs(fileName);
    if (!ifs.is_open()) {
      std::cout << std::format("Config file '{}' does not exist\n", fileName);
      std::exit(1);
    }

    po::store(po::parse_config_file(ifs, desc), vm);
  }

  po::notify(vm);

  GlobalOptions options;

  // Populate general options
  options.tMax = getValue<double>(vm, "tmax");
  options.integrationDt = getValue<double>(vm, "dt");

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
    std::exit(1);
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
  options.printStep = getValue<int>(vm, "printStep");

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
                 const std::filesystem::path& file) {
  try {
    boost::property_tree::ptree pt;

    // General options
    pt.put("tmax", options.tMax);
    pt.put("dt", options.integrationDt);

    // Control options
    boost::property_tree::ptree controlOptionsPt;
    controlOptionsPt.put("number", options.controlOptions.numOfParams);
    controlOptionsPt.put("min", options.controlOptions.uMin);
    controlOptionsPt.put("max", options.controlOptions.uMax);

    std::stringstream initialSs;
    std::stringstream targetSs;
    std::copy(options.controlOptions.initialState.begin(),
              options.controlOptions.initialState.end(),
              std::ostream_iterator<double>(initialSs, ","));
    std::copy(options.controlOptions.targetState.begin(),
              options.controlOptions.targetState.end(),
              std::ostream_iterator<double>(targetSs, ","));
    controlOptionsPt.put("initial", initialSs.str());
    controlOptionsPt.put("target", targetSs.str());
    pt.add_child("control", controlOptionsPt);

    // Optimization method options
    pt.put("name", methodToName(options.method));

    boost::property_tree::ptree wolfOptionsPt;
    wolfOptionsPt.put("num", options.wolfOpt.wolfNum);
    wolfOptionsPt.put("best", options.wolfOpt.numBest);
    pt.add_child("wolf", wolfOptionsPt);

    boost::property_tree::ptree evolutionsOptionsPt;
    evolutionsOptionsPt.put("population", options.evolutionOpt.populationSize);
    evolutionsOptionsPt.put("mutation", options.evolutionOpt.mutationRate);
    evolutionsOptionsPt.put("crossover", options.evolutionOpt.crossoverRate);
    pt.add_child("evolution", evolutionsOptionsPt);

    pt.put("iter", options.iter);

    // Other options
    pt.put("seed", options.seed);
    pt.put("print.step", options.printStep);
    pt.put("control.savefile", options.controlSaveFile);
    pt.put("clear.savefile", options.clearSaveBeforeStart);

    // Write to file
    boost::property_tree::write_ini(file.string(), pt);

  } catch (const std::exception& e) {
    std::cerr << std::format("Error writing configuration: {}\n", e.what());
  }
}
}  // namespace optimization
