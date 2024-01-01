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
}  // namespace

namespace optimization {
GlobalOptions parseOptions(int argc, const char* argv[]) noexcept {
  namespace po = boost::program_options;

  // clang-format off
  po::options_description desc("Global Options");
  desc.add_options()
      ("help,h", "display help")
      // General options
      ("tmax,t", po::value<double>()->default_value(10.0),
       "Maximum simulation time (default: 10.0)")
      ("dt", po::value<double>()->default_value(0.01),
          "Integration time step (default: 0.01)")
      // Control options
      ("control.number", po::value<std::size_t>()->default_value(30),
          "Number of control parameters (default: 30)")
      ("control.min", po::value<double>()->default_value(-10.0),
          "Minimum control value (default: -10.0)")
      ("control.max", po::value<double>()->default_value(10.0),
          "Maximum control value (default: 10.0)")
      ("control.initial",
          po::value<std::vector<double>>()->multitoken()->default_value({10,10,0}),
          "Initial state values (comma-separated list, default: 10,10,0)")
      ("control.target",
          po::value<std::vector<double>>()->multitoken()->default_value({0, 0, 0}),
          "Target state values (comma-separated list, default: 0,0,0)")
      // Optimization method options
      ("method", po::value<std::string>()->default_value("wolf"),
          "Optimization method (wolf or evolution, default: wolf)")
      ("wolf.num", po::value<int>()->default_value(1000),
          "Number of wolves for Gray Wolf method (default: 1000)")
      ("wolf.best", po::value<int>()->default_value(3),
          "Number of best wolves for Gray Wolf method (default: 3)")
      ("evolution.population", po::value<int>()->default_value(500),
          "Population size for Evolution method (default: 500)")
      ("evolution.mutation", po::value<double>()->default_value(0.1),
          "Mutation rate for Evolution method (default: 0.1)")
      ("evolution.crossover", po::value<double>()->default_value(0.8),
          "Crossover rate for Evolution method (default: 0.8)")
      // Other options
      ("file,f", po::value<std::string>(),
          "Control save file (optimization will start with control saved here and "
          "optimized control will be saved here)")
      ("clear,c", po::bool_switch()->default_value(false),
          "Clear control save file before start (default: false)")
      ("seed,s", po::value<unsigned int>()->default_value(std::random_device{}()),
          "Random seed (type: unsigned int) (default: random)")
      ("printStep", po::value<int>()->default_value(5),
          "Print results every N steps (default: 5)")
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

  try {
    GlobalOptions options;

    // Populate general options
    options.tMax = vm["tMax"].as<double>();
    options.integrationDt = vm["dt"].as<double>();

    // Populate control options
    options.controlOptions.numOfParams = vm["control.number"].as<std::size_t>();
    options.controlOptions.uMin = vm["control.min"].as<double>();
    options.controlOptions.uMax = vm["control.max"].as<double>();
    options.controlOptions.initialState =
        vm["control.initial"].as<std::vector<double>>();
    options.controlOptions.targetState =
        vm["control.target"].as<std::vector<double>>();

    // Populate optimization method options
    const auto& method{vm["method"].as<std::string>()};
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

    options.wolfOpt.wolfNum = vm["wolf.num"].as<int>();
    options.wolfOpt.numBest = vm["wolf.best"].as<int>();

    options.evolutionOpt.populationSize = vm["evolution.population"].as<int>();
    options.evolutionOpt.mutationRate = vm["evolution.mutation"].as<double>();
    options.evolutionOpt.crossoverRate = vm["evolution.crossover"].as<double>();

    // Populate other options
    options.controlSaveFile = vm["file"].as<std::string>();
    options.clearSaveBeforeStart = vm["clear"].as<bool>();

    options.seed = vm["seed"].as<int>();
    options.printStep = vm["printStep"].as<int>();

    if (options.clearSaveBeforeStart) {
      std::ofstream ofs(options.controlSaveFile,
                        std::ofstream::out | std::ofstream::trunc);
    }

    return options;
  } catch (const boost::bad_any_cast& e) {
    std::cout << e.what() << "\n";
    std::exit(1);
  }
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
