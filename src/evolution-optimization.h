#pragma once

#include "global.h"
#include "gray-code.h"

#include <algorithm>
#include <array>
#include <random>
#include <vector>

namespace optimization {
constexpr int seed{1};
constexpr double threshold{0.6};

struct DoubleGenerator {
  static int get() {
    static std::mt19937 gen(seed);
    static std::uniform_real_distribution<> dis(-100, 100);
    return dis(gen);
  }
};

template <int D>
struct IntGenerator {
  static int get() {
    static std::mt19937 gen(seed);
    static std::uniform_int_distribution<> dis(0, D - 1);
    return dis(gen);
  }
};

struct Probability {
  static int get() {
    static std::mt19937 gen(seed);
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
  }
};

template <int D>
void mutateGen(DoubleGrayCode<D>& code) {
  if (Probability::get() > threshold) {
    code.changeBit(IntGenerator<D>::get());
  }
}

template <int D>
std::pair<DoubleGrayCode<D>, DoubleGrayCode<D>> crossover(
    DoubleGrayCode<D> lhs, DoubleGrayCode<D> rhs) {
  uint64_t mask = (1ULL << IntGenerator<D>::get()) - 1;

  uint64_t bitsFromNum1 = lhs.getGray() & mask;
  uint64_t bitsFromNum2 = rhs.getGray() & mask;

  lhs &= ~mask;
  rhs &= ~mask;

  lhs |= bitsFromNum2;
  rhs |= bitsFromNum1;

  return {lhs, rhs};
}

/**
 * @brief
 *
 * @tparam N number of steps in piecewise function
 * @tparam D number of decimals after comma
 * @tparam Fit function which calculates fitness. Should return [1, infinity) (1
 * is best)
 * @tparam P number of individuals in population
 */
template <int N, int D, Regular1OutFunction<Vector<N, double>> Fit, int P = 100>
class Evolution {
  using Gray = DoubleGrayCode<D>;
  using Chromosome = Vector<N, Gray>;

 public:
  Evolution(Fit fit) : fit_{fit} {
  }

  Vector<N, double> solve(int NumIterations) {
    std::pair<Chromosome, double> best{{}, std::numeric_limits<double>::max()};
    std::array<Chromosome, P> population{Evolution::generatePopulation()};
    for (int i{0}; i < NumIterations; ++i) {
      auto [fitness, minIndex] = evaluatePopulation(population);
      if (fitness[minIndex] < best.second) {
        best = {population[minIndex], fitness[minIndex]};
      }
      crossoverPopulation(population, fitness);
      population[0] = best.first;
      Evolution::mutatePopulation(population);
    }
    return chromosomeToDoubles(best.first);
  }

 private:
  static Vector<N, double> chromosomeToDoubles(const Chromosome& chromosome) {
    Vector<N, double> doubles;
    std::transform(chromosome.begin(), chromosome.end(), doubles.begin(),
                   [](const Gray& code) { return code.getDouble(); });
    return doubles;
  }

  static void mutatePopulation(std::array<Chromosome, P>& population) {
    for (auto& individual : population) {
      mutateGen(individual);
    }
  }

  void crossoverPopulation(std::array<Chromosome, P>& population,
                           std::array<double, P> fitness) {
    std::array<Chromosome, P> newPop;
    for (int i = 0; i < P; i += 2) {
      const auto lhsIndex{IntGenerator<P>::get()};
      auto rhsIndex{IntGenerator<P>::get()};
      if (rhsIndex == lhsIndex) {
        if (rhsIndex < P - 1) {
          ++rhsIndex;
        } else {
          --rhsIndex;
        }
      }
      const auto lhsFit{fitness(lhsIndex)};
      const auto rhsFit{fitness(rhsIndex)};
      if (std::max(1 / lhsFit, 1 / rhsFit) > Probability::get()) {
        auto [childLhs, childRhs] =
            crossover(population[lhsIndex], population[rhsIndex]);
        newPop[i] = childLhs;
        newPop[i + 1] = childRhs;
      } else {
        newPop[i] = population[lhsIndex];
        newPop[i + 1] = population[rhsIndex];
      }
    }
    population = newPop;
  }
  std::pair<std::array<double, P>, int> evaluatePopulation(
      std::array<Chromosome, P>& population) {
    double min = std::numeric_limits<double>::max();
    std::pair<std::array<double, P>, int> fitness{{}, -1};
    for (int i{0}; i < P; ++i) {
      fitness.first[i] = fit_(population[i]);
      if (fitness.first[i] < min) {
        fitness.second = i;
        min = fitness.first[i];
      }
    }
    return fitness;
  }

  static std::array<Vector<N, Gray>, P> generatePopulation() {
    // generate random population (P chromosomes of size N each)
    std::array<Vector<N, Gray>, P> population;
    std::generate(population.begin(), population.end(),
                  []() -> Vector<N, Gray> {
                    Vector<N, Gray> chromosome;
                    std::generate(chromosome.begin(), chromosome.end(), []() {
                      return DoubleGrayCode<D>{DoubleGenerator::get()};
                    });
                    return chromosome;
                  });
    return population;
  }

  Fit fit_;
};
}  // namespace optimization
