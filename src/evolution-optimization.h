#pragma once

#include "global.h"
#include "gray-code.h"

#include <algorithm>
#include <array>
#include <memory>
#include <ranges>
#include <vector>

namespace optimization {
template <uint64_t N, uint64_t D>
void mutateGen(DoubleGrayCode<D>& code, double threshold) {
  if (Probability::get() > threshold) {
    code.changeBit(IntGenerator<N>::get());
  }
}

template <uint64_t D>
void mutateGenSimple(DoubleGrayCode<D>& code, double threshold) {
  if (Probability::get() > threshold) {
    code += DoubleGenerator::get() / 100;
  }
}

template <uint64_t N, uint64_t D>
std::pair<DoubleGrayCode<D>, DoubleGrayCode<D>> crossover(
    DoubleGrayCode<D> lhs, DoubleGrayCode<D> rhs) {
  uint64_t mask = (1ULL << IntGenerator<N>::get()) - 1;

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
 * @tparam N number of parameters of function being optimized (number of steps
 * in piecewise function)
 * @tparam D multiplier for gray code
 * @tparam Z zero for gray code
 * @tparam Fit function which calculates fitness. Should return [1, infinity)
 * (where 1 is best)
 * @tparam P number of individuals in population
 */
template <int N, uint64_t D, uint64_t Z,
          Regular1OutFunction<StaticTensor<N, double>> Fit, int P = 100>
class Evolution {
  using Gray = DoubleGrayCode<D>;
  using Chromosome = StaticTensor<N, Gray>;

 public:
  Evolution(Fit fit, double uMin, double uMax)
      : fit_{fit}, uMin_{uMin}, uMax_{uMax} {
  }

  StaticTensor<N, double> solve(int NumIterations) {
    std::pair<Chromosome, double> best{{}, std::numeric_limits<double>::max()};
    auto population{generatePopulation()};

    for (int i{0}; best.second > 1 + kEps && i < NumIterations; ++i) {
      auto [fitness, minIndex] = evaluatePopulation(*population);
      if (fitness[minIndex] < best.second) {
        best = {(*population)[minIndex], fitness[minIndex]};
      }

      auto newPopulation{
          std::make_unique<std::array<Chromosome, P>>(*population)};
      auto newFitness{fitness};
      (*newPopulation)[0] = best.first;
      for (int i{0}; i < P; ++i) {
        int n1, n2, n3;
        do {
          n1 = IntGenerator<P>::get();
          n2 = IntGenerator<P>::get();
          n3 = IntGenerator<P>::get();
        } while (n1 == n2 || n2 == n3 || n3 == n1);

        int j{n1};
        j = fitness[n2] < fitness[j] ? n2 : j;
        j = fitness[n3] < fitness[j] ? n3 : j;
        (*newPopulation)[i] = (*population)[j];
        newFitness[i] = fitness[j];
      }

      crossoverPopulation(*newPopulation, newFitness, 1 / best.second);
      (*newPopulation)[0] = best.first;

      Evolution::mutatePopulation(*newPopulation, 0.9);
      population = std::move(newPopulation);

      for (auto& chromosome : *population) {
        for (auto& gen : chromosome) {
          if (gen.getDouble() < uMin_) {
            gen = uMin_;
          } else if (gen.getDouble() > uMax_) {
            gen = uMax_;
          }
        }
      }

      std::stringstream ss{};
      ss << "\33[2K\riter " << i << " functional " << best.second;
      std::cout << ss.rdbuf() << std::flush;
    }
    return chromosomeToDoubles(best.first);
  }

 private:
  static StaticTensor<N, double> chromosomeToDoubles(
      const Chromosome& chromosome) {
    StaticTensor<N, double> doubles;
    std::transform(chromosome.cbegin(), chromosome.cend(), doubles.begin(),
                   [](const Gray& code) { return code.getDouble(); });
    return doubles;
  }

  static void mutatePopulation(std::array<Chromosome, P>& population,
                               double threshold) {
    for (auto& individual : population) {
      for (auto& gen : individual) {
        // mutateGen<Z * 2, D>(gen, threshold);
        mutateGenSimple(gen, threshold);
      }
    }
  }

  void crossoverPopulation(std::array<Chromosome, P>& population,
                           std::array<double, P> fitness, double probModifier) {
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
      const auto lhsFit{fitness[lhsIndex]};
      const auto rhsFit{fitness[rhsIndex]};
      const auto prob{std::max(1 / lhsFit, 1 / rhsFit)};
      if (prob > Probability::get() * probModifier) {
        for (int j{0}; j < N; ++j) {
          auto [childLhs, childRhs] = crossover<Z * 2, D>(
              population[lhsIndex][j], population[rhsIndex][j]);
          newPop[i][j] = childLhs;
          newPop[i + 1][j] = childRhs;
        }
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
      fitness.first[i] = fitAdapter(population[i]);
      if (fitness.first[i] < min) {
        fitness.second = i;
        min = fitness.first[i];
      }
    }
    return fitness;
  }

  std::unique_ptr<std::array<Chromosome, P>> generatePopulation() {
    // generate random population (P chromosomes of size N each)
    auto population{std::make_unique<std::array<Chromosome, P>>()};
    std::generate(
        population->begin(), population->end(), [this]() -> Chromosome {
          Chromosome chromosome;
          std::generate(chromosome.begin(), chromosome.end(), [this]() {
            return DoubleGrayCode<D>{(uMax_ + uMin_) / 2 +
                                     DoubleGenerator::get() /
                                         DoubleGenerator::absLimit() *
                                         (uMax_ - uMin_) / 2};
          });
          return chromosome;
        });
    return population;
  }

  double fitAdapter(const StaticTensor<N, DoubleGrayCode<D>>& q) {
    StaticTensor<N, double> qDouble;
    for (int i{0}; i < N; ++i) {
      qDouble[i] = q[i].getDouble();
    }
    return fit_(qDouble);
  }

  Fit fit_;
  double uMin_;
  double uMax_;
};
}  // namespace optimization
