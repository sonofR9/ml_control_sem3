#pragma once

#include "global.h"
#include "gray-code.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <memory>
#include <ranges>
#include <sstream>

namespace optimization {
template <uint64_t N, uint64_t D, uint64_t Z>
void mutateGen(DoubleGrayCode<D, Z>& code, double threshold) {
  if (Probability::get() > threshold) {
    code.changeBit(IntGenerator<N>::get());
  }
}

template <uint64_t D, uint64_t Z>
void mutateGenSimple(DoubleGrayCode<D, Z>& code, double threshold) {
  if (Probability::get() > threshold) {
    code += DoubleGenerator::get() / 100;
  }
}

template <uint64_t N, uint64_t D, uint64_t Z>
std::pair<DoubleGrayCode<D, Z>, DoubleGrayCode<D, Z>> crossover(
    DoubleGrayCode<D, Z> lhs, DoubleGrayCode<D, Z> rhs) {
  uint64_t mask = (1ULL << IntGenerator<N>::get()) - 1;

  uint64_t bitsFromNum1 = lhs.getGray() & mask;
  uint64_t bitsFromNum2 = rhs.getGray() & mask;

  lhs &= ~mask;
  rhs &= ~mask;

  lhs |= bitsFromNum2;
  rhs |= bitsFromNum1;

  return {lhs, rhs};
}

//  @tparam N number of parameters of function being optimized (number of steps
//  in piecewise function)
/**
 * @brief
 * @tparam D multiplier for gray code
 * @tparam Z zero for gray code
 * @tparam Fit function which calculates fitness. Should return [1, infinity)
 * (where 1 is best) accepts Tensor<double> (StaticTensor<N, double>)
 * @tparam P number of individuals in population
 */
template <uint64_t D, uint64_t Z, template <typename> class Alloc,
          Regular1OutFunction<Tensor<double, Alloc<double>>> Fit,
          PrintFunction Printer = decltype(&coutPrint)>
class Evolution {
  using Gray = DoubleGrayCode<D, Z>;
  /// @brief StaticTensor<N, Gray>
  using Chromosome = Tensor<Gray, Alloc<Gray>>;

  using Population = Tensor<Chromosome, Alloc<Chromosome>>;
  using FitnessArr = Tensor<double, Alloc<double>>;

 public:
  struct Limits {
    double min;
    double max;
  };

  struct Rates {
    double mutation;
    double crossover;
  };

  /**
   * @brief Construct a new Evolution object
   *
   * @param fit
   * @param paramsCount number of parameters of function being optimized (number
   * of steps in piecewise function)
   * @param uMin
   * @param uMax
   */
  Evolution(Fit fit, std::size_t paramsCount, Limits limits, std::size_t P,
            Rates rates, Printer printer = &coutPrint)
      : fit_{fit}, paramsCount_{paramsCount}, uMin_{limits.min},
        uMax_{limits.max}, populationSize_{P}, mutationRate_{rates.mutation},
        crossoverRate_{rates.crossover}, printer_{printer} {
  }

  void setBaseline(const Tensor<double, Alloc<double>>& baseline,
                   double maxDifference) noexcept {
    assert((baseline.size() == paramsCount_));
    baseline_ = Chromosome(baseline.size());
    std::transform(baseline.begin(), baseline.end(), baseline_.begin(),
                   [](double v) { return Gray(v); });

    maxDifference_ = maxDifference;
  }

  /**
   * @return Tensor (StaticTensor<N, double>)
   */
  [[nodiscard]] Tensor<double, Alloc<double>> solve(int NumIterations) const {
    std::pair<Chromosome, double> best{{}, std::numeric_limits<double>::max()};
    auto populationPtr{generatePopulation()};
    auto newPopulationPtr{generateEmptyPopulation()};

    for (int i{0}; best.second > 1 + kEps && i < NumIterations; ++i) {
      auto [fitness, minIndex] = evaluatePopulation(*populationPtr);
      if (fitness[minIndex] < best.second) {
        best = {(*populationPtr)[minIndex], fitness[minIndex]};
      }

      auto newFitness = FitnessArr(populationSize_);
      (*newPopulationPtr)[0] = best.first;
      for (std::size_t i{0}; i < populationSize_; ++i) {
        int n1, n2, n3;
        do {
          n1 = VaryingIntGenerator::get(0, populationSize_ - 1);
          n2 = VaryingIntGenerator::get(0, populationSize_ - 1);
          n3 = VaryingIntGenerator::get(0, populationSize_ - 1);
        } while (n1 == n2 || n2 == n3 || n3 == n1);

        int j{n1};
        j = fitness[n2] < fitness[j] ? n2 : j;
        j = fitness[n3] < fitness[j] ? n3 : j;
        (*newPopulationPtr)[i] = (*populationPtr)[j];
        newFitness[i] = fitness[j];
      }

      // *best.second is necessary:
      // crossover happens when (1/ min(fit1, fit2)) * probModifier > random
      // At the start 1/fit->0 => crossover will not happen. With modification
      // equation above closer to 1 * trueProbModifier > random
      crossoverPopulation(*newPopulationPtr, newFitness,
                          crossoverRate_ * best.second, *populationPtr);
      // now new population is stored in populationPtr

      mutatePopulation(*populationPtr);
      for (auto& chromosome : *populationPtr) {
        for (auto& gen : chromosome) {
          if (gen.getDouble() < uMin_) {
            gen = uMin_;
          } else if (gen.getDouble() > uMax_) {
            gen = uMax_;
          }
        }
      }

      (*populationPtr)[0] = best.first;

      printer_(i + 1, best.second);
    }
    return chromosomeToDoubles(best.first);
  }

 private:
  /**
   * @param chromosome
   * @return StaticTensor<N, double>
   */
  static Tensor<double, Alloc<double>> chromosomeToDoubles(
      const Chromosome& chromosome) {
    auto doubles = Tensor<double, Alloc<double>>(chromosome.size());
    std::transform(chromosome.cbegin(), chromosome.cend(), doubles.begin(),
                   [](const Gray& code) { return code.getDouble(); });
    return doubles;
  }

  void mutatePopulation(Population& population) const {
    const auto threshold{1 / mutationRate_};
    for (auto& individual : population) {
      for (auto& gen : individual) {
        mutateGenSimple(gen, threshold);
      }
    }
  }

  void crossoverPopulation(const Population& population,
                           const FitnessArr& fitness, double probModifier,
                           Population& newPop) const {
    // auto newPopPointer{generateEmptyPopulation()};
    for (std::size_t i = 0; i < populationSize_ - 1; i += 2) {
      const auto lhsIndex{VaryingIntGenerator::get(0, populationSize_ - 1)};
      auto rhsIndex{VaryingIntGenerator::get(0, populationSize_ - 1)};
      if (rhsIndex == lhsIndex) {
        if (rhsIndex < populationSize_ - 1) {
          ++rhsIndex;
        } else {
          --rhsIndex;
        }
      }
      const auto lhsFit{fitness[lhsIndex]};
      const auto rhsFit{fitness[rhsIndex]};
      const auto prob{1 / std::min(lhsFit, rhsFit)};
      if (prob * probModifier > Probability::get()) {
        for (std::size_t j{0}; j < paramsCount_; ++j) {
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
  }

  std::pair<FitnessArr, int> evaluatePopulation(
      const Population& population) const {
    double min = std::numeric_limits<double>::max();
    // TODO(novak) preallocate
    std::pair<FitnessArr, int> fitness{FitnessArr(populationSize_), -1};

    std::transform(
        std::execution::par_unseq, population.begin(), population.end(),
        fitness.first.begin(),
        [this](const Chromosome& q) -> double { return fitAdapter(q); });
    for (std::size_t i{0}; i < populationSize_; ++i) {
      if (fitness.first[i] < min) {
        fitness.second = i;
        min = fitness.first[i];
      }
    }

    return fitness;
  }

  std::unique_ptr<Population> generatePopulation() const {
    // generate random population (P chromosomes of size N each)
    // TODO(novak) Tensor => size
    auto population{std::make_unique<Population>(populationSize_)};
    std::generate(
        population->begin(), population->end(), [this]() -> Chromosome {
          auto chromosome = Chromosome(paramsCount_);
          if (!baseline_.empty()) {
            std::transform(baseline_.begin(), baseline_.end(),
                           chromosome.begin(), [this](Gray v) {
                             return v + DoubleGenerator::get() /
                                            DoubleGenerator::absLimit() *
                                            maxDifference_;
                           });
          } else {
            std::generate(chromosome.begin(), chromosome.end(), [this]() {
              return Gray{(uMax_ + uMin_) / 2 +
                          DoubleGenerator::get() / DoubleGenerator::absLimit() *
                              (uMax_ - uMin_) / 2};
            });
          }
          return chromosome;
        });

    if (!baseline_.empty()) {
      (*population)[0] = baseline_;
    }
    return population;
  }

  std::unique_ptr<Population> generateEmptyPopulation() const {
    // generate random population (P chromosomes of size N each)
    // TODO(novak) Tensor => size
    auto population{std::make_unique<Population>(populationSize_)};
    std::generate(population->begin(), population->end(),
                  [this]() -> Chromosome {
                    auto chromosome = Chromosome(paramsCount_);
                    return chromosome;
                  });
    return population;
  }

  double fitAdapter(const Chromosome& q) const {
    assert((q.size() == paramsCount_));
    auto qDouble = Tensor<double, Alloc<double>>(q.size());
    for (std::size_t i{0}; i < paramsCount_; ++i) {
      qDouble[i] = q[i].getDouble();
    }
    return fit_(qDouble);
  }

  Fit fit_;
  const std::size_t paramsCount_;

  const double uMin_;
  const double uMax_;

  const std::size_t populationSize_;
  const double mutationRate_;
  const double crossoverRate_;

  Chromosome baseline_{};
  double maxDifference_;

  Printer printer_;
};
}  // namespace optimization
