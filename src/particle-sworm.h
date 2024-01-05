#include "global.h"
#include "tensor.h"

#include <algorithm>
#include <cstddef>
#include <execution>
#include <iostream>

namespace optimization {

// TODO(novak) give template param Alloc => functions uses Alloc<double> etc.?

/**
 * @brief
 *
 * @tparam Fit fitness function. Accepts Tensor like StaticTensor<N, double>
 * @tparam P number of wolfs in population
 * @tparam B number of best wolfs
 */
template <template <typename> class Alloc,
          Regular1OutFunction<Tensor<double, Alloc<double>>> Fit,
          PrintFunction Printer = decltype(&coutPrint)>
class GrayWolfAlgorithm {
  /// @brief StaticTensor<N, double>
  using Specimen = Tensor<double, Alloc<double>>;
  /// @brief StaticTensor<P, Specimen>
  using Population = Tensor<Specimen, Alloc<Specimen>>;

 public:
  struct Parameters {
    std::size_t populationSize;
    std::size_t bestNum;
  };

  /**
   * @brief Construct a new Gray Wolf Algorithm object
   *
   * @param fit
   * @param paramsCount number of parameters of function being optimized (number
   * of steps in piecewise function)
   * @param limit
   */
  GrayWolfAlgorithm(Fit fit, std::size_t paramsCount, double limit,
                    Parameters params, Printer printer = &coutPrint)
      : fit_{fit}, paramsCount_{paramsCount}, limit_{std::abs(limit)},
        populationSize_{params.populationSize}, numBest_{params.bestNum},
        printer_{printer} {
  }

  void setBaseline(Specimen baseline, double maxDifference) noexcept {
    assert((baseline.size() == paramsCount_));
    baseline_ = std::move(baseline);
    maxDifference_ = maxDifference;
  }

  /**
   * @brief
   *
   * @param numIterations
   * @return returns Tensor with shape (N)
   */
  Tensor<double, Alloc<double>> solve(int numIterations) {
    auto generated{GrayWolfAlgorithm::generatePopulation()};
    auto& population{*generated};
    auto best = Tensor<Specimen, Alloc<Specimen>>(numBest_);
    auto ksi = Tensor<double, Alloc<double>>(2 * numBest_);
    for (int i{0}; i < numIterations; ++i) {
      getBest(population, best);

      const double alpha{2.0 * (1 - 1.0 * i / numIterations)};

      for (auto& spec : population) {
        GrayWolfAlgorithm::generateKsi(ksi);
        for (std::size_t j{0}; j < paramsCount_; ++j) {
          double qj{spec[j]};
          double res{0};
          for (std::size_t k{0}; k < numBest_; ++k) {
            res +=
                best[k][j] - (2.0 * ksi[2 * k + 1] - 1) * alpha *
                                 std::abs((2 * ksi[2 * k]) * best[k][j] - qj);
          }
          spec[j] = res / numBest_;
          if (spec[j] > limit_) {
            spec[j] = limit_;
          } else if (spec[j] < -limit_) {
            spec[j] = -limit_;
          }
        }
      }
      population[0] = best[0];

      printer_(i + 1, fit_(best[0]));
    }

    return *std::min_element(population.begin(), population.end(),
                             [this](const auto& lhs, const auto& rhs) {
                               return fit_(lhs) < fit_(rhs);
                             });
  }

 private:
  /**
   * @return shape 2*Best.size()
   */
  void generateKsi(Tensor<double, Alloc<double>>& result) const {
    std::generate(result.begin(), result.end(),
                  []() -> double { return 2.0 * (Probability::get() - 0.5); });
  }

  /**
   * @return shape Best.size()
   */
  void getBest(Population& population,
               Tensor<Specimen, Alloc<Specimen>>& best) {
    using CalcSpecimen = std::pair<int, double>;
    thread_local static auto calcPop =
        Tensor<CalcSpecimen, Alloc<CalcSpecimen>>(populationSize_);
    calcPop.resize(populationSize_);

    std::transform(std::execution::par_unseq, population.begin(),
                   population.end(), calcPop.begin(),
                   [this](const Specimen& q) -> CalcSpecimen {
                     return {0, fit_(q)};
                   });
    for (std::size_t i{0}; i < populationSize_; ++i) {
      calcPop[i].first = i;
    }

    thread_local static auto bestCalc =
        Tensor<CalcSpecimen, Alloc<CalcSpecimen>>(numBest_);
    bestCalc.resize(numBest_);
    std::partial_sort_copy(
        calcPop.begin(), calcPop.end(), bestCalc.begin(), bestCalc.end(),
        [](const CalcSpecimen& lhs, const CalcSpecimen& rhs) {
          return lhs.second < rhs.second;
        });

    for (std::size_t i{0}; i < numBest_; ++i) {
      best[i] = population[bestCalc[i].first];
    }
  }

  std::unique_ptr<Population> generatePopulation() {
    // generate random population (P chromosomes of size N each)
    auto result{std::make_unique<Population>(populationSize_)};
    Population& population{*result};
    std::ranges::generate(population, [this]() -> Specimen {
      auto chromosome = Specimen(paramsCount_);
      if (!baseline_.empty()) {
        std::transform(baseline_.begin(), baseline_.end(), chromosome.begin(),
                       [this](double v) {
                         return v + DoubleGenerator::get() /
                                        DoubleGenerator::absLimit() *
                                        maxDifference_;
                       });
      } else {
        std::ranges::generate(chromosome, [this]() {
          return DoubleGenerator::get() / DoubleGenerator::absLimit() * limit_;
        });
      }
      return chromosome;
    });

    if (!baseline_.empty()) {
      population[0] = baseline_;
    }
    return result;
  }

  Fit fit_;
  std::size_t paramsCount_;

  double limit_;

  std::size_t populationSize_;
  std::size_t numBest_;

  Specimen baseline_{};
  double maxDifference_;

  Printer printer_;
};
}  // namespace optimization
