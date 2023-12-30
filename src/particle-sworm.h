#include "global.h"

#include <algorithm>
#include <cstddef>
#include <iostream>

namespace optimization {
/**
 * @brief
 *
 * @tparam N number of parameters
 * @tparam Fit fitness function
 * @tparam P number of in population
 * @tparam B number of best wolfs
 */
template <int N, Regular1OutFunction<Tensor<N, double>> Fit, int P = 100,
          int B = 3>
class GrayWolfAlgorithm {
  using Specimen = Tensor<N, double>;
  using Population = std::array<Specimen, P>;

 public:
  GrayWolfAlgorithm(Fit fit, double limit)
      : fit_{fit}, limit_{std::abs(limit)} {
  }

  Tensor<N, double> solve(int numIterations) {
    auto population{GrayWolfAlgorithm::generatePopulation()};

    for (int i{0}; i < numIterations; ++i) {
      const auto& best{getBest(population)};
      // std::cout << best[0] << " fit " << fit_(best[0]) << std::endl;

      const double alpha{2.0 * (1 - 1.0 * i / numIterations)};

      for (auto& spec : population) {
        const auto& ksi{GrayWolfAlgorithm::generateKsi()};
        for (int j{0}; j < N; ++j) {
          double qj{spec[j]};
          double res{0};
          for (int k{0}; k < B; ++k) {
            res += best[k][j] -
                   (2.0 * ksi[2 * k + 1]) *
                       std::abs((2 * ksi[2 * k] - 1) * alpha * best[k][j] - qj);

            // res += best[k][j] -
            //  (2.0 * ksi[2 * k + 1] - 1) *alpha *
            //  std::abs((2 * ksi[2 * k]) * best[k][j] - qj);
          }
          spec[j] = res / B;
          if (spec[j] > limit_) {
            spec[j] = limit_;
          } else if (spec[j] < -limit_) {
            spec[j] = -limit_;
          }
        }
      }

      std::stringstream ss{};
      ss << "\33[2K\riter " << i << " functional " << fit_(best[0]);
      std::cout << ss.rdbuf() << std::flush;
    }

    return *std::min_element(population.begin(), population.end(),
                             [this](const auto& lhs, const auto& rhs) {
                               return fit_(lhs) < fit_(rhs);
                             });
  }

 private:
  static std::array<double, 2ZU * B> generateKsi() {
    std::array<double, 2ZU * B> result{};
    std::generate(result.begin(), result.end(),
                  []() -> double { return 2.0 * (Probability::get() - 0.5); });
    return result;
  }

  std::array<Specimen, B> getBest(Population& population) {
    using CalcSpecimen = std::pair<int, double>;
    std::array<CalcSpecimen, P> calcPop{};
    for (int i{0}; i < P; ++i) {
      calcPop[i].first = i;
      calcPop[i].second = fit_(population[i]);
    }

    std::array<CalcSpecimen, B> best;
    std::partial_sort_copy(
        calcPop.begin(), calcPop.end(), best.begin(), best.end(),
        [](const CalcSpecimen& lhs, const CalcSpecimen& rhs) {
          return lhs.second < rhs.second;
        });

    std::array<Specimen, B> result;
    for (int i{0}; i < B; ++i) {
      result[i] = population[best[i].first];
    }
    return result;
  }

  Population generatePopulation() {
    // generate random population (P chromosomes of size N each)
    std::array<Tensor<N, double>, P> population;
    std::generate(
        population.begin(), population.end(), [this]() -> Tensor<N, double> {
          Tensor<N, double> chromosome;
          std::generate(chromosome.begin(), chromosome.end(), [this]() {
            return DoubleGenerator::get() / DoubleGenerator::absLimit() *
                   limit_;
          });
          return chromosome;
        });
    return population;
  }

  Fit fit_;
  double limit_;
};
}  // namespace optimization
