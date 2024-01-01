#include "global.h"

#include <algorithm>
#include <cstddef>
#include <execution>
#include <iostream>

namespace optimization {
/**
 * @brief
 *
 * @tparam N number of parameters
 * @tparam Fit fitness function
 * @tparam P number of wolfs in population
 * @tparam B number of best wolfs
 */
template <int N, Regular1OutFunction<StaticTensor<N, double>> Fit, int P = 100,
          int B = 3>
class GrayWolfAlgorithm {
  using Specimen = StaticTensor<N, double>;
  using Population = std::array<Specimen, P>;

 public:
  GrayWolfAlgorithm(Fit fit, double limit)
      : fit_{fit}, limit_{std::abs(limit)} {
  }

  StaticTensor<N, double> solve(int numIterations) {
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
  static std::array<double, std::size_t(2) * B> generateKsi() {
    std::array<double, std::size_t(2) * B> result{};
    std::generate(result.begin(), result.end(),
                  []() -> double { return 2.0 * (Probability::get() - 0.5); });
    return result;
  }

  std::array<Specimen, B> getBest(Population& population) {
    using CalcSpecimen = std::pair<int, double>;
    std::array<CalcSpecimen, P> calcPop{};

    std::transform(std::execution::par_unseq, population.begin(),
                   population.end(), calcPop.begin(),
                   [this](const Specimen& q) -> CalcSpecimen {
                     return {0, fit_(q)};
                   });
    for (int i{0}; i < P; ++i) {
      calcPop[i].first = i;
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
    std::array<StaticTensor<N, double>, P> population;
    std::ranges::generate(population, [this]() -> StaticTensor<N, double> {
      StaticTensor<N, double> chromosome;
      std::ranges::generate(chromosome, [this]() {
        return DoubleGenerator::get() / DoubleGenerator::absLimit() * limit_;
      });
      return chromosome;
    });
    return population;
  }

  Fit fit_;
  double limit_;
};
}  // namespace optimization
